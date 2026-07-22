"""
Project an NFL GAME OUTCOME: win / tie / loss, spread, total, team touchdowns.

    python 39_nfl_game_projections.py --home KC --away BUF
    python 39_nfl_game_projections.py --team KC          # auto next game

The football sibling of 14_game_projections.py (with soccer's three-way
outcome, since NFL ties are rare but real). When the trained game models
(37_nfl_train_game_model.py) exist it uses them; before that -- Week 1, or a
fresh setup -- it falls back to the opponent-adjusted team ratings in
nfl_common (strength-of-schedule aware), so it always returns a call and simply
sharpens as the season's results come in.

Everything derives from margin + total + total_td, so nothing contradicts:

    P(home win) = Phi(margin / sigma_margin) · (1 - P(tie))
    home score  = (total + margin) / 2 ;  away score = (total - margin) / 2
    home TDs    = total_td · home_score / (home_score + away_score)
"""

import os
import json
import math
import argparse
from datetime import date, timedelta

import nfl_common as nc


MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Heuristic fallbacks (before the models are trained).
LEAGUE_AVG_TOTAL = 44.0
HEUR_SIGMA = {"margin": 13.0, "total": 10.0, "td": 1.8}
TIE_SCALE = 6.0            # points; how quickly tie prob decays as margin grows
DEFAULT_TIE_RATE = 0.003   # ~0.3% of NFL games end tied


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def confidence_label(confidence):
    if confidence >= 0.65:
        return "STRONG"
    if confidence >= 0.57:
        return "LEAN"
    return "PASS (too close to call)"


upcoming_games = nc.upcoming_games   # re-export for api.py / daily_picks


_model_cache = None


def load_game_models():
    global _model_cache
    if _model_cache is not None:
        return _model_cache or None
    meta_path = os.path.join(MODEL_DIR, "nfl_game_metadata.json")
    if not os.path.exists(meta_path):
        _model_cache = {}
        return None
    try:
        import joblib
        with open(meta_path) as f:
            meta = json.load(f)
        models = {}
        for target in ("margin", "total", "td"):
            p = os.path.join(MODEL_DIR, f"nfl_game_{target}_mean.joblib")
            if not os.path.exists(p):
                continue
            models[target] = {
                "mean": joblib.load(p),
                "q16": joblib.load(os.path.join(MODEL_DIR, f"nfl_game_{target}_q16.joblib")),
                "q84": joblib.load(os.path.join(MODEL_DIR, f"nfl_game_{target}_q84.joblib")),
            }
        _model_cache = {"meta": meta, "models": models} if models else {}
        return _model_cache or None
    except Exception:  # noqa: BLE001
        _model_cache = {}
        return None


def _resolve_matchup(schedule, home, away, game_date, game_id):
    """Fill date/game_id from the schedule's next meeting; trust the schedule on
    which side hosts."""
    if game_date is None:
        for g in schedule:
            if g.get("status") != "upcoming":
                continue
            if {nc.normalize_team(g.get("home_team")), nc.normalize_team(g.get("away_team"))} \
                    == {home, away}:
                game_date = g.get("game_date")
                game_id = game_id or g.get("game_id")
                if nc.normalize_team(g.get("home_team")) != home:
                    home, away = away, home
                break
    return home, away, game_date, game_id


def _feature_row(meta, home, away, schedule, gdate, season_type):
    ratings = nc.team_ratings(schedule)
    hf = nc.team_form(schedule, home)
    af = nc.team_form(schedule, away)
    feats = {
        "home_days_rest": nc.days_rest(schedule, home, gdate),
        "away_days_rest": nc.days_rest(schedule, away, gdate),
        "home_rating": ratings.get(home),
        "away_rating": ratings.get(away),
        "season_type": season_type,
    }
    for prefix, f in (("home", hf), ("away", af)):
        feats[f"{prefix}_season_games"] = f["season_games"]
        feats[f"{prefix}_season_ppg"] = f["season_ppg"]
        feats[f"{prefix}_season_papg"] = f["season_papg"]
        feats[f"{prefix}_season_net"] = f["season_net"]
        feats[f"{prefix}_season_win_pct"] = f["season_win_pct"]
        feats[f"{prefix}_l5_ppg"] = f["l5_ppg"]
        feats[f"{prefix}_l5_papg"] = f["l5_papg"]
    return feats, ratings, hf, af


def _anchors(feats, ratings, home, away):
    rgap = None
    if feats["home_rating"] is not None and feats["away_rating"] is not None:
        rgap = feats["home_rating"] - feats["away_rating"]
    elif feats["home_season_net"] is not None and feats["away_season_net"] is not None:
        rgap = feats["home_season_net"] - feats["away_season_net"]
    a_margin = (rgap if rgap is not None else 0.0) + nc.HFA_POINTS
    if all(feats[k] is not None for k in
           ("home_season_ppg", "home_season_papg", "away_season_ppg", "away_season_papg")):
        a_total = (feats["home_season_ppg"] + feats["home_season_papg"]
                   + feats["away_season_ppg"] + feats["away_season_papg"]) / 2.0
    else:
        a_total = LEAGUE_AVG_TOTAL
    return a_margin, a_total, a_total / 7.0


def project_game(home, away, game_date=None, game_id=None, season_type="auto"):
    home, away = nc.normalize_team(home), nc.normalize_team(away)
    if not home or not away:
        raise ValueError("Unknown team name/abbreviation.")
    schedule = nc.fetch_schedule_rows()
    home, away, game_date, game_id = _resolve_matchup(schedule, home, away, game_date, game_id)
    gdate = nc.parse_date(game_date) or (date.today() + timedelta(days=1))
    if season_type not in ("regular", "playoffs"):
        season_type = "regular"

    cache = load_game_models()
    meta = cache["meta"] if cache else {}
    feats, ratings, hf, af = _feature_row(meta, home, away, schedule, gdate, season_type)
    a_margin, a_total, a_td = _anchors(feats, ratings, home, away)

    method = "heuristic"
    if cache and cache["models"].get("margin") and cache["models"].get("total"):
        import pandas as pd
        X = pd.DataFrame([feats]).reindex(columns=meta["features"])
        for c in meta.get("categorical_features", []):
            X[c] = X[c].astype("category")
        models = cache["models"]
        floors = meta.get("sigma_floor", {})

        def predict(target, anchor, floor_key):
            m = models[target]
            pred = anchor + float(m["mean"].predict(X)[0])
            q16 = anchor + float(m["q16"].predict(X)[0])
            q84 = anchor + float(m["q84"].predict(X)[0])
            sigma = max((q84 - q16) / 2.0, floors.get(floor_key, HEUR_SIGMA[floor_key]))
            return pred, sigma

        margin, sigma_margin = predict("margin", a_margin, "margin")
        total, sigma_total = predict("total", a_total, "total")
        if models.get("td"):
            total_td, sigma_td = predict("td", a_td, "td")
        else:
            total_td, sigma_td = total / 7.0, HEUR_SIGMA["td"]
        method = "model"
    else:
        margin, sigma_margin = a_margin, HEUR_SIGMA["margin"]
        total, sigma_total = a_total, HEUR_SIGMA["total"]
        total_td, sigma_td = a_td, HEUR_SIGMA["td"]

    total = max(total, 0.0)
    total_td = max(total_td, 0.0)
    p_home_raw = normal_cdf(margin / sigma_margin)
    # Ties peak in pick'em games and fade as the projected margin grows. Tiny
    # and honest (NFL ties are ~0.3% overall); the win probs are scaled to leave
    # room for it so the three-way sums to 1.
    tie_rate = meta.get("tie_base_rate", DEFAULT_TIE_RATE) or DEFAULT_TIE_RATE
    p_tie = round(min(0.05, 5.0 * tie_rate * math.exp(-abs(margin) / TIE_SCALE)), 4)
    p_home = round(p_home_raw * (1 - p_tie), 4)
    p_away = round((1 - p_home_raw) * (1 - p_tie), 4)

    home_score = (total + margin) / 2.0
    away_score = (total - margin) / 2.0
    denom = home_score + away_score
    home_td = total_td * home_score / denom if denom else total_td / 2.0
    away_td = total_td - home_td

    winner = home if margin > 0 else away
    confidence = max(p_home, p_away)
    result = {
        "method": method,
        "sport": "nfl",
        "home_team": home,
        "away_team": away,
        "game_date": gdate.isoformat(),
        "game_id": game_id,
        "season_type": season_type,
        "p_home_win": p_home,
        "p_away_win": p_away,
        "p_tie": p_tie,
        "predicted_winner": winner,
        "predicted_outcome": winner,
        "confidence": round(confidence, 4),
        "confidence_label": confidence_label(confidence),
        "projected_margin": round(margin, 1),
        "sigma_margin": round(sigma_margin, 1),
        "projected_total": round(total, 1),
        "sigma_total": round(sigma_total, 1),
        "projected_home_score": round(home_score, 1),
        "projected_away_score": round(away_score, 1),
        "projected_total_td": round(total_td, 1),
        "projected_home_td": round(home_td, 1),
        "projected_away_td": round(away_td, 1),
    }
    result["factors"] = _build_factors(result, feats, ratings, schedule, home, away)
    return result


def _build_factors(r, feats, ratings, schedule, home, away):
    rnd = lambda v, n=1: round(v, n) if isinstance(v, (int, float)) else v
    factors = []
    hr, ar = feats.get("home_rating"), feats.get("away_rating")
    if hr is not None and ar is not None:
        factors.append({
            "title": "Team strength (schedule-adjusted)",
            "value": f"{home} {rnd(hr):+} vs {away} {rnd(ar):+}",
            "detail": "Opponent-adjusted point rating (Simple Rating System). "
                      "Because it already accounts for who each team has played, "
                      "this is the strength-of-schedule–aware starting point.",
        })
    # Who has had / faces the harder slate.
    h_sos = nc.strength_of_schedule(home, schedule)
    a_sos = nc.strength_of_schedule(away, schedule)
    if h_sos["full_sos"] is not None and a_sos["full_sos"] is not None:
        factors.append({
            "title": "Strength of schedule",
            "value": f"{home} SoS {h_sos['full_sos']:+} · {away} SoS {a_sos['full_sos']:+}",
            "detail": "Average opponent rating across the full season (games played "
                      "and still to come). Higher = a tougher road.",
        })
    if feats.get("home_season_net") is not None and feats.get("away_season_net") is not None:
        factors.append({
            "title": "Recent scoring",
            "value": f"{home} net {rnd(feats['home_season_net']):+}, "
                     f"{away} net {rnd(feats['away_season_net']):+}",
            "detail": f"Points scored minus allowed per game — {home} "
                      f"{rnd(feats['home_season_ppg'])}-{rnd(feats['home_season_papg'])}, "
                      f"{away} {rnd(feats['away_season_ppg'])}-{rnd(feats['away_season_papg'])}.",
        })
    hrest, arest = feats.get("home_days_rest"), feats.get("away_days_rest")
    if hrest is not None and arest is not None:
        edge = ("even" if hrest == arest else
                f"{home} rested" if hrest > arest else f"{away} rested")
        factors.append({
            "title": "Rest", "value": f"{home} {hrest}d, {away} {arest}d — {edge}",
            "detail": "Days since each team's last game (short weeks and byes both move it).",
        })
    factors.append({
        "title": "Home field", "value": f"{home} at home",
        "detail": f"Worth about {nc.HFA_POINTS} points, baked into the margin anchor.",
    })
    factors.append({
        "title": "Projection method",
        "value": (f"{r['method'].title()} → {home} {r['projected_home_score']} - "
                  f"{r['projected_away_score']} {away}"),
        "detail": (f"Margin {r['projected_margin']:+} ± {r['sigma_margin']}, total "
                   f"{r['projected_total']} ± {r['sigma_total']}, ~{r['projected_home_td']}/"
                   f"{r['projected_away_td']} team TDs. Win/tie/loss and score are all "
                   f"derived from those, so they agree."),
    })
    return factors


def main():
    parser = argparse.ArgumentParser(description="Project an NFL game outcome.")
    parser.add_argument("--team", help="Team abbrev; auto-detect its next game.")
    parser.add_argument("--home")
    parser.add_argument("--away")
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    if args.team:
        team = nc.normalize_team(args.team)
        nexts = [g for g in nc.upcoming_games(days=60)
                 if nc.same_team(g.get("home_team"), team) or nc.same_team(g.get("away_team"), team)]
        if not nexts:
            raise SystemExit(f"No upcoming game found for {team}.")
        g = nexts[0]
        home, away, gdate, gid = g["home_team"], g["away_team"], g["game_date"], g["game_id"]
    elif args.home and args.away:
        home, away, gdate, gid = args.home, args.away, args.date, None
    else:
        raise SystemExit("Give either --team, or --home and --away.")

    try:
        r = project_game(home, away, game_date=gdate, game_id=gid)
    except (LookupError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")

    print(f"\n{r['away_team']} @ {r['home_team']}  {r['game_date']}  [{r['method']}]")
    print(f"  {r['predicted_winner']} — {max(r['p_home_win'], r['p_away_win'])*100:.0f}% "
          f"({r['confidence_label']})")
    print(f"  Score: {r['home_team']} {r['projected_home_score']} - "
          f"{r['projected_away_score']} {r['away_team']}  "
          f"(total {r['projected_total']}, TDs {r['projected_home_td']}/{r['projected_away_td']})")
    for f in r["factors"]:
        print(f"   • {f['title']}: {f['value']}")


if __name__ == "__main__":
    main()

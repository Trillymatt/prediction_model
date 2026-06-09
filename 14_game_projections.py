"""
Project a GAME OUTCOME: winner probability + projected final score.

    # Auto-detect the next scheduled game for a team:
    python 14_game_projections.py --team NYK

    # Or name the matchup explicitly (home team first):
    python 14_game_projections.py --home NYK --away SAS --date 2026-06-10

The game-level sibling of 09_projections.py. Loads the models trained by
13_train_game_model.py (margin + total, each with mean/q16/q84) and builds the
same as-of features at inference time from the completed games in nba_schedule
-- official final scores, exactly the quantity the training features were
computed from (a team's score in the logs is the sum of its players' points).

Everything is derived from the two model outputs, so nothing can contradict:

    P(home win) = Phi(projected_margin / sigma_margin)
    home score  = (total + margin) / 2,   away score = (total - margin) / 2

Output includes plain-English `factors` cards explaining the call, same shape
as the player engine, so the frontend renders both the same way.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import os
import json
import math
import argparse
from datetime import datetime, date, timedelta

from dotenv import load_dotenv
from supabase import create_client, Client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCHEDULE_TABLE = "nba_schedule"
PAGE_SIZE = 1000
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
REST_CAP = 14            # same cap as training
MIN_TEAM_GAMES = 5       # same gate as training

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit(
        "Missing credentials. Set SUPABASE_URL and SUPABASE_KEY in your .env file."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def current_season() -> str:
    """Return the in-progress NBA season label, e.g. '2025-26' (same as 01-08)."""
    today = datetime.now()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


CURRENT_SEASON = current_season()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def normal_cdf(z: float) -> float:
    """Standard-normal CDF Phi(z) via the error function (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def parse_date(raw):
    """ISO 'YYYY-MM-DD' -> date (None if unparseable)."""
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def season_type_for_game(game_id, game_date) -> str:
    """Playoffs or Regular Season, from the game_id prefix when we have it.

    game_id prefixes: 002 regular season, 004 playoffs, 005 play-in (treated as
    Playoffs intensity). Falls back to the date heuristic 09 uses.
    """
    gid = str(game_id or "")
    if gid.startswith("004") or gid.startswith("005"):
        return "Playoffs"
    if gid.startswith("002"):
        return "Regular Season"
    d = parse_date(game_date)
    if d and (d.month in (5, 6) or (d.month == 4 and d.day >= 19)):
        return "Playoffs"
    return "Regular Season"


def avg(total, count):
    return total / count if count else None


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def fetch_season_schedule() -> list:
    """All of this season's schedule rows, oldest-first, paging past the cap."""
    rows = []
    start = 0
    while True:
        res = (
            supabase.table(SCHEDULE_TABLE)
            .select("game_id,game_date,home_team,away_team,status,home_score,away_score")
            .eq("season", CURRENT_SEASON)
            .order("game_date", desc=False)
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )
        page = res.data or []
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return rows


def upcoming_games(days: int = 10) -> list:
    """Upcoming games in the next `days` days, soonest first."""
    today = date.today().isoformat()
    horizon = (date.today() + timedelta(days=days)).isoformat()
    res = (
        supabase.table(SCHEDULE_TABLE)
        .select("game_id,game_date,home_team,away_team")
        .eq("status", "upcoming")
        .gte("game_date", today)
        .lte("game_date", horizon)
        .order("game_date", desc=False)
        .execute()
    )
    return res.data or []


def team_results(schedule_rows: list, team: str) -> list:
    """This team's completed games (oldest-first) as {scored, allowed, won, date}."""
    out = []
    for g in schedule_rows:
        if g.get("status") != "completed":
            continue
        hs, as_ = g.get("home_score"), g.get("away_score")
        if hs is None or as_ is None:
            continue
        if g.get("home_team") == team:
            scored, allowed = float(hs), float(as_)
        elif g.get("away_team") == team:
            scored, allowed = float(as_), float(hs)
        else:
            continue
        out.append({
            "date": parse_date(g.get("game_date")),
            "scored": scored,
            "allowed": allowed,
            "won": 1.0 if scored > allowed else 0.0,
        })
    return out


def team_features(results: list, prefix: str, game_date) -> dict:
    """The model's feature block for one team, mirroring 12's TeamState reads."""
    n = len(results)
    recent = results[-10:]
    last5 = results[-5:]
    ppg = avg(sum(r["scored"] for r in results), n)
    papg = avg(sum(r["allowed"] for r in results), n)
    rest = None
    if results and game_date:
        rest = min((game_date - results[-1]["date"]).days, REST_CAP)
    return {
        f"{prefix}_days_rest": rest,
        f"{prefix}_season_games": n,
        f"{prefix}_season_ppg": ppg,
        f"{prefix}_season_papg": papg,
        f"{prefix}_season_net": (ppg - papg) if (ppg is not None and papg is not None) else None,
        f"{prefix}_season_win_pct": avg(sum(r["won"] for r in results), n),
        f"{prefix}_l10_ppg": avg(sum(r["scored"] for r in recent), len(recent)),
        f"{prefix}_l10_papg": avg(sum(r["allowed"] for r in recent), len(recent)),
        f"{prefix}_l10_win_pct": avg(sum(r["won"] for r in recent), len(recent)),
        f"{prefix}_l5_ppg": avg(sum(r["scored"] for r in last5), len(last5)),
        f"{prefix}_l5_papg": avg(sum(r["allowed"] for r in last5), len(last5)),
    }


def season_series(schedule_rows: list, home: str, away: str) -> dict:
    """This season's completed meetings between the two teams (incl. playoffs)."""
    home_wins = away_wins = 0
    margins = []   # from `home`'s perspective
    for g in schedule_rows:
        if g.get("status") != "completed":
            continue
        teams = {g.get("home_team"), g.get("away_team")}
        if teams != {home, away}:
            continue
        hs, as_ = g.get("home_score"), g.get("away_score")
        if hs is None or as_ is None:
            continue
        if g.get("home_team") == home:
            margin = hs - as_
        else:
            margin = as_ - hs
        margins.append(margin)
        if margin > 0:
            home_wins += 1
        else:
            away_wins += 1
    return {
        "games": len(margins),
        "home_wins": home_wins,
        "away_wins": away_wins,
        "avg_margin": round(sum(margins) / len(margins), 1) if margins else None,
    }


_model_cache = None


def load_game_models():
    """Load the trained game models + metadata once. None if not trained yet."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache or None
    meta_path = os.path.join(MODEL_DIR, "game_metadata.json")
    if not os.path.exists(meta_path):
        _model_cache = {}
        return None
    try:
        import joblib
        with open(meta_path) as f:
            meta = json.load(f)
        models = {}
        for target in ("margin", "total"):
            models[target] = {
                "mean": joblib.load(os.path.join(MODEL_DIR, f"game_{target}_mean.joblib")),
                "q16": joblib.load(os.path.join(MODEL_DIR, f"game_{target}_q16.joblib")),
                "q84": joblib.load(os.path.join(MODEL_DIR, f"game_{target}_q84.joblib")),
            }
        _model_cache = {"meta": meta, "models": models}
        return _model_cache
    except Exception:  # noqa: BLE001 - any load failure => not available
        _model_cache = {}
        return None


# ---------------------------------------------------------------------------
# The game projection engine
# ---------------------------------------------------------------------------
def build_game_factors(result, series) -> list:
    """Plain-English {title, value, detail} cards explaining the call."""
    home, away = result["home_team"], result["away_team"]
    f = result["features"]
    rnd = lambda v, n=1: round(v, n) if isinstance(v, (int, float)) else v
    factors = []

    # 1) Season strength (the anchor).
    factors.append({
        "title": "Season strength",
        "value": f"{home} net {rnd(f['home_season_net']):+} vs {away} net {rnd(f['away_season_net']):+}",
        "detail": (
            f"Net rating = points scored minus allowed per game this season. "
            f"{home} averages {rnd(f['home_season_ppg'])}-{rnd(f['home_season_papg'])}, "
            f"{away} averages {rnd(f['away_season_ppg'])}-{rnd(f['away_season_papg'])}. "
            f"This gap is the projection's starting point."
        ),
    })

    # 2) Recent form.
    h_l10 = f["home_l10_win_pct"]
    a_l10 = f["away_l10_win_pct"]
    if h_l10 is not None and a_l10 is not None:
        factors.append({
            "title": "Recent form",
            "value": f"{home} {round(h_l10 * 10)}-{10 - round(h_l10 * 10)} last 10, "
                     f"{away} {round(a_l10 * 10)}-{10 - round(a_l10 * 10)}",
            "detail": (
                f"Last-10 scoring: {home} {rnd(f['home_l10_ppg'])}-{rnd(f['home_l10_papg'])}, "
                f"{away} {rnd(f['away_l10_ppg'])}-{rnd(f['away_l10_papg'])}. "
                f"The model adjusts the season baseline toward current form."
            ),
        })

    # 3) Home court.
    factors.append({
        "title": "Home court",
        "value": f"{home} at home",
        "detail": "Home teams win about 55% of NBA games on equal footing; the "
                  "model learned that edge from ~4 seasons of results and applies "
                  "it as part of its adjustment.",
    })

    # 4) Rest.
    hr, ar = f.get("home_days_rest"), f.get("away_days_rest")
    if hr is not None and ar is not None:
        edge = ("even" if hr == ar else
                f"{home} has the rest edge" if hr > ar else f"{away} has the rest edge")
        factors.append({
            "title": "Rest",
            "value": f"{home} {hr}d rest, {away} {ar}d — {edge}",
            "detail": "Days since each team's previous game. Back-to-backs and "
                      "long layoffs both move the model.",
        })

    # 5) Season series (context card; the model sees form, not this head-to-head).
    if series and series["games"]:
        lead = (f"{home} leads {series['home_wins']}-{series['away_wins']}"
                if series["home_wins"] > series["away_wins"]
                else f"{away} leads {series['away_wins']}-{series['home_wins']}"
                if series["away_wins"] > series["home_wins"]
                else f"tied {series['home_wins']}-{series['away_wins']}")
        factors.append({
            "title": "Season series",
            "value": f"{lead} this season",
            "detail": f"Across {series['games']} meeting(s) (playoffs included), "
                      f"average margin {series['avg_margin']:+} for {home}. Shown "
                      f"for context — small samples, so the model leans on full-season form.",
        })

    # 6) Game type + method.
    factors.append({
        "title": "Game type",
        "value": result["season_type"],
        "detail": "The model learned regular-season and playoff scoring patterns "
                  "separately and applies the one for this game.",
    })
    factors.append({
        "title": "Projection method",
        "value": (f"Trained model → {home} {result['projected_home_score']} - "
                  f"{result['projected_away_score']} {away}"),
        "detail": (f"Gradient-boosted models predict the margin "
                   f"({result['projected_margin']:+} ± {result['sigma_margin']}) and the "
                   f"total ({result['projected_total']} ± {result['sigma_total']}); the "
                   f"win probability and score are derived from those, so they always agree."),
    })
    return factors


def project_game(home: str, away: str, game_date=None, game_id=None,
                 season_type: str = "auto") -> dict:
    """Project the outcome of `away` @ `home`: P(win), margin, total, score.

    The single entry point for the API/CLI. Raises LookupError when the teams
    don't have enough completed games this season, or the models aren't trained.
    """
    cache = load_game_models()
    if not cache:
        raise LookupError(
            "Game models not trained yet. Run 12_build_game_training_data.py "
            "then 13_train_game_model.py."
        )
    meta, models = cache["meta"], cache["models"]

    home, away = home.upper().strip(), away.upper().strip()
    schedule = fetch_season_schedule()

    # Resolve date / game_id from the schedule when not given (next meeting).
    if game_date is None:
        for g in schedule:
            if (g.get("status") == "upcoming"
                    and {g.get("home_team"), g.get("away_team")} == {home, away}):
                game_date = g.get("game_date")
                game_id = game_id or g.get("game_id")
                # If the schedule says the other team hosts, trust the schedule.
                if g.get("home_team") != home:
                    home, away = g.get("home_team"), g.get("away_team")
                break
    gdate = parse_date(game_date) or (date.today() + timedelta(days=1))

    home_results = team_results(schedule, home)
    away_results = team_results(schedule, away)
    if len(home_results) < MIN_TEAM_GAMES or len(away_results) < MIN_TEAM_GAMES:
        raise LookupError(
            f"Not enough completed games this season for {home} "
            f"({len(home_results)}) / {away} ({len(away_results)})."
        )

    feats = {}
    feats.update(team_features(home_results, "home", gdate))
    feats.update(team_features(away_results, "away", gdate))
    if season_type in ("Regular Season", "Playoffs"):
        feats["season_type"] = season_type
    else:
        feats["season_type"] = season_type_for_game(game_id, gdate.isoformat())

    import pandas as pd
    X = pd.DataFrame([feats]).reindex(columns=meta["features"])
    for c in meta["categorical_features"]:
        X[c] = X[c].astype("category")

    # Anchors must mirror 13's anchors_for().
    anchor_margin = feats["home_season_net"] - feats["away_season_net"]
    anchor_total = (feats["home_season_ppg"] + feats["home_season_papg"]
                    + feats["away_season_ppg"] + feats["away_season_papg"]) / 2.0

    floors = meta.get("sigma_floor", {})
    margin = anchor_margin + float(models["margin"]["mean"].predict(X)[0])
    m_q16 = anchor_margin + float(models["margin"]["q16"].predict(X)[0])
    m_q84 = anchor_margin + float(models["margin"]["q84"].predict(X)[0])
    sigma_margin = max((m_q84 - m_q16) / 2.0, floors.get("margin", 8.0))

    total = anchor_total + float(models["total"]["mean"].predict(X)[0])
    t_q16 = anchor_total + float(models["total"]["q16"].predict(X)[0])
    t_q84 = anchor_total + float(models["total"]["q84"].predict(X)[0])
    sigma_total = max((t_q84 - t_q16) / 2.0, floors.get("total", 10.0))

    p_home = normal_cdf(margin / sigma_margin)
    home_score = (total + margin) / 2.0
    away_score = (total - margin) / 2.0

    series = season_series(schedule, home, away)
    confidence = max(p_home, 1 - p_home)

    result = {
        "method": "model",
        "home_team": home,
        "away_team": away,
        "game_date": gdate.isoformat(),
        "game_id": game_id,
        "season_type": feats["season_type"],
        "p_home_win": round(p_home, 4),
        "p_away_win": round(1 - p_home, 4),
        "predicted_winner": home if p_home >= 0.5 else away,
        "confidence": round(confidence, 4),
        "confidence_label": confidence_label(confidence),
        "projected_margin": round(margin, 1),
        "sigma_margin": round(sigma_margin, 1),
        "projected_total": round(total, 1),
        "sigma_total": round(sigma_total, 1),
        "projected_home_score": round(home_score, 1),
        "projected_away_score": round(away_score, 1),
        "anchor_margin": round(anchor_margin, 1),
        "anchor_total": round(anchor_total, 1),
        "season_series": series,
        "features": {k: (round(v, 2) if isinstance(v, float) else v)
                     for k, v in feats.items()},
    }
    result["factors"] = build_game_factors(result, series)
    return result


def confidence_label(confidence: float) -> str:
    """Same buckets as the player engine, so the UI language matches."""
    if confidence >= 0.65:
        return "STRONG"
    if confidence >= 0.57:
        return "LEAN"
    return "PASS (too close to call)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def format_report(r: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"{r['away_team']} @ {r['home_team']}   {r['game_date']}  ({r['season_type']})")
    lines.append("-" * 60)
    lines.append(
        f"  WINNER:      {r['predicted_winner']}  "
        f"({max(r['p_home_win'], r['p_away_win']) * 100:.1f}% — {r['confidence_label']})"
    )
    lines.append(
        f"  SCORE:       {r['home_team']} {r['projected_home_score']} - "
        f"{r['projected_away_score']} {r['away_team']}"
    )
    lines.append(
        f"  MARGIN:      {r['projected_margin']:+} (± {r['sigma_margin']})   "
        f"TOTAL: {r['projected_total']} (± {r['sigma_total']})"
    )
    lines.append("-" * 60)
    lines.append("  Why:")
    for f in r.get("factors", []):
        lines.append(f"   • {f['title']}: {f['value']}")
        lines.append(f"       {f['detail']}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Project an NBA game outcome.")
    parser.add_argument("--team", help="Team abbrev; auto-detect its next game.")
    parser.add_argument("--home", help="Home team abbrev (with --away).")
    parser.add_argument("--away", help="Away team abbrev (with --home).")
    parser.add_argument("--date", default=None, help="Game date YYYY-MM-DD (optional).")
    args = parser.parse_args()

    if args.team:
        team = args.team.upper().strip()
        nexts = [g for g in upcoming_games(days=30)
                 if team in (g.get("home_team"), g.get("away_team"))]
        if not nexts:
            raise SystemExit(f"No upcoming game found for {team}.")
        g = nexts[0]
        home, away, gdate, gid = g["home_team"], g["away_team"], g["game_date"], g["game_id"]
    elif args.home and args.away:
        home, away, gdate, gid = args.home, args.away, args.date, None
    else:
        raise SystemExit("Give either --team, or --home and --away.")

    try:
        result = project_game(home, away, game_date=gdate, game_id=gid)
    except (LookupError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")
    print(format_report(result))


if __name__ == "__main__":
    main()

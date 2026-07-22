"""
Project an NFL player's stat for their next game and grade a user's line.

    python 38_nfl_projections.py --player "Patrick Mahomes" --stat pass_yds --line 275.5
    python 38_nfl_projections.py --player "Bijan Robinson" --stat rush_yds --opponent TB
    python 38_nfl_projections.py --player "CeeDee Lamb" --stat any_td

The football sibling of 09_projections.py. Reads the box-score logs
(nfl_player_game_logs) and, when they exist, the trained prop models
(35_nfl_train_props.py). Two engines, one interface:

* TRAINED MODEL (passing/rushing/receiving yards, attempts, receptions,
  targets, TDs, INTs, and combos of them): anchors to the player's recent-form
  rate and predicts the learned adjustment on top, with a per-row spread from
  its quantile models.
* HEURISTIC (the fallback before models are trained, and for markets without a
  model): a recent-form blend nudged by the opponent's defense for that stat.

`any_td` (anytime touchdown) is graded from the Poisson tail on expected
rushing + receiving touchdowns -- the right distribution for a 0.5 TD line.

Either way P(over) = Phi((projection - line) / sigma). Everything degrades
gracefully: no logs yet -> a clear LookupError; no models yet -> the heuristic.
"""

import os
import json
import math
import argparse
import statistics
from datetime import datetime

import nfl_common as nc


CURRENT_SEASON = nc.current_season()
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Single stats the models cover (and the accumulator/feature keys). Must match
# 34/35's STAT_KEYS exactly so inference features line up with training.
MODELED_STATS = (
    "pass_yds", "pass_att", "completions", "pass_td", "interceptions",
    "rush_yds", "rush_att", "rush_td",
    "rec_yds", "receptions", "targets", "rec_td",
)

# Friendly name -> log columns summed per game. Combos sum their components.
STAT_DEFS = {s: [s] for s in MODELED_STATS}
STAT_DEFS.update({
    "pass_rush_yds": ["pass_yds", "rush_yds"],
    "rush_rec_yds": ["rush_yds", "rec_yds"],
    "any_td": ["rush_td", "rec_td"],            # anytime touchdown (Poisson-graded)
    "total_td": ["pass_td", "rush_td", "rec_td"],
})
MODEL_COMBOS = {
    "pass_rush_yds": ["pass_yds", "rush_yds"],
    "rush_rec_yds": ["rush_yds", "rec_yds"],
    "any_td": ["rush_td", "rec_td"],
}
POISSON_STATS = ("any_td",)   # graded from the Poisson tail, not the normal CDF

STAT_NOUNS = {
    "pass_yds": "passing yards", "pass_att": "pass attempts",
    "completions": "completions", "pass_td": "passing TDs",
    "interceptions": "interceptions", "rush_yds": "rushing yards",
    "rush_att": "rush attempts", "rush_td": "rushing TDs",
    "rec_yds": "receiving yards", "receptions": "receptions",
    "targets": "targets", "rec_td": "receiving TDs",
    "pass_rush_yds": "pass + rush yards", "rush_rec_yds": "rush + rec yards",
    "any_td": "anytime touchdown", "total_td": "total touchdowns",
}

WEIGHT_L3, WEIGHT_L5, WEIGHT_SEASON = 0.5, 0.3, 0.2
MATCHUP_DAMPENING = 0.5
SIGMA_WINDOW = 12

LOG_COLUMNS = [
    "game_date", "season", "season_type", "opponent", "home_away",
] + list(MODELED_STATS)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def mean(values):
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def game_value(row, columns):
    return float(sum((row.get(c) or 0) for c in columns))


def search_players(query, limit=10):
    return nc.search_players(query, limit=limit)


# expose for api.py's autocomplete/roster wiring (mirrors engine.supabase usage)
supabase = nc.supabase
PLAYERS_TABLE = nc.PLAYERS_TABLE


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def find_player(name):
    res = (
        nc.supabase.table(nc.PLAYERS_TABLE)
        .select("player_id,player_name,team,position")
        .ilike("player_name", f"%{name}%")
        .execute()
    )
    rows = res.data or []
    if not rows:
        raise LookupError(f"No NFL player matching '{name}'.")
    exact = [r for r in rows if (r.get("player_name") or "").lower() == name.lower()]
    if exact:
        return exact[0]
    if len(rows) > 1:
        names = ", ".join(r.get("player_name", "?") for r in rows[:10])
        raise LookupError(f"'{name}' is ambiguous. Did you mean: {names}")
    return rows[0]


def fetch_player_games(player_id):
    return nc.fetch_all(
        nc.LOGS_TABLE, ",".join(LOG_COLUMNS),
        filters=[("eq", "player_id", player_id)], order_col="game_date",
    )


def next_game_for_team(team_name):
    """(opponent, home_away, game_date) for a team's next upcoming game."""
    if not team_name:
        return None, None, None
    for g in nc.upcoming_games(days=60):
        if nc.same_team(g.get("home_team"), team_name):
            return g.get("away_team"), "HOME", g.get("game_date")
        if nc.same_team(g.get("away_team"), team_name):
            return g.get("home_team"), "AWAY", g.get("game_date")
    return None, None, None


_opp_cache = {"season": None, "value": None}


def _league_and_opponent_allowed(opponent):
    """{stat: (opponent_per_game_allowed, league_avg_allowed)} for this season,
    computed live from the logs (defense = stats conceded to opposing players)."""
    if _opp_cache["season"] != CURRENT_SEASON:
        rows = nc.fetch_all(
            nc.LOGS_TABLE, "opponent,game_id," + ",".join(MODELED_STATS),
            filters=[("eq", "season", CURRENT_SEASON)],
        )
        totals = {s: 0.0 for s in MODELED_STATS}
        team_totals = {}          # team -> {stat: sum}
        team_games = {}           # team -> set(game_id)
        for r in rows:
            team = nc.normalize_team(r.get("opponent"))
            if not team:
                continue
            tt = team_totals.setdefault(team, {s: 0.0 for s in MODELED_STATS})
            team_games.setdefault(team, set()).add(r.get("game_id"))
            for s in MODELED_STATS:
                v = r.get(s) or 0
                tt[s] += v
                totals[s] += v
        total_team_games = sum(len(g) for g in team_games.values()) or 1
        league = {s: totals[s] / total_team_games for s in MODELED_STATS}
        _opp_cache.update(season=CURRENT_SEASON,
                          value={"team_totals": team_totals, "team_games": team_games,
                                 "league": league})
    data = _opp_cache["value"]
    opp = nc.normalize_team(opponent) if opponent else None
    out = {}
    for s in MODELED_STATS:
        lg = data["league"][s]
        opp_pg = None
        if opp and opp in data["team_totals"]:
            n = len(data["team_games"][opp]) or 1
            opp_pg = data["team_totals"][opp][s] / n
        out[s] = (opp_pg, lg)
    return out


# ---------------------------------------------------------------------------
# Trained models
# ---------------------------------------------------------------------------
_model_cache = None


def load_models():
    global _model_cache
    if _model_cache is not None:
        return _model_cache or None
    meta_path = os.path.join(MODEL_DIR, "nfl_metadata.json")
    if not os.path.exists(meta_path):
        _model_cache = {}
        return None
    try:
        import joblib
        with open(meta_path) as f:
            meta = json.load(f)
        models = {}
        for s in MODELED_STATS:
            p = os.path.join(MODEL_DIR, f"nfl_{s}_mean.joblib")
            if not os.path.exists(p):
                continue
            models[s] = {
                "mean": joblib.load(p),
                "q16": joblib.load(os.path.join(MODEL_DIR, f"nfl_{s}_q16.joblib")),
                "q84": joblib.load(os.path.join(MODEL_DIR, f"nfl_{s}_q84.joblib")),
            }
        _model_cache = {"meta": meta, "models": models} if models else {}
        return _model_cache or None
    except Exception:  # noqa: BLE001 - any load failure => heuristic
        _model_cache = {}
        return None


def _form(played, season_games, col):
    allv = [game_value(g, [col]) for g in played]
    return mean(allv[-3:]), mean(allv[-5:]), mean([game_value(g, [col]) for g in season_games])


def _model_feature_row(position, played, season_games, opponent, home_away,
                       next_date, season_type, allowed_ctx):
    cache = load_models()
    if not cache:
        return None
    meta = cache["meta"]
    feats, anchors = {}, {}
    for s in MODELED_STATS:
        l3, l5, season = _form(played, season_games, s)
        feats[f"l3_{s}"], feats[f"l5_{s}"], feats[f"season_{s}"] = l3, l5, season
        anchors[s] = season if season is not None else (l5 if l5 is not None else l3)
        feats[f"opp_{s}_allowed"] = allowed_ctx[s][0]
    feats["home"] = 1 if home_away == "HOME" else 0
    feats["season_games_todate"] = len(season_games)
    feats["opp_games_todate"] = None
    feats["position"] = position or "UNK"
    feats["season_type"] = season_type if season_type in ("regular", "playoffs") else "regular"

    import pandas as pd
    X = pd.DataFrame([feats]).reindex(columns=meta["features"])
    for c in meta["categorical_features"]:
        X[c] = X[c].astype("category")
    return {"X": X, "anchors": anchors, "meta": meta}


def model_projection(components, position, played, season_games, opponent,
                     home_away, next_date, season_type, allowed_ctx, empirical_sigma):
    cache = load_models()
    if not cache or any(c not in cache["models"] for c in components):
        return None
    fr = _model_feature_row(position, played, season_games, opponent, home_away,
                            next_date, season_type, allowed_ctx)
    if fr is None or any(fr["anchors"][c] is None for c in components):
        return None
    X, anchors, models = fr["X"], fr["anchors"], cache["models"]
    floors = fr["meta"].get("sigma_floors", {})
    default_floor = fr["meta"].get("default_sigma_floor", 1.0)

    total_anchor = sum(anchors[c] for c in components)
    projection = max(sum(anchors[c] + float(models[c]["mean"].predict(X)[0])
                         for c in components), 0.0)
    if len(components) == 1:
        c = components[0]
        q16 = anchors[c] + float(models[c]["q16"].predict(X)[0])
        q84 = anchors[c] + float(models[c]["q84"].predict(X)[0])
        sigma = max((q84 - q16) / 2.0, floors.get(c, default_floor))
    else:
        sigma = empirical_sigma
    return {"projection": round(projection, 2),
            "sigma": round(sigma, 2) if sigma else None,
            "anchor": round(total_anchor, 2)}


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------
def build_factors(result, stat, opponent, home_away, model_out, l3, l5, season_avg,
                  opp_allowed, league_allowed):
    noun = STAT_NOUNS.get(stat, stat)
    rnd = lambda v: round(v, 1) if isinstance(v, (int, float)) else v
    factors = [{
        "title": "Recent form",
        "value": f"{rnd(result['projection'])} {noun} projected",
        "detail": f"Baseline from recent games — L3 {rnd(l3)}, L5 {rnd(l5)}, "
                  f"season {rnd(season_avg)}.",
    }]
    if opponent and opp_allowed is not None and league_allowed:
        verdict = ("an EASIER matchup — they give up more than average"
                   if opp_allowed > league_allowed
                   else "a TOUGHER matchup — they give up less than average")
        factors.append({
            "title": "Opponent defense",
            "value": f"{opponent} allows {rnd(opp_allowed)} {noun}/g",
            "detail": f"League average is {rnd(league_allowed)}. That's {verdict}.",
        })
    if home_away:
        factors.append({
            "title": "Location",
            "value": f"Playing {'at home' if home_away == 'HOME' else 'on the road'}",
            "detail": "Home/road context for the matchup.",
        })
    factors.append({
        "title": "Projection method",
        "value": (f"Trained model → {result['projection']} (± {result['sigma']})"
                  if result.get("method") == "model"
                  else f"Heuristic → {result['projection']} (± {result['sigma']})"),
        "detail": ("A gradient-boosted model adjusts the recent-form rate by the "
                   "matchup, home/away and season stage."
                   if result.get("method") == "model"
                   else "No trained model for this market yet, so a recent-form blend "
                        "with an opponent-defense nudge is used."),
    })
    return factors


def confidence_label(confidence):
    if confidence >= 0.65:
        return "STRONG"
    if confidence >= 0.57:
        return "LEAN"
    return "PASS (too close to call)"


# ---------------------------------------------------------------------------
# The projection engine
# ---------------------------------------------------------------------------
def project_player(player_name, stat, line=None, opponent=None, home_away=None,
                   season_type="auto"):
    stat = stat.lower()
    if stat not in STAT_DEFS:
        raise ValueError(f"Unknown stat '{stat}'. Choose from: {', '.join(sorted(STAT_DEFS))}")
    columns = STAT_DEFS[stat]

    player = find_player(player_name)
    player_id = player["player_id"]
    position = player.get("position")

    games = fetch_player_games(player_id)
    if not games:
        raise LookupError(f"No game logs found for {player['player_name']}. "
                          f"Backfill nfl_player_game_logs (see NFL_SETUP.md).")
    played = games
    season_games = [g for g in played if str(g.get("season")) == CURRENT_SEASON]
    form_games = season_games if len(season_games) >= 3 else played

    per_game = [game_value(g, columns) for g in form_games]
    l3, l5 = mean(per_game[-3:]), mean(per_game[-5:])
    season_avg = mean([game_value(g, columns) for g in season_games]) or mean(per_game)

    # Resolve matchup.
    next_date = None
    if opponent is None and home_away is None:
        opponent, home_away, next_date = next_game_for_team(player.get("team"))
    opponent = nc.normalize_team(opponent) if opponent else None
    season_type = season_type if season_type in ("regular", "playoffs") else "regular"

    # Heuristic base: weighted recent-form blend.
    parts = [(l3, WEIGHT_L3), (l5, WEIGHT_L5), (season_avg, WEIGHT_SEASON)]
    parts = [(v, w) for v, w in parts if v is not None]
    total_w = sum(w for _, w in parts)
    projection = sum(v * w for v, w in parts) / total_w if total_w else (mean(per_game) or 0.0)

    allowed_ctx = _league_and_opponent_allowed(opponent)
    # Matchup nudge (heuristic only): opponent defense on this stat vs league.
    opp_allowed = league_allowed = None
    comp_opp = [allowed_ctx[c][0] for c in columns if allowed_ctx.get(c)]
    comp_lg = [allowed_ctx[c][1] for c in columns if allowed_ctx.get(c)]
    if all(v is not None for v in comp_opp) and comp_opp:
        opp_allowed, league_allowed = sum(comp_opp), sum(comp_lg)
        if league_allowed:
            projection *= 1.0 + MATCHUP_DAMPENING * (opp_allowed / league_allowed - 1.0)

    recent = per_game[-SIGMA_WINDOW:]
    sigma = statistics.pstdev(recent) if len(recent) >= 2 else None

    # Model override where every component has a trained model.
    method = "heuristic"
    components = [stat] if stat in MODELED_STATS else MODEL_COMBOS.get(stat)
    model_out = None
    if components:
        model_out = model_projection(components, position, played, season_games,
                                     opponent, home_away, next_date, season_type,
                                     allowed_ctx, sigma)
    if model_out is not None:
        projection = model_out["projection"]
        sigma = model_out["sigma"] if model_out["sigma"] is not None else sigma
        method = "model"

    result = {
        "method": method,
        "sport": "nfl",
        "player_name": player["player_name"],
        "team": player.get("team"),
        "position": position,
        "stat": stat,
        "opponent": opponent,
        "home_away": home_away,
        "games_used": len(form_games),
        "l3": round(l3, 2) if l3 is not None else None,
        "l5": round(l5, 2) if l5 is not None else None,
        "season_avg": round(season_avg, 2) if season_avg is not None else None,
        "season_type": season_type,
        "projection": round(projection, 2),
        "sigma": round(sigma, 2) if sigma is not None else None,
        "injury": nc.injury_status(player_id),
        "line": line,
    }
    # L5/L10-style splits the frontend card expects (l10 == season here).
    result["l10"] = result["l5"]
    result["factors"] = build_factors(
        result, stat, opponent, home_away, model_out, l3, l5, season_avg,
        opp_allowed, league_allowed,
    )

    # Grade the line.
    if line is not None:
        if stat in POISSON_STATS:
            lam = max(projection, 1e-6)
            p_over = 1.0 - math.exp(-lam)      # P(>=1 TD) for a 0.5 line
            if line and line >= 1.5:            # P(>=2), etc. (rare)
                k = int(math.floor(line)) + 1
                cum = sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k))
                p_over = 1.0 - cum
            p_under = 1.0 - p_over
            pick = "OVER" if p_over >= 0.5 else "UNDER"
            confidence = max(p_over, p_under)
            result.update({
                "p_over": round(p_over, 4), "p_under": round(p_under, 4),
                "recommendation": pick, "confidence": round(confidence, 4),
                "confidence_label": confidence_label(confidence),
            })
        elif sigma:
            z = (projection - line) / sigma
            p_over = normal_cdf(z)
            pick = "OVER" if p_over >= 0.5 else "UNDER"
            confidence = max(p_over, 1 - p_over)
            result.update({
                "p_over": round(p_over, 4), "p_under": round(1 - p_over, 4),
                "recommendation": pick, "confidence": round(confidence, 4),
                "confidence_label": confidence_label(confidence),
            })
        else:
            result["note"] = "Not enough history to estimate a spread; projection only."
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Project an NFL player's stat.")
    parser.add_argument("--player", required=True)
    parser.add_argument("--stat", required=True,
                        help=f"One of: {', '.join(sorted(STAT_DEFS))}")
    parser.add_argument("--line", type=float, default=None)
    parser.add_argument("--opponent", default=None)
    loc = parser.add_mutually_exclusive_group()
    loc.add_argument("--home", action="store_true")
    loc.add_argument("--away", action="store_true")
    args = parser.parse_args()

    home_away = "HOME" if args.home else ("AWAY" if args.away else None)
    try:
        r = project_player(args.player, args.stat, line=args.line,
                           opponent=args.opponent, home_away=home_away)
    except (LookupError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")

    tag = "trained model" if r["method"] == "model" else "heuristic"
    print(f"\n{r['player_name']} ({r.get('team')}, {r.get('position')})  [{tag}]")
    print(f"  {r['stat']}: {r['projection']} ± {r['sigma']}")
    for f in r["factors"]:
        print(f"   • {f['title']}: {f['value']}")
    if r.get("recommendation"):
        print(f"  >>> {r['recommendation']} {r['line']} "
              f"({r['confidence'] * 100:.1f}% — {r['confidence_label']})")


if __name__ == "__main__":
    main()

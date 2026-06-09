"""
Project a player's stat for their next game and grade a user-supplied line.

    # Auto-detect the next game from nba_schedule, grade a points line:
    python 09_projections.py --player "LeBron James" --stat points --line 24.5

    # Force the opponent / location (e.g. offseason, or "what if"):
    python 09_projections.py --player "Anthony Edwards" --stat pra --line 44.5 \
        --opponent DEN --home

    # Just see the projection (no line to grade):
    python 09_projections.py --player "Nikola Jokic" --stat rebounds

This is the "brain" of the tool. The user brings their OWN line from whatever
book they use (DraftKings, FanDuel, PrizePicks, ...) -- we don't scrape odds. We
read everything 01-08 loaded into Supabase, build a projection, and report how
confident the data is that the real result lands over/under that line.

Two engines, same interface
----------------------------
* TRAINED MODEL (points / rebounds / assists): if models/ has been built by
  10_build_training_data.py + 11_train_model.py, those stats use the gradient-
  boosted model. It anchors to the recent-form baseline and predicts the learned
  adjustment on top, with a per-prediction spread from its quantile models.
* HEURISTIC (everything else + combos, and the fallback when no model exists):
    base      = weighted blend of recent form and season (L5 50% / L10 30% / season 20%)
    location  = a nudge toward the player's home/away split
    pace      = opponent possessions vs league avg -- lifts/sinks ALL stats (dampened)
    matchup   = opponent's points-allowed-to-position vs league avg (points stats, dampened)
    H2H       = blend toward this player's history vs THIS opponent, weighted by sample
    sigma     = the player's real game-to-game standard deviation

Either way: P(over) = Phi((projection - line) / sigma).

This is a confidence/research tool, not a guarantee. Variance is real; the output
shows the projection, the spread, and the probability honestly.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import os
import json
import math
import argparse
import statistics
from datetime import datetime

from dotenv import load_dotenv
from supabase import create_client, Client

from nba_api.stats.static import teams


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOGS_TABLE = "nba_player_game_logs"
PLAYERS_TABLE = "nba_players"
SCHEDULE_TABLE = "nba_schedule"
DEF_TABLE = "nba_defensive_ratings"
TEAM_STATS_TABLE = "nba_team_stats"
H2H_TABLE = "nba_head_to_head"
INJURIES_TABLE = "nba_injuries"
PAGE_SIZE = 1000          # rows per Supabase select page (PostgREST default cap)

# Trained models (10/11). When present, points/rebounds/assists use the model
# directly, and combos made purely of those (PRA/PR/PA/RA) sum the component
# model projections. Everything else (steals/blocks/threes/turnovers/stocks)
# falls back to the heuristic below.
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODELED_STATS = ("points", "rebounds", "assists")
MODEL_COMBOS = {
    "pra": ["points", "rebounds", "assists"],
    "pr": ["points", "rebounds"],
    "pa": ["points", "assists"],
    "ra": ["rebounds", "assists"],
}
POSITION_NAMES = {"G": "guards", "F": "forwards", "C": "centers", "Unknown": "this position"}
STAT_NOUNS = {
    "points": "points", "rebounds": "rebounds", "assists": "assists",
    "threes": "threes", "steals": "steals", "blocks": "blocks",
    "turnovers": "turnovers", "pra": "pts+reb+ast", "pr": "pts+reb",
    "pa": "pts+ast", "ra": "reb+ast", "stocks": "stl+blk",
    "fgm": "field goals made", "fga": "field goal attempts",
    "ftm": "free throws made", "fta": "free throw attempts",
    "threes_attempted": "three-point attempts",
    "fouls": "personal fouls", "oreb": "offensive rebounds",
    "dreb": "defensive rebounds",
}

# Form-blend weights. Normalized at runtime over whichever windows have data, so
# a player with only 3 games this season still gets a sensible base.
WEIGHT_L5 = 0.50
WEIGHT_L10 = 0.30
WEIGHT_SEASON = 0.20

# How hard the opponent's defense-vs-position is allowed to move the projection.
# 1.0 would apply the full ratio; 0.5 halves it so one matchup number can't
# swing a points projection by 40%. (Points-bearing stats only -- it's the only
# position-specific defensive metric we store.)
MATCHUP_DAMPENING = 0.50

# How hard the opponent's pace moves the projection. Pace drives possessions,
# which lifts/sinks ALL counting stats, so this applies to every stat.
PACE_DAMPENING = 0.50

# Head-to-head (this player's history vs THIS opponent) blend. The weight ramps
# up with sample size -- a 2-game sample barely counts, but once a player has
# H2H_FULL_WEIGHT_GAMES meetings the blend reaches H2H_MAX_WEIGHT.
H2H_MAX_WEIGHT = 0.35
H2H_FULL_WEIGHT_GAMES = 6

# How much of the projection leans on the relevant home/away split (the rest
# stays on the overall form blend).
LOCATION_WEIGHT = 0.30

# Games used to estimate game-to-game spread (sigma). Recent enough to reflect a
# player's current role, long enough to be stable.
SIGMA_WINDOW = 15

# Maps a stat component to its nba_head_to_head column. Only points/rebounds/
# assists have stored H2H splits, so combos made purely of those (pra, pr, ...)
# get an H2H blend; stats involving steals/blocks/threes/turnovers do not.
H2H_COLUMN = {
    "points": "avg_points",
    "rebounds": "avg_rebounds",
    "assists": "avg_assists",
}

# Friendly stat name -> the game-log columns summed per game to produce it.
# Combos (pra, pr, ...) just sum their components within each game.
STAT_DEFS = {
    "points": ["points"],
    "rebounds": ["rebounds"],
    "assists": ["assists"],
    "steals": ["steals"],
    "blocks": ["blocks"],
    "turnovers": ["turnovers"],
    "threes": ["three_made"],
    "threes_attempted": ["three_attempted"],
    "fgm": ["fg_made"],
    "fga": ["fg_attempted"],
    "ftm": ["ft_made"],
    "fta": ["ft_attempted"],
    "pra": ["points", "rebounds", "assists"],
    "pr": ["points", "rebounds"],
    "pa": ["points", "assists"],
    "ra": ["rebounds", "assists"],
    "stocks": ["steals", "blocks"],
}

# Stats whose game-log columns don't exist in Supabase yet (need `ALTER TABLE
# nba_player_game_logs ADD COLUMN fouls int, ADD COLUMN oreb int, ADD COLUMN
# dreb int;` + a historical re-pull with 01 --full). Probed once at import:
# the moment the columns exist, these stats appear in STAT_DEFS / the API
# automatically -- no code change needed.
OPTIONAL_STAT_DEFS = {
    "fouls": ["fouls"],
    "oreb": ["oreb"],
    "dreb": ["dreb"],
}

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit(
        "Missing credentials. Set SUPABASE_URL and SUPABASE_KEY in your .env file."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Game-log columns fetched for every projection (all single-stat components).
LOG_COLUMNS = [
    "game_date", "season", "season_type", "opponent", "home_away",
    "minutes_played", "points", "rebounds", "assists", "steals", "blocks",
    "turnovers", "three_made", "three_attempted", "fg_made", "fg_attempted",
    "ft_made", "ft_attempted",
]


def _enable_optional_stats():
    """Activate fouls/oreb/dreb if their columns exist in Supabase (see above)."""
    optional_cols = sorted({c for cols in OPTIONAL_STAT_DEFS.values() for c in cols})
    try:
        supabase.table(LOGS_TABLE).select(",".join(optional_cols)).limit(1).execute()
    except Exception:  # noqa: BLE001 - columns absent => stats stay off
        return
    STAT_DEFS.update(OPTIONAL_STAT_DEFS)
    LOG_COLUMNS.extend(optional_cols)


_enable_optional_stats()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def current_season() -> str:
    """Return the in-progress NBA season label, e.g. '2025-26'.

    Same Oct-rollover logic as 01-08 so every script agrees on "current".
    """
    today = datetime.now()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


CURRENT_SEASON = current_season()


def season_type_for_date(iso_date) -> str:
    """Guess Regular Season vs Playoffs from a game date.

    The NBA regular season runs ~Oct through mid-April; the play-in + playoffs run
    from ~April 19 into June. So a date from April 19 onward (through June) is
    treated as Playoffs, everything else as Regular Season. Approximate, but it
    needs no schema change -- the schedule table doesn't store season_type yet.
    """
    try:
        d = datetime.strptime(str(iso_date)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return "Regular Season"
    if d.month in (5, 6) or (d.month == 4 and d.day >= 19):
        return "Playoffs"
    return "Regular Season"


def resolve_season_type(season_type, next_date) -> str:
    """Turn an explicit choice or 'auto' into a concrete season type.

    'auto' derives it from the upcoming game's date (falling back to today when
    we don't know the date, e.g. a manually-entered opponent in the offseason).
    """
    if season_type in ("Regular Season", "Playoffs"):
        return season_type
    return season_type_for_date(next_date or datetime.now().date().isoformat())


def mean(values):
    """Mean of the non-None values, or None if there are none."""
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def normal_cdf(z: float) -> float:
    """Standard-normal CDF Phi(z) via the error function (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def position_bucket(position) -> str:
    """Normalize a position string to G / F / C (same buckets as 05)."""
    if not position:
        return "Unknown"
    primary = position.split("-")[0].strip().lower()
    return {"guard": "G", "forward": "F", "center": "C"}.get(primary, "Unknown")


def game_value(row: dict, columns) -> float:
    """Sum the requested stat columns for one game (None treated as 0)."""
    return float(sum((row.get(c) or 0) for c in columns))


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def find_player(name: str):
    """Look up a player by (case-insensitive, partial) name in nba_players.

    Returns a single {player_id, player_name, team, position} dict. Raises
    LookupError with the candidates if the name is ambiguous or absent.
    """
    res = (
        supabase.table(PLAYERS_TABLE)
        .select("player_id,player_name,team,position")
        .ilike("player_name", f"%{name}%")
        .execute()
    )
    rows = res.data or []
    if not rows:
        raise LookupError(f"No player matching '{name}'.")
    # Prefer an exact (case-insensitive) hit if the partial match returned several.
    exact = [r for r in rows if (r.get("player_name") or "").lower() == name.lower()]
    if exact:
        return exact[0]
    if len(rows) > 1:
        names = ", ".join(r.get("player_name", "?") for r in rows[:10])
        raise LookupError(f"'{name}' is ambiguous. Did you mean one of: {names}")
    return rows[0]


def fetch_player_games(player_id) -> list:
    """All game-log rows for a player, oldest-first. Pages past the 1000 cap."""
    columns = ",".join(LOG_COLUMNS)
    rows = []
    start = 0
    while True:
        res = (
            supabase.table(LOGS_TABLE)
            .select(columns)
            .eq("player_id", player_id)
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


def team_abbrev_maps():
    """Return (name_lower -> abbrev, abbrev -> full_name) from nba_api.

    The name map accepts a full name ('New York Knicks'), a nickname
    ('Knicks' -- what nba_players.team actually stores), or an abbreviation
    ('NYK'), so team lookups work no matter which form a table uses.
    """
    name_to_abbr = {}
    abbr_to_name = {}
    for t in teams.get_teams():
        abbr = t["abbreviation"]
        name_to_abbr[t["full_name"].lower()] = abbr
        name_to_abbr[t["nickname"].lower()] = abbr
        name_to_abbr[abbr.lower()] = abbr
        abbr_to_name[abbr] = t["full_name"]
    return name_to_abbr, abbr_to_name


def next_game_for_team(team_abbr: str):
    """Return (opponent_abbr, home_away, game_date) for the team's next game.

    Reads nba_schedule for the earliest 'upcoming' game involving the team.
    Returns (None, None, None) if nothing upcoming (e.g. offseason).
    """
    if not team_abbr:
        return None, None, None
    res = (
        supabase.table(SCHEDULE_TABLE)
        .select("game_date,home_team,away_team,status")
        .eq("status", "upcoming")
        .or_(f"home_team.eq.{team_abbr},away_team.eq.{team_abbr}")
        .order("game_date", desc=False)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return None, None, None
    game = rows[0]
    game_date = game.get("game_date")
    if game.get("home_team") == team_abbr:
        return game.get("away_team"), "HOME", game_date
    return game.get("home_team"), "AWAY", game_date


def defense_vs_position(opponent_abbr: str, bucket: str, abbr_to_name: dict):
    """Return (opp_pts_allowed_to_position, league_avg) for the current season.

    Reads nba_defensive_ratings (Regular Season). Either value may be None if the
    matchup or league baseline can't be resolved yet.
    """
    res = (
        supabase.table(DEF_TABLE)
        .select("team_name,position,points_allowed_per_game,season_type")
        .eq("season", CURRENT_SEASON)
        .eq("position", bucket)
        .eq("season_type", "Regular Season")
        .execute()
    )
    rows = res.data or []
    if not rows:
        return None, None
    league_avg = mean([r.get("points_allowed_per_game") for r in rows])

    opp_value = None
    opp_full = (abbr_to_name.get(opponent_abbr) or "").lower()
    for r in rows:
        if (r.get("team_name") or "").lower() == opp_full:
            opp_value = r.get("points_allowed_per_game")
            break
    return opp_value, league_avg


def team_pace(opponent_abbr: str, abbr_to_name: dict):
    """Return (opponent_pace, league_avg_pace) for the current season.

    Reads nba_team_stats (Regular Season). Either value may be None if pace
    hasn't been recorded for the opponent / league yet.
    """
    res = (
        supabase.table(TEAM_STATS_TABLE)
        .select("team_name,pace,season_type")
        .eq("season", CURRENT_SEASON)
        .eq("season_type", "Regular Season")
        .execute()
    )
    rows = res.data or []
    if not rows:
        return None, None
    league_avg = mean([r.get("pace") for r in rows])

    opp_value = None
    opp_full = (abbr_to_name.get(opponent_abbr) or "").lower()
    for r in rows:
        if (r.get("team_name") or "").lower() == opp_full:
            opp_value = r.get("pace")
            break
    return opp_value, league_avg


def head_to_head(player_id, opponent_abbr: str, columns):
    """Return (h2h_value, games_played) for this player vs this opponent.

    Uses the all-time H2H row (biggest sample) from nba_head_to_head. h2h_value
    is the sum of the per-component averages for the requested stat -- but only
    if EVERY component has a stored H2H column (else there's no honest H2H
    figure and we return (None, 0)). opponent_abbr matches opponent_team, which
    is stored as a team abbreviation.
    """
    if any(c not in H2H_COLUMN for c in columns):
        return None, 0
    select_cols = "games_played," + ",".join(H2H_COLUMN[c] for c in columns)
    res = (
        supabase.table(H2H_TABLE)
        .select(select_cols)
        .eq("player_id", player_id)
        .eq("opponent_team", opponent_abbr)
        .eq("season", "all-time")
        .limit(1)
        .execute()
    )
    rows = res.data or []
    if not rows:
        return None, 0
    row = rows[0]
    component_avgs = [row.get(H2H_COLUMN[c]) for c in columns]
    if any(v is None for v in component_avgs):
        return None, 0
    return sum(component_avgs), row.get("games_played") or 0


def injury_status(player_id):
    """Return the most recent injury status string for a player, or None."""
    res = (
        supabase.table(INJURIES_TABLE)
        .select("status,reason,game_date")
        .eq("player_id", player_id)
        .order("game_date", desc=True)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return rows[0] if rows else None


_positions_cache = None


def all_positions() -> dict:
    """Return {player_id: G/F/C bucket} for every player, loaded once and cached."""
    global _positions_cache
    if _positions_cache is None:
        res = supabase.table(PLAYERS_TABLE).select("player_id,position").execute()
        _positions_cache = {
            r["player_id"]: position_bucket(r.get("position"))
            for r in (res.data or [])
            if r.get("player_id") is not None
        }
    return _positions_cache


def opponent_allowed_to_position(opponent_abbr: str, bucket: str):
    """Return {points,rebounds,assists} allowed PER GAME to `bucket` this season.

    Computed live from this season's game logs the same way 10_build_training_data
    does, so the model sees inference features built identically to training ones:
    sum each stat conceded by `opponent_abbr` to players of `bucket`, divided by
    the number of games that opponent has played. Returns Nones if unresolved.
    """
    empty = {"points": None, "rebounds": None, "assists": None}
    if not opponent_abbr:
        return empty
    positions = all_positions()
    columns = "player_id,points,rebounds,assists,game_date"
    rows = []
    start = 0
    while True:
        res = (
            supabase.table(LOGS_TABLE)
            .select(columns)
            .eq("season", CURRENT_SEASON)
            .eq("opponent", opponent_abbr)
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )
        page = res.data or []
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    if not rows:
        return empty

    games = len({r.get("game_date") for r in rows if r.get("game_date")})
    if not games:
        return empty
    totals = {"points": 0.0, "rebounds": 0.0, "assists": 0.0}
    for r in rows:
        if positions.get(r.get("player_id")) != bucket:
            continue
        for stat in totals:
            totals[stat] += (r.get(stat) or 0)
    return {stat: totals[stat] / games for stat in totals}


_model_cache = None


def load_models():
    """Load the trained models + metadata once. Returns None if not trained yet."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache or None   # cached miss is stored as {}
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        _model_cache = {}
        return None
    try:
        import joblib
        with open(meta_path) as f:
            meta = json.load(f)
        models = {}
        for stat in MODELED_STATS:
            models[stat] = {
                "mean": joblib.load(os.path.join(MODEL_DIR, f"{stat}_mean.joblib")),
                "q16": joblib.load(os.path.join(MODEL_DIR, f"{stat}_q16.joblib")),
                "q84": joblib.load(os.path.join(MODEL_DIR, f"{stat}_q84.joblib")),
            }
        _model_cache = {"meta": meta, "models": models}
        return _model_cache
    except Exception:  # noqa: BLE001 - any load failure => use heuristic
        _model_cache = {}
        return None


def _model_form(played, season_games, col):
    """Rolling (last-5 / last-10 over ALL played games) + season-to-date average.

    Mirrors how 10_build_training_data computed these: the rolling windows span
    the season boundary (last N games regardless of season), the season average
    is restricted to the current season.
    """
    all_vals = [game_value(g, [col]) for g in played]
    l5 = mean(all_vals[-5:])
    l10 = mean(all_vals[-10:])
    season = mean([game_value(g, [col]) for g in season_games])
    return l5, l10, season


def _model_feature_row(player_id, bucket, played, season_games,
                       opponent, home_away, next_date, season_type):
    """Build the 1-row feature frame the trained models expect (or None).

    Features are constructed identically to 10_build_training_data so inference
    matches training. Returns a dict with the ready-to-predict DataFrame plus the
    per-stat anchors, opponent stat-allowed-to-position, head-to-head averages,
    and the resolved season type.
    """
    cache = load_models()
    if not cache:
        return None
    meta = cache["meta"]

    feats, anchors = {}, {}
    for s in MODELED_STATS:
        l5, l10, season = _model_form(played, season_games, s)
        feats[f"l5_{s}"], feats[f"l10_{s}"], feats[f"season_{s}"] = l5, l10, season
        # Anchor = season avg, falling back to L10 then L5 (matches training).
        anchors[s] = season if season is not None else (l10 if l10 is not None else l5)

    m5, m10, _ = _model_form(played, season_games, "minutes_played")
    feats["l5_minutes"], feats["l10_minutes"] = m5, m10
    feats["home"] = 1 if home_away == "HOME" else 0
    feats["season_games_todate"] = len(season_games)

    # Days of rest, if we know when the next game is (capped to training range).
    feats["days_rest"] = None
    if next_date:
        last = max((g.get("game_date") for g in played if g.get("game_date")), default=None)
        try:
            d0 = datetime.strptime(last[:10], "%Y-%m-%d").date()
            d1 = datetime.strptime(next_date[:10], "%Y-%m-%d").date()
            feats["days_rest"] = min((d1 - d0).days, 14)
        except (ValueError, TypeError):
            pass

    # Head-to-head averages vs this opponent (all of the player's history).
    opp_games = (
        [g for g in played if (g.get("opponent") or "").upper() == opponent]
        if opponent else []
    )
    h2h = {}
    for s in MODELED_STATS:
        h2h[s] = mean([game_value(g, [s]) for g in opp_games]) if opp_games else None
        feats[f"h2h_{s}"] = h2h[s]

    # Opponent's stat allowed per game to this position (this season).
    allowed = opponent_allowed_to_position(opponent, bucket)
    for s in MODELED_STATS:
        feats[f"opp_{s}_allowed_to_pos"] = allowed.get(s)

    feats["position"] = bucket
    feats["season_type"] = (
        season_type if season_type in ("Regular Season", "Playoffs") else "Regular Season"
    )

    import pandas as pd
    X = pd.DataFrame([feats]).reindex(columns=meta["features"])
    for c in meta["categorical_features"]:
        X[c] = X[c].astype("category")

    return {
        "X": X,
        "anchors": anchors,
        "allowed": allowed,
        "h2h": h2h,
        "opp_games": len(opp_games),
        "season_type": feats["season_type"],
    }


def model_projection(components, player_id, bucket, played, season_games,
                     opponent, home_away, next_date, season_type, empirical_sigma):
    """Predict a single modeled stat or a combo of them with the trained models.

    `components` is a list of base stats: ['points'] for a single stat, or e.g.
    ['points','rebounds','assists'] for PRA. Every component must have a model,
    or we return None so the caller uses the heuristic.

    Single stat: sigma comes from the quantile models (per-row spread). Combo:
    means are summed and sigma is the player's empirical game-to-game std of the
    combo -- summing the component quantiles would overstate the spread because a
    player's pts/reb/ast in a game are correlated.
    """
    cache = load_models()
    if not cache or any(c not in MODELED_STATS for c in components):
        return None
    fr = _model_feature_row(player_id, bucket, played, season_games,
                            opponent, home_away, next_date, season_type)
    if fr is None or any(fr["anchors"][c] is None for c in components):
        return None

    X, anchors, models = fr["X"], fr["anchors"], cache["models"]
    total_anchor = sum(anchors[c] for c in components)
    total_mean = sum(anchors[c] + float(models[c]["mean"].predict(X)[0]) for c in components)
    projection = max(total_mean, 0.0)

    if len(components) == 1:
        c = components[0]
        q16 = anchors[c] + float(models[c]["q16"].predict(X)[0])
        q84 = anchors[c] + float(models[c]["q84"].predict(X)[0])
        sigma = max((q84 - q16) / 2.0, cache["meta"].get("sigma_floor", 1.0))
    else:
        sigma = empirical_sigma
        q16 = projection - sigma if sigma else None
        q84 = projection + sigma if sigma else None

    return {
        "projection": round(projection, 2),
        "sigma": round(sigma, 2) if sigma else None,
        "q16": round(max(q16, 0.0), 2) if q16 is not None else None,
        "q84": round(q84, 2) if q84 is not None else None,
        "anchor": round(total_anchor, 2),
        "components": components,
        "allowed": fr["allowed"],
        "h2h": fr["h2h"],
        "opp_games": fr["opp_games"],
        "season_type": fr["season_type"],
    }


def build_factors(result, stat, bucket, opponent, home_away, season_type, model_out,
                  l5, l10, season_avg, location_split, opp_pace, league_pace,
                  pace_factor, opp_def, league_def, h2h_value, h2h_games):
    """Build a list of {title, value, detail} cards explaining the projection.

    These turn the raw numbers into plain English for the UI -- each card names a
    factor, shows its value, and explains what it means / which way it pushes.
    """
    noun = STAT_NOUNS.get(stat, stat)
    pos = POSITION_NAMES.get(bucket, "this position")
    loc_word = {"HOME": "at home", "AWAY": "on the road"}.get(home_away)
    rnd = lambda v: round(v, 1) if isinstance(v, (int, float)) else v
    l5, l10, season_avg, location_split = rnd(l5), rnd(l10), rnd(season_avg), rnd(location_split)
    factors = []

    # 1) Recent form (the anchor / baseline).
    anchor = model_out["anchor"] if model_out else result.get("season_avg")
    factors.append({
        "title": "Recent form",
        "value": f"{anchor} {noun}",
        "detail": f"Baseline from this player's recent games — "
                  f"L5 {l5}, L10 {l10}, season {season_avg}.",
    })

    # 2) Opponent defense. For points we know the league average, so we can call
    #    it favorable/tough; for other stats we state the number plainly.
    if opponent:
        if stat == "points" and opp_def and league_def:
            verdict = ("an EASIER matchup — they give up more than average"
                       if opp_def > league_def
                       else "a TOUGHER matchup — they give up less than average")
            factors.append({
                "title": "Opponent defense",
                "value": f"{opponent} allows {opp_def} pts/g to {pos}",
                "detail": f"League average is {league_def}. That's {verdict}.",
            })
        elif model_out and model_out["allowed"].get(stat if stat in MODELED_STATS else None) is not None:
            val = model_out["allowed"][stat]
            factors.append({
                "title": "Opponent defense",
                "value": f"{opponent} allows {round(val, 1)} {noun}/g to {pos}",
                "detail": f"How much {opponent} typically concedes to {pos} this season.",
            })
        elif model_out and stat in MODEL_COMBOS:
            parts = ", ".join(
                f"{round(model_out['allowed'][c], 1)} {c}"
                for c in MODEL_COMBOS[stat] if model_out["allowed"].get(c) is not None
            )
            if parts:
                factors.append({
                    "title": "Opponent defense",
                    "value": f"{opponent} allows {parts} per game to {pos}",
                    "detail": "Combined into the matchup for this combo stat.",
                })

    # 3) Head-to-head history vs this opponent.
    h2h_n = (model_out["opp_games"] if model_out else h2h_games) or 0
    h2h_avg = None
    if model_out and stat in MODELED_STATS:
        h2h_avg = model_out["h2h"].get(stat)
    elif model_out and stat in MODEL_COMBOS:
        vals = [model_out["h2h"].get(c) for c in MODEL_COMBOS[stat]]
        h2h_avg = round(sum(v for v in vals if v is not None), 1) if all(v is not None for v in vals) else None
    elif h2h_value is not None:
        h2h_avg = h2h_value
    if opponent and h2h_n and h2h_avg is not None:
        factors.append({
            "title": "Head-to-head",
            "value": f"{round(h2h_avg, 1)} {noun} avg vs {opponent}",
            "detail": f"Across {h2h_n} career game(s) against this opponent.",
        })

    # 4) Pace (heuristic stats only -- the model handles tempo internally).
    if result.get("method") == "heuristic" and opp_pace and league_pace:
        tempo = "fast" if opp_pace > league_pace else "slow"
        factors.append({
            "title": "Pace",
            "value": f"{opponent} plays at {round(opp_pace, 1)} (lg {round(league_pace, 1)})",
            "detail": f"A {tempo} pace means {'more' if tempo == 'fast' else 'fewer'} "
                      f"possessions, nudging counting stats {'up' if tempo == 'fast' else 'down'}.",
        })

    # 5) Home/away.
    if loc_word:
        extra = (f" His {home_away.lower()} average is {location_split}."
                 if location_split is not None else "")
        factors.append({
            "title": "Location",
            "value": f"Playing {loc_word}",
            "detail": f"Factored into the projection.{extra}",
        })

    # 6) Game type.
    factors.append({
        "title": "Game type",
        "value": season_type,
        "detail": "The model learned regular-season and playoff scoring patterns "
                  "separately and applies the one you selected."
                  if result.get("method") == "model"
                  else "Regular-season and playoff games are treated the same in the heuristic.",
    })

    # 7) How it was produced.
    if result.get("method") == "model":
        factors.append({
            "title": "Projection method",
            "value": f"Trained model → {result['projection']} (± {result['sigma']})",
            "detail": "A gradient-boosted model (trained on ~87k player-games) adjusts "
                      "the recent-form baseline using the matchup, rest, and home/away.",
        })
    else:
        factors.append({
            "title": "Projection method",
            "value": f"Heuristic → {result['projection']} (± {result['sigma']})",
            "detail": "No trained model for this stat yet, so a weighted recent-form "
                      "blend with pace/defense/H2H adjustments is used.",
        })

    return factors


# ---------------------------------------------------------------------------
# The projection engine
# ---------------------------------------------------------------------------
def project_player(player_name: str, stat: str, line: float = None,
                   opponent: str = None, home_away: str = None,
                   season_type: str = "auto") -> dict:
    """Project `stat` for `player_name`'s next game and (optionally) grade `line`.

    `season_type` is "Regular Season", "Playoffs", or "auto" (derive it from the
    game date). It feeds the model so a playoff projection uses the playoff
    pattern the model learned.

    Returns a dict with the projection, the spread (sigma), a plain-English
    `factors` breakdown of why, and -- if a line was supplied -- P(over)/P(under),
    a recommendation, and a confidence. The single entry point for the API/CLI.
    """
    stat = stat.lower()
    if stat not in STAT_DEFS:
        raise ValueError(
            f"Unknown stat '{stat}'. Choose from: {', '.join(sorted(STAT_DEFS))}"
        )
    columns = STAT_DEFS[stat]

    player = find_player(player_name)
    player_id = player["player_id"]
    bucket = position_bucket(player.get("position"))

    games = fetch_player_games(player_id)
    # Only count games the player actually played -- DNPs (0/None minutes) would
    # drag every average and inflate the variance.
    played = [g for g in games if g.get("minutes_played")]
    if not played:
        raise LookupError(f"No games with minutes found for {player['player_name']}.")

    season_games = [g for g in played if g.get("season") == CURRENT_SEASON]
    # Fall back to all-time form if the current season hasn't really started.
    form_games = season_games if len(season_games) >= 3 else played

    per_game = [game_value(g, columns) for g in form_games]
    l5 = mean(per_game[-5:])
    l10 = mean(per_game[-10:])
    season_avg = mean([game_value(g, columns) for g in season_games]) or mean(per_game)

    # --- Base: weighted blend of the windows that actually have data ---------
    parts = [(l5, WEIGHT_L5), (l10, WEIGHT_L10), (season_avg, WEIGHT_SEASON)]
    parts = [(v, w) for v, w in parts if v is not None]
    total_w = sum(w for _, w in parts)
    base = sum(v * w for v, w in parts) / total_w if total_w else mean(per_game)

    # --- Resolve the matchup (opponent + location) ---------------------------
    name_to_abbr, abbr_to_name = team_abbrev_maps()
    team_abbr = name_to_abbr.get((player.get("team") or "").lower())

    next_date = None
    if opponent is None and home_away is None:
        opponent, home_away, next_date = next_game_for_team(team_abbr)
    opponent = opponent.upper() if opponent else None

    # Resolve "auto" -> Regular Season / Playoffs from the upcoming game's date.
    season_type = resolve_season_type(season_type, next_date)

    # `adjustments` records every step applied on top of the form blend, so the
    # output can explain exactly why the projection differs from the averages.
    adjustments = []

    # --- Location nudge: lean part of the projection on the home/away split ---
    location_split = None
    if home_away in ("HOME", "AWAY"):
        loc_games = [g for g in season_games if g.get("home_away") == home_away]
        location_split = mean([game_value(g, columns) for g in loc_games])
    projection = base
    if location_split is not None:
        projection = (1 - LOCATION_WEIGHT) * base + LOCATION_WEIGHT * location_split
        adjustments.append(f"{home_away.lower()} split blend -> {projection:.2f}")

    # --- Pace factor: opponent possessions vs league average (ALL stats) ------
    pace_factor = 1.0
    opp_pace = league_pace = None
    if opponent:
        opp_pace, league_pace = team_pace(opponent, abbr_to_name)
        if opp_pace and league_pace:
            pace_factor = 1.0 + PACE_DAMPENING * (opp_pace / league_pace - 1.0)
            projection *= pace_factor
            adjustments.append(f"pace x{pace_factor:.3f}")

    # --- Matchup factor: opponent defense-vs-position (points stats only) -----
    matchup_factor = 1.0
    opp_def = league_def = None
    if "points" in columns and opponent:
        opp_def, league_def = defense_vs_position(opponent, bucket, abbr_to_name)
        if opp_def and league_def:
            matchup_factor = 1.0 + MATCHUP_DAMPENING * (opp_def / league_def - 1.0)
            projection *= matchup_factor
            adjustments.append(f"def-vs-{bucket} x{matchup_factor:.3f}")

    # --- Head-to-head: blend toward this player's history vs this opponent ----
    # Weight ramps with sample size so a tiny sample barely moves the number.
    h2h_value = h2h_games = h2h_weight = None
    if opponent:
        h2h_value, h2h_games = head_to_head(player_id, opponent, columns)
        if h2h_value is not None and h2h_games:
            h2h_weight = H2H_MAX_WEIGHT * min(h2h_games / H2H_FULL_WEIGHT_GAMES, 1.0)
            projection = (1 - h2h_weight) * projection + h2h_weight * h2h_value
            adjustments.append(
                f"H2H {h2h_value:.1f} over {h2h_games}g (w={h2h_weight:.2f}) -> {projection:.2f}"
            )

    # --- Spread (sigma) from recent game-to-game variation -------------------
    recent = per_game[-SIGMA_WINDOW:]
    sigma = statistics.pstdev(recent) if len(recent) >= 2 else None

    # The heuristic projection above is the fallback. If a trained model covers
    # this stat -- points/rebounds/assists, or a combo made purely of them -- it
    # overrides projection + sigma. Steals/blocks/threes/turnovers stay heuristic.
    method = "heuristic"
    components = (
        [stat] if stat in MODELED_STATS
        else MODEL_COMBOS.get(stat)            # None for non-modelable stats
    )
    model_out = None
    if components:
        model_out = model_projection(
            components, player_id, bucket, played, season_games,
            opponent, home_away, next_date, season_type, sigma,
        )
    if model_out is not None:
        projection = model_out["projection"]
        sigma = model_out["sigma"] if model_out["sigma"] is not None else sigma
        method = "model"

    result = {
        "method": method,
        "player_name": player["player_name"],
        "team": player.get("team"),
        "position": player.get("position"),
        "position_bucket": bucket,
        "stat": stat,
        "opponent": opponent,
        "home_away": home_away,
        "games_used": len(form_games),
        "l5": round(l5, 2) if l5 is not None else None,
        "l10": round(l10, 2) if l10 is not None else None,
        "season_avg": round(season_avg, 2) if season_avg is not None else None,
        "location_split": round(location_split, 2) if location_split is not None else None,
        "pace_factor": round(pace_factor, 3),
        "opp_pace": round(opp_pace, 2) if opp_pace else None,
        "league_pace": round(league_pace, 2) if league_pace else None,
        "matchup_factor": round(matchup_factor, 3),
        "opp_def_vs_pos": round(opp_def, 2) if opp_def else None,
        "league_def_vs_pos": round(league_def, 2) if league_def else None,
        "h2h_value": round(h2h_value, 2) if h2h_value is not None else None,
        "h2h_games": h2h_games,
        "h2h_weight": round(h2h_weight, 2) if h2h_weight is not None else None,
        "adjustments": adjustments,
        "season_type": season_type,
        "projection": round(projection, 2),
        "sigma": round(sigma, 2) if sigma is not None else None,
        "injury": injury_status(player_id),
        "line": line,
    }
    if model_out is not None:
        result.update({
            "model_anchor": model_out["anchor"],
            "model_q16": model_out["q16"],
            "model_q84": model_out["q84"],
        })

    # --- Plain-English "why" breakdown ---------------------------------------
    result["factors"] = build_factors(
        result, stat, bucket, opponent, home_away, season_type, model_out,
        l5=l5, l10=l10, season_avg=season_avg, location_split=location_split,
        opp_pace=opp_pace, league_pace=league_pace, pace_factor=pace_factor,
        opp_def=opp_def, league_def=league_def,
        h2h_value=h2h_value, h2h_games=h2h_games,
    )

    # --- Grade the user's line -----------------------------------------------
    if line is not None and sigma:
        z = (projection - line) / sigma
        p_over = normal_cdf(z)
        p_under = 1.0 - p_over
        pick = "OVER" if p_over >= 0.5 else "UNDER"
        confidence = max(p_over, p_under)
        result.update({
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "recommendation": pick,
            "confidence": round(confidence, 4),
            "confidence_label": confidence_label(confidence),
        })
    elif line is not None and not sigma:
        result["note"] = "Not enough game history to estimate spread; showing projection only."

    return result


def confidence_label(confidence: float) -> str:
    """Turn a 0.5-1.0 confidence into a plain-English bucket."""
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
    head = f"{r['player_name']}  ({r.get('team') or '?'}, {r.get('position') or '?'})"
    lines.append(head)
    matchup = ""
    if r.get("opponent"):
        loc = {"HOME": "vs", "AWAY": "@"}.get(r.get("home_away"), "vs")
        matchup = f"  {loc} {r['opponent']}"
    tag = "trained model" if r.get("method") == "model" else "heuristic"
    lines.append(f"Stat: {r['stat'].upper()}{matchup}  ({r.get('season_type')})   [{tag}]")
    lines.append("-" * 60)
    lines.append(f"  PROJECTION:   {r['projection']}   (sigma +/- {r['sigma']})")
    lines.append("-" * 60)
    lines.append("  Why:")
    for f in r.get("factors", []):
        lines.append(f"   • {f['title']}: {f['value']}")
        lines.append(f"       {f['detail']}")

    inj = r.get("injury")
    if inj and (inj.get("status") or "").lower() not in ("", "active", "available"):
        lines.append(f"  !! INJURY:    {inj.get('status')} - {inj.get('reason') or ''}".rstrip(" -"))

    if r.get("line") is not None and "recommendation" in r:
        lines.append("-" * 60)
        lines.append(f"  Your line:    {r['line']}")
        lines.append(
            f"  OVER {r['p_over'] * 100:.1f}%   |   UNDER {r['p_under'] * 100:.1f}%"
        )
        lines.append(
            f"  >>> {r['recommendation']} {r['line']}  "
            f"({r['confidence'] * 100:.1f}% - {r['confidence_label']})"
        )
    elif r.get("note"):
        lines.append(f"  Note: {r['note']}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Project an NBA player's stat and grade your own betting line."
    )
    parser.add_argument("--player", required=True, help="Player name (partial ok).")
    parser.add_argument(
        "--stat", required=True,
        help=f"One of: {', '.join(sorted(STAT_DEFS))}",
    )
    parser.add_argument("--line", type=float, default=None,
                        help="The over/under line from YOUR book (optional).")
    parser.add_argument("--opponent", default=None,
                        help="Opponent team abbrev (e.g. DEN). Default: auto from schedule.")
    loc = parser.add_mutually_exclusive_group()
    loc.add_argument("--home", action="store_true", help="Force home game.")
    loc.add_argument("--away", action="store_true", help="Force away game.")
    parser.add_argument("--playoffs", action="store_true",
                        help="Treat as a playoff game (default: regular season).")
    args = parser.parse_args()

    home_away = None
    if args.home:
        home_away = "HOME"
    elif args.away:
        home_away = "AWAY"

    try:
        result = project_player(
            args.player, args.stat, line=args.line,
            opponent=args.opponent, home_away=home_away,
            season_type="Playoffs" if args.playoffs else "Regular Season",
        )
    except (LookupError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")

    print(format_report(result))


if __name__ == "__main__":
    main()

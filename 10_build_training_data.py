"""
Build a POINT-IN-TIME training dataset from nba_player_game_logs.

    python 10_build_training_data.py            # writes training_data.csv

This is the foundation of the trained model. For every historical player-game we
emit one row whose features are computed using ONLY games that happened strictly
before that game's date -- no peeking at the outcome or at same-day results. That
"as-of" discipline is what makes the later backtest trustworthy; a model trained
on leaked features looks brilliant in testing and loses money in production.

How leakage is prevented
------------------------
Games are processed in date order, one calendar day at a time:

  1. For every game on day D, compute features from the running state, which at
     that moment reflects only days < D.
  2. AFTER all of day D's features are recorded, fold day D's games into the
     state (player histories, opponent-defense accumulators, ...).

So a game's own result -- and every other result from the same day -- is invisible
when its features are built.

Features per row (all as-of the game date)
------------------------------------------
  form:      l5 / l10 / season-to-date averages of pts, reb, ast, and minutes
  context:   home flag, days of rest, games played this season so far
  opponent:  points/reb/ast the opponent has allowed PER GAME to this player's
             position so far this season (+ how many games that average is built
             on, so the model can distrust tiny samples)
  history:   this player's avg pts/reb/ast vs THIS opponent so far (+ sample size)
  position:  G / F / C bucket

Targets (labels): the player's actual points, rebounds, assists that game.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import os
from collections import defaultdict, deque
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOGS_TABLE = "nba_player_game_logs"
PLAYERS_TABLE = "nba_players"
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data.csv")
PAGE_SIZE = 1000          # rows per Supabase select page (PostgREST default cap)

# A row is only emitted once the player and the opponent each have enough prior
# games for the as-of features to mean something (early-season noise is dropped).
MIN_PLAYER_GAMES = 5
MIN_OPP_GAMES = 5
ROLLING_MAX = 10          # longest rolling window we need (L10)

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit(
        "Missing credentials. Set SUPABASE_URL and SUPABASE_KEY in your .env file."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

STAT_KEYS = ("points", "rebounds", "assists")   # the three we model + accumulate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def position_bucket(position) -> str:
    """Normalize a position string to G / F / C (same buckets as 05)."""
    if not position:
        return "Unknown"
    primary = position.split("-")[0].strip().lower()
    return {"guard": "G", "forward": "F", "center": "C"}.get(primary, "Unknown")


def num(value):
    """Coerce a stored stat to float, treating None/'' as 0.0."""
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def avg(total, count):
    """Safe mean; None when there's nothing to average."""
    return total / count if count else None


def load_positions() -> dict:
    """Return {player_id: G/F/C bucket} from nba_players."""
    res = supabase.table(PLAYERS_TABLE).select("player_id,position").execute()
    return {
        r["player_id"]: position_bucket(r.get("position"))
        for r in (res.data or [])
        if r.get("player_id") is not None
    }


def fetch_all_logs() -> list:
    """Pull every game-log row we need, oldest-first, paging past the 1000 cap."""
    columns = (
        "player_id,player_name,game_date,season,season_type,opponent,home_away,"
        "minutes_played,points,rebounds,assists"
    )
    rows = []
    start = 0
    while True:
        res = (
            supabase.table(LOGS_TABLE)
            .select(columns)
            .order("game_date", desc=False)
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )
        page = res.data or []
        rows.extend(page)
        print(f"  fetched {len(rows)} rows ...")
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return rows


def parse_date(raw):
    """ISO 'YYYY-MM-DD' -> date (None if unparseable)."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Running state (all "as-of" the day currently being processed)
# ---------------------------------------------------------------------------
class State:
    """Accumulators updated chronologically; every read is point-in-time."""

    def __init__(self):
        # Per player: recent games (for rolling windows) and season running sums.
        self.recent = defaultdict(lambda: deque(maxlen=ROLLING_MAX))
        self.season_sum = defaultdict(lambda: defaultdict(float))   # (pid, season) -> stat sums + 'n'
        self.games_played = defaultdict(int)                        # pid -> games so far
        self.last_date = {}                                         # pid -> date of previous game
        # Per (player, opponent): head-to-head running sums.
        self.h2h = defaultdict(lambda: defaultdict(float))          # (pid, opp) -> stat sums + 'n'
        # Per (defending_team, position): stat conceded; per team: games played.
        self.opp_allowed = defaultdict(lambda: defaultdict(float))  # (team, pos) -> stat sums
        self.team_games = defaultdict(int)                          # team -> games defended

    # --- reads (features) ---------------------------------------------------
    def player_features(self, pid, season):
        recent = self.recent[pid]
        last5 = list(recent)[-5:]
        last10 = list(recent)
        ssum = self.season_sum[(pid, season)]
        n_season = ssum.get("n", 0)
        feats = {
            "games_played_todate": self.games_played[pid],
            "season_games_todate": n_season,
        }
        for stat in STAT_KEYS:
            feats[f"l5_{stat}"] = avg(sum(g[stat] for g in last5), len(last5))
            feats[f"l10_{stat}"] = avg(sum(g[stat] for g in last10), len(last10))
            feats[f"season_{stat}"] = avg(ssum.get(stat, 0.0), n_season)
        feats["l5_minutes"] = avg(sum(g["minutes"] for g in last5), len(last5))
        feats["l10_minutes"] = avg(sum(g["minutes"] for g in last10), len(last10))
        return feats

    def h2h_features(self, pid, opp):
        h = self.h2h[(pid, opp)]
        n = h.get("n", 0)
        feats = {"h2h_games": n}
        for stat in STAT_KEYS:
            feats[f"h2h_{stat}"] = avg(h.get(stat, 0.0), n)
        return feats

    def opponent_features(self, opp, pos):
        games = self.team_games[opp]
        feats = {"opp_games_todate": games}
        allowed = self.opp_allowed[(opp, pos)]
        for stat in STAT_KEYS:
            feats[f"opp_{stat}_allowed_to_pos"] = avg(allowed.get(stat, 0.0), games)
        return feats

    def rest_days(self, pid, game_date):
        prev = self.last_date.get(pid)
        return (game_date - prev).days if prev else None

    # --- writes (fold a finished day into state) ----------------------------
    def add_game(self, g):
        pid, season, opp, pos = g["player_id"], g["season"], g["opponent"], g["pos"]
        vals = {stat: g[stat] for stat in STAT_KEYS}
        vals["minutes"] = g["minutes"]

        self.recent[pid].append(vals)
        ssum = self.season_sum[(pid, season)]
        h = self.h2h[(pid, opp)] if opp else None
        for stat in STAT_KEYS:
            ssum[stat] += vals[stat]
            if h is not None:
                h[stat] += vals[stat]
            if opp:
                self.opp_allowed[(opp, pos)][stat] += vals[stat]
        ssum["n"] += 1
        if h is not None:
            h["n"] += 1
        self.games_played[pid] += 1
        self.last_date[pid] = g["date"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading player positions ...")
    positions = load_positions()
    print(f"  {len(positions)} players with a position")

    print(f"Reading game logs from {LOGS_TABLE} ...")
    raw = fetch_all_logs()
    print(f"Total game-log rows: {len(raw)}\n")

    # Normalize rows and bucket them by calendar day, preserving date order.
    by_day = defaultdict(list)
    for r in raw:
        gdate = parse_date(r.get("game_date"))
        if gdate is None or r.get("player_id") is None:
            continue
        by_day[gdate].append({
            "player_id": r["player_id"],
            "player_name": r.get("player_name"),
            "date": gdate,
            "season": r.get("season"),
            "season_type": r.get("season_type"),
            "opponent": (r.get("opponent") or "").strip() or None,
            "home": 1 if r.get("home_away") == "HOME" else 0,
            "pos": positions.get(r["player_id"], "Unknown"),
            "minutes": num(r.get("minutes_played")),
            "points": num(r.get("points")),
            "rebounds": num(r.get("rebounds")),
            "assists": num(r.get("assists")),
        })

    state = State()
    out_rows = []

    for day in sorted(by_day):
        games = by_day[day]

        # 1) READ: build features for every game on this day from prior state.
        for g in games:
            # Only emit rows for games the player actually played and where the
            # as-of features rest on enough prior data to be meaningful.
            if g["minutes"] <= 0:
                continue
            if state.games_played[g["player_id"]] < MIN_PLAYER_GAMES:
                continue
            if not g["opponent"] or state.team_games[g["opponent"]] < MIN_OPP_GAMES:
                continue

            row = {
                "player_id": g["player_id"],
                "player_name": g["player_name"],
                "game_date": g["date"].isoformat(),
                "season": g["season"],
                "season_type": g["season_type"],
                "opponent": g["opponent"],
                "position": g["pos"],
                "home": g["home"],
                "days_rest": state.rest_days(g["player_id"], g["date"]),
            }
            row.update(state.player_features(g["player_id"], g["season"]))
            row.update(state.h2h_features(g["player_id"], g["opponent"]))
            row.update(state.opponent_features(g["opponent"], g["pos"]))
            # Labels (the actual outcome we want to predict).
            row["target_points"] = g["points"]
            row["target_rebounds"] = g["rebounds"]
            row["target_assists"] = g["assists"]
            out_rows.append(row)

        # 2) WRITE: now that the day's features are locked, fold the day in.
        opponents_today = set()
        for g in games:
            state.add_game(g)
            if g["opponent"]:
                opponents_today.add(g["opponent"])
        for opp in opponents_today:
            state.team_games[opp] += 1

    df = pd.DataFrame(out_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("=" * 60)
    print(f"DONE. Wrote {len(df)} training rows to {OUTPUT_CSV}")
    if not df.empty:
        seasons = ", ".join(sorted(s for s in df["season"].dropna().unique()))
        print(f"Seasons covered: {seasons}")
        print(f"Columns ({len(df.columns)}): {', '.join(df.columns)}")


if __name__ == "__main__":
    main()

"""
Build a POINT-IN-TIME training dataset for NFL player props.

    python 34_nfl_build_props_training.py     # writes nfl_training_data.csv

The football sibling of 10_build_training_data.py. For every historical
player-game in nfl_player_game_logs we emit one row whose features are computed
using ONLY games strictly before that game's date -- the same "as-of" discipline
that keeps the backtest honest. Games are processed one calendar day at a time:
features are read from the running state (which reflects only earlier days),
then the day's results are folded in.

Features per row (all as-of the game date)
------------------------------------------
  form:      L3 / L5 / season-to-date per-game rate of every modeled stat
  context:   home flag, season games played so far, position, season type
  opponent:  each stat the opponent's DEFENSE has allowed per game so far this
             season (pass yds / rush yds / receptions / TDs ...), + how many
             games that average rests on

Targets: the player's actual passing / rushing / receiving lines that game.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import os
from collections import defaultdict, deque
from datetime import datetime

import pandas as pd

import nfl_common as nc


OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_training_data.csv")

# The stats we model (also the accumulator keys). Combos (pass+rush yds etc.)
# are derived at inference time by summing components, like the NBA engine.
STAT_KEYS = (
    "pass_yds", "pass_att", "completions", "pass_td", "interceptions",
    "rush_yds", "rush_att", "rush_td",
    "rec_yds", "receptions", "targets", "rec_td",
)

# A row is only emitted once the player and the opponent each have enough prior
# games this season for the as-of features to mean something.
MIN_PLAYER_GAMES = 3
MIN_OPP_GAMES = 3
ROLLING_MAX = 8


def num(value):
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def avg(total, count):
    return total / count if count else None


def parse_date(raw):
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def load_positions() -> dict:
    res = nc.supabase.table(nc.PLAYERS_TABLE).select("player_id,position").execute()
    return {r["player_id"]: (r.get("position") or "UNK")
            for r in (res.data or []) if r.get("player_id") is not None}


def fetch_all_logs() -> list:
    columns = ("player_id,player_name,team,opponent,game_id,game_date,season,"
               "season_type,home_away," + ",".join(STAT_KEYS))
    return nc.fetch_all(nc.LOGS_TABLE, columns, order_col="game_date")


class State:
    """Per-player + per-defense accumulators; every read is point-in-time."""

    def __init__(self):
        self.recent = defaultdict(lambda: deque(maxlen=ROLLING_MAX))
        self.season_sum = defaultdict(lambda: defaultdict(float))   # (pid, season)
        self.games_played = defaultdict(int)                        # pid, this season
        self.opp_allowed = defaultdict(lambda: defaultdict(float))  # team -> stat sums
        self.games_defended = defaultdict(int)                      # team -> games

    def player_features(self, pid, season):
        recent = list(self.recent[pid])
        last3, last5 = recent[-3:], recent[-5:]
        ssum = self.season_sum[(pid, season)]
        n = ssum.get("n", 0)
        feats = {"season_games_todate": n}
        for s in STAT_KEYS:
            feats[f"l3_{s}"] = avg(sum(g[s] for g in last3), len(last3))
            feats[f"l5_{s}"] = avg(sum(g[s] for g in last5), len(last5))
            feats[f"season_{s}"] = avg(ssum.get(s, 0.0), n)
        return feats

    def opponent_features(self, opp):
        games = self.games_defended[opp]
        feats = {"opp_games_todate": games}
        for s in STAT_KEYS:
            feats[f"opp_{s}_allowed"] = avg(self.opp_allowed[opp].get(s, 0.0), games)
        return feats

    def add_game(self, g):
        pid, season = g["player_id"], g["season"]
        vals = {s: g[s] for s in STAT_KEYS}
        self.recent[pid].append(vals)
        ssum = self.season_sum[(pid, season)]
        for s in STAT_KEYS:
            ssum[s] += vals[s]
            if g["opponent"]:
                self.opp_allowed[g["opponent"]][s] += vals[s]
        ssum["n"] += 1
        self.games_played[pid] += 1


def main():
    print("Loading player positions ...")
    positions = load_positions()
    print(f"  {len(positions)} players\n")

    print(f"Reading game logs from {nc.LOGS_TABLE} ...")
    raw = fetch_all_logs()
    print(f"Total game-log rows: {len(raw)}\n")

    by_day = defaultdict(list)
    for r in raw:
        gdate = parse_date(r.get("game_date"))
        if gdate is None or r.get("player_id") is None:
            continue
        row = {
            "player_id": r["player_id"],
            "player_name": r.get("player_name"),
            "date": gdate,
            "season": str(r.get("season")),
            "season_type": r.get("season_type") or "regular",
            "opponent": (r.get("opponent") or "").strip() or None,
            "game_id": r.get("game_id"),
            "home": 1 if r.get("home_away") == "HOME" else 0,
            "position": positions.get(r["player_id"], "UNK"),
        }
        for s in STAT_KEYS:
            row[s] = num(r.get(s))
        by_day[gdate].append(row)

    state = State()
    out_rows = []
    for day in sorted(by_day):
        games = by_day[day]

        # 1) READ features from prior state.
        for g in games:
            if state.games_played[g["player_id"]] < MIN_PLAYER_GAMES:
                continue
            if not g["opponent"] or state.games_defended[g["opponent"]] < MIN_OPP_GAMES:
                continue
            row = {
                "player_id": g["player_id"],
                "player_name": g["player_name"],
                "game_date": g["date"].isoformat(),
                "season": g["season"],
                "season_type": g["season_type"],
                "opponent": g["opponent"],
                "position": g["position"],
                "home": g["home"],
            }
            row.update(state.player_features(g["player_id"], g["season"]))
            row.update(state.opponent_features(g["opponent"]))
            for s in STAT_KEYS:
                row[f"target_{s}"] = g[s]
            out_rows.append(row)

        # 2) WRITE the day into state.
        teams_defended_today = set()
        for g in games:
            state.add_game(g)
            if g["opponent"]:
                teams_defended_today.add((g["opponent"], g["game_id"]))
        for team, _gid in teams_defended_today:
            state.games_defended[team] += 1

    df = pd.DataFrame(out_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print("=" * 60)
    print(f"DONE. Wrote {len(df)} training rows to {OUTPUT_CSV}")
    if not df.empty:
        seasons = ", ".join(sorted(str(s) for s in df["season"].dropna().unique()))
        print(f"Seasons covered: {seasons}")


if __name__ == "__main__":
    main()

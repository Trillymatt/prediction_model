"""
Pull per-player box scores for completed NFL games into Supabase.

    # Normal nightly run: games completed in the last few days
    python 32_nfl_player_logs.py

    # Backfill after seeding the schedule (one-time; 2-3 seasons of history
    # makes the prop models real -- load those seasons' schedules first):
    python 32_nfl_player_logs.py --backfill 2023-09-01

    # One game, verbose (for debugging the feed):
    python 32_nfl_player_logs.py --game-id 401671789 --check

The football sibling of 21_soccer_player_logs.py. For every completed game in
nfl_schedule it calls ESPN's public event-summary endpoint and writes one row
per player who recorded a passing / rushing / receiving line into
nfl_player_game_logs, keyed on (player_id, game_id) so re-runs upsert instead
of duplicating. Player ids are ESPN athlete ids -- the same id space as
nfl_players (31_nfl_rosters.py), so autocomplete, the roster view and these
logs all line up.

This is stage 2b of the NFL pipeline (see NFL_SETUP.md): it's the data the
prop models (34/35) and the team-TD target of the game model (36/37) train on.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import time
import argparse
import traceback
from datetime import date, timedelta, datetime

import requests

import nfl_common as nc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOGS_TABLE = nc.LOGS_TABLE
ON_CONFLICT = "player_id,game_id"
API_TIMEOUT = 30
REQUEST_PAUSE = 0.6
BATCH_SIZE = 200

SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
)

# ESPN box-score category -> {espn stat key: our column}. The "cmp/att" style
# combined fields are split by _split_pair below. Keys ESPN doesn't send for a
# category simply stay at their default (0 / None), so a run-only back never
# errors on missing receiving keys.
PASSING = {
    "completions/passingAttempts": ("completions", "pass_att"),
    "passingYards": "pass_yds",
    "passingTouchdowns": "pass_td",
    "interceptions": "interceptions",
}
RUSHING = {
    "rushingAttempts": "rush_att",
    "rushingYards": "rush_yds",
    "rushingTouchdowns": "rush_td",
}
RECEIVING = {
    "receptions": "receptions",
    "receivingTargets": "targets",
    "receivingYards": "rec_yds",
    "receivingTouchdowns": "rec_td",
}
FUMBLES = {
    "fumblesLost": "fumbles_lost",
}
CATEGORY_MAPS = {
    "passing": PASSING,
    "rushing": RUSHING,
    "receiving": RECEIVING,
    "fumbles": FUMBLES,
}

# Every stat column we may write, so each row is dense (0 where a player had no
# line in a category) -- the training builder can then treat missing as zero.
STAT_COLUMNS = [
    "pass_att", "completions", "pass_yds", "pass_td", "interceptions",
    "rush_att", "rush_yds", "rush_td",
    "targets", "receptions", "rec_yds", "rec_td",
    "fumbles_lost",
]


# ---------------------------------------------------------------------------
# ESPN fetch + parse
# ---------------------------------------------------------------------------
def fetch_summary(event_id):
    resp = requests.get(
        SUMMARY_URL,
        params={"event": event_id},
        timeout=API_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (prediction-model nfl log sync)"},
    )
    resp.raise_for_status()
    return resp.json() or {}


def _to_int(value):
    if value in (None, "", "-", "--"):
        return None
    try:
        return int(round(float(str(value).replace(",", ""))))
    except (TypeError, ValueError):
        return None


def _split_pair(value):
    """'20/31' -> (20, 31); tolerant of odd formats."""
    try:
        a, b = str(value).split("/")
        return _to_int(a), _to_int(b)
    except (ValueError, AttributeError):
        return None, None


def parse_boxscore(summary: dict, game: dict):
    """ESPN summary -> a list of nfl_player_game_logs rows for one game."""
    box = (summary.get("boxscore") or {}).get("players") or []
    home = nc.normalize_team(game.get("home_team"))
    away = nc.normalize_team(game.get("away_team"))

    rows = {}   # player_id -> row (merged across passing/rushing/receiving)
    for team_block in box:
        team = nc.normalize_team((team_block.get("team") or {}).get("displayName"))
        if team == home:
            opponent, side = away, "HOME"
        elif team == away:
            opponent, side = home, "AWAY"
        else:  # unknown team (shouldn't happen) -- best effort
            opponent, side = (away if team != away else home), "HOME"

        for cat in team_block.get("statistics") or []:
            colmap = CATEGORY_MAPS.get((cat.get("name") or "").lower())
            if not colmap:
                continue
            keys = cat.get("keys") or []
            for ath in cat.get("athletes") or []:
                athlete = ath.get("athlete") or {}
                pid = _to_int(athlete.get("id"))
                name = athlete.get("displayName")
                if not pid or not name:
                    continue
                stat_vals = dict(zip(keys, ath.get("stats") or []))
                row = rows.get(pid)
                if row is None:
                    row = {
                        "player_id": pid,
                        "player_name": name,
                        "team": team,
                        "opponent": opponent,
                        "game_id": game["game_id"],
                        "game_date": game.get("game_date"),
                        "season": str(game.get("season")) if game.get("season") else None,
                        "season_type": game.get("season_type"),
                        "week": game.get("week"),
                        "home_away": side,
                    }
                    for c in STAT_COLUMNS:
                        row[c] = 0
                    rows[pid] = row
                for key, col in colmap.items():
                    if key not in stat_vals:
                        continue
                    if isinstance(col, tuple):        # combined "cmp/att"
                        made, att = _split_pair(stat_vals[key])
                        row[col[0]] = made or 0
                        row[col[1]] = att or 0
                    else:
                        row[col] = _to_int(stat_vals[key]) or 0
    return list(rows.values())


# ---------------------------------------------------------------------------
# Supabase writes
# ---------------------------------------------------------------------------
_warned_no_upsert = False


def write_logs(rows):
    """Upsert on (player_id, game_id); the unique index is in NFL_SETUP.md."""
    global _warned_no_upsert
    if not rows:
        return 0
    try:
        nc.supabase.table(LOGS_TABLE).upsert(rows, on_conflict=ON_CONFLICT).execute()
        return len(rows)
    except Exception:  # noqa: BLE001 - missing table/index => warn once, re-raise
        if not _warned_no_upsert:
            print("  (!) Upsert on (player_id, game_id) failed -- create the "
                  "nfl_player_game_logs table + unique index from NFL_SETUP.md.")
            _warned_no_upsert = True
        raise


def games_to_process(since: date, game_id=None):
    """Completed games from nfl_schedule we should pull box scores for."""
    filters = [("eq", "status", "completed"),
               ("gte", "game_date", since.isoformat())]
    if game_id:
        filters = [("eq", "game_id", game_id)]
    return nc.fetch_all(
        nc.SCHEDULE_TABLE,
        "game_id,game_date,season,season_type,week,home_team,away_team,status",
        filters=filters,
        order_col="game_date",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sync NFL player box-score logs.")
    parser.add_argument("--days-back", type=int, default=4,
                        help="Process games completed in the last N days (default 4).")
    parser.add_argument("--backfill", metavar="YYYY-MM-DD", default=None,
                        help="Process every completed game since this date.")
    parser.add_argument("--game-id", type=int, default=None,
                        help="Process a single game id.")
    parser.add_argument("--check", action="store_true",
                        help="Parse + print only; write nothing.")
    args = parser.parse_args()

    if args.backfill:
        try:
            since = datetime.strptime(args.backfill, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit("--backfill must be YYYY-MM-DD")
    else:
        since = date.today() - timedelta(days=args.days_back)

    games = games_to_process(since, game_id=args.game_id)
    print(f"{len(games)} completed game(s) to process since {since}\n")

    failures = []
    total_logs = 0
    for i, game in enumerate(games, 1):
        label = (f"{game.get('game_date')} {game.get('away_team')} @ "
                 f"{game.get('home_team')}")
        try:
            summary = fetch_summary(game["game_id"])
            logs = parse_boxscore(summary, game)
        except Exception as exc:  # noqa: BLE001 - one bad game shouldn't kill the run
            failures.append((label, str(exc)))
            print(f"  !! {label}: {exc}")
            time.sleep(REQUEST_PAUSE)
            continue

        if args.check:
            print(f"  {label}: {len(logs)} player rows")
            for row in sorted(logs, key=lambda r: -(r["pass_yds"] + r["rush_yds"]
                                                    + r["rec_yds"]))[:6]:
                print(f"     {row['player_name']} ({row['team']}) "
                      f"{row['pass_yds']}pass {row['rush_yds']}rush "
                      f"{row['rec_yds']}rec  TD {row['pass_td']}/{row['rush_td']}/{row['rec_td']}")
        else:
            try:
                written = 0
                buf = logs
                while buf:
                    chunk, buf = buf[:BATCH_SIZE], buf[BATCH_SIZE:]
                    written += write_logs(chunk)
                total_logs += written
                print(f"  [{i}/{len(games)}] {label}: {written} rows")
            except Exception as exc:  # noqa: BLE001
                failures.append((label, str(exc)))
                print(f"  !! {label} write failed: {exc}")
                traceback.print_exc()
        time.sleep(REQUEST_PAUSE)

    print("\n" + "=" * 50)
    print(f"DONE. Player log rows written: {total_logs}")
    if failures:
        print(f"\n{len(failures)} game(s) failed:")
        for where, err in failures[:20]:
            print(f"  - {where}: {err}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()

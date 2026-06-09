"""
Compute per-player rolling/season averages from stored game logs.

    python 06_player_averages.py

Reads nba_player_game_logs straight from Supabase (no nba_api here) and, for each
player, computes rolling last-5 / last-10 figures plus current-season splits.
Results are upserted into nba_player_averages keyed on player_id, so re-running
refreshes each player's single row rather than piling up duplicates.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY (service_role) in .env
"""

import os
import traceback
from collections import defaultdict
from datetime import date, datetime

from dotenv import load_dotenv
from supabase import create_client, Client


def current_season() -> str:
    """Return the in-progress NBA season label, e.g. '2025-26'.

    The NBA season spans two calendar years and tips off in October, so months
    Oct-Dec belong to the season starting that year; Jan-Sep to the prior one.
    (Same logic as 01_nba_data_pull.py so every script agrees on "current".)
    """
    today = datetime.now()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_TABLE = "nba_player_game_logs"
TABLE_NAME = "nba_player_averages"
ON_CONFLICT = "player_id"
CURRENT_SEASON = current_season()   # e.g. '2025-26'; includes regular season + playoffs
BATCH_SIZE = 100          # rows per Supabase upsert
PAGE_SIZE = 1000          # rows per Supabase select page (PostgREST default cap)

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit(
        "Missing credentials. Set SUPABASE_URL and SUPABASE_KEY in your .env file."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def avg(values):
    """Mean of a list of numbers, rounded to 2 dp; None if the list is empty."""
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 2)


def fetch_all_logs():
    """Pull every game-log row, paging past PostgREST's 1000-row response cap."""
    columns = (
        "player_id,player_name,game_date,season,home_away,"
        "points,rebounds,assists,minutes_played,usage_rate"
    )
    rows = []
    start = 0
    while True:
        res = (
            supabase.table(SOURCE_TABLE)
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


def upsert_batch(batch):
    """Upsert a batch of records into Supabase. Returns count written."""
    if not batch:
        return 0
    supabase.table(TABLE_NAME).upsert(batch, on_conflict=ON_CONFLICT).execute()
    return len(batch)


def build_average_record(player_id, player_name, games):
    """Compute one nba_player_averages row from a player's game list.

    `games` must be sorted ascending by game_date. Rolling windows take the most
    recent N games (the tail); season splits are filtered to CURRENT_SEASON.
    """
    last_5 = games[-5:]
    last_10 = games[-10:]

    season_games = [g for g in games if g.get("season") == CURRENT_SEASON]
    home_games = [g for g in season_games if g.get("home_away") == "HOME"]
    away_games = [g for g in season_games if g.get("home_away") == "AWAY"]

    return {
        "player_id": player_id,
        "player_name": player_name,
        "last_5_points": avg([g.get("points") for g in last_5]),
        "last_10_points": avg([g.get("points") for g in last_10]),
        "last_5_rebounds": avg([g.get("rebounds") for g in last_5]),
        "last_10_rebounds": avg([g.get("rebounds") for g in last_10]),
        "last_5_assists": avg([g.get("assists") for g in last_5]),
        "last_10_assists": avg([g.get("assists") for g in last_10]),
        "last_5_minutes": avg([g.get("minutes_played") for g in last_5]),
        "last_10_minutes": avg([g.get("minutes_played") for g in last_10]),
        "season_avg_points": avg([g.get("points") for g in season_games]),
        "season_avg_rebounds": avg([g.get("rebounds") for g in season_games]),
        "season_avg_assists": avg([g.get("assists") for g in season_games]),
        "home_avg_points": avg([g.get("points") for g in home_games]),
        "away_avg_points": avg([g.get("points") for g in away_games]),
        "usage_rate_last_5": avg([g.get("usage_rate") for g in last_5]),
        "calculated_date": date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Reading game logs from {SOURCE_TABLE} ...")
    try:
        logs = fetch_all_logs()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to read game logs: {exc}")
        traceback.print_exc()
        return
    print(f"Total game-log rows: {len(logs)}\n")

    # Group rows by player, preserving the game_date ascending order from the query.
    by_player = defaultdict(list)
    names = {}
    for row in logs:
        pid = row.get("player_id")
        if pid is None:
            continue
        by_player[pid].append(row)
        if row.get("player_name"):
            names[pid] = row["player_name"]

    total_players = len(by_player)
    print(f"Unique players: {total_players}\n")

    buffer = []
    total_written = 0
    failures = []

    for idx, (player_id, games) in enumerate(by_player.items(), start=1):
        try:
            # Defensive re-sort in case the query order wasn't fully honored.
            games.sort(key=lambda g: g.get("game_date") or "")
            record = build_average_record(player_id, names.get(player_id), games)
            buffer.append(record)

            if idx % 50 == 0 or idx == total_players:
                print(f"[{idx}/{total_players}] computed averages")

            while len(buffer) >= BATCH_SIZE:
                chunk, buffer = buffer[:BATCH_SIZE], buffer[BATCH_SIZE:]
                total_written += upsert_batch(chunk)
                print(f"    -> upserted batch, total written: {total_written}")

        except Exception as exc:  # noqa: BLE001 - keep going on any failure
            failures.append((names.get(player_id, player_id), str(exc)))
            print(f"FAILED ({names.get(player_id, player_id)}): {exc}")
            traceback.print_exc()

    if buffer:
        try:
            total_written += upsert_batch(buffer)
            print(f"    -> upserted final batch, total written: {total_written}")
        except Exception as exc:  # noqa: BLE001 - keep going on any failure
            failures.append(("final batch", str(exc)))
            print(f"FAILED final batch: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"DONE. Total player-average records written: {total_written}")
    if failures:
        print(f"\n{len(failures)} player computations failed:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()

"""
Compute per-player vs. opponent splits from stored game logs.

    python 07_head_to_head.py

Reads nba_player_game_logs from Supabase (no nba_api here) and, for each unique
(player, opponent) pair, computes games_played and average points/rebounds/
assists/minutes -- once across all seasons ('all-time') and once filtered to the
current season ('2024-25'). Both rows are upserted into nba_head_to_head keyed on
(player_id, opponent, season) so re-running refreshes rather than duplicates.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY (service_role) in .env
"""

import os
import traceback
from collections import defaultdict
from datetime import datetime

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
TABLE_NAME = "nba_head_to_head"
ON_CONFLICT = "player_id,opponent_team,season"
CURRENT_SEASON = current_season()   # e.g. '2025-26'; includes regular season + playoffs
ALL_TIME_LABEL = "all-time"
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
        "player_id,player_name,opponent,season,"
        "points,rebounds,assists,minutes_played"
    )
    rows = []
    start = 0
    while True:
        res = (
            supabase.table(SOURCE_TABLE)
            .select(columns)
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


def build_record(player_id, player_name, opponent, season_label, games):
    """Compute one nba_head_to_head row from a player's games vs. one opponent."""
    return {
        "player_id": player_id,
        "player_name": player_name,
        "opponent_team": opponent,
        "season": season_label,
        "games_played": len(games),
        "avg_points": avg([g.get("points") for g in games]),
        "avg_rebounds": avg([g.get("rebounds") for g in games]),
        "avg_assists": avg([g.get("assists") for g in games]),
        "avg_minutes": avg([g.get("minutes_played") for g in games]),
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

    # Bucket games by (player_id, opponent). Keep a parallel current-season bucket.
    all_time = defaultdict(list)
    current = defaultdict(list)
    names = {}

    for row in logs:
        pid = row.get("player_id")
        opp = row.get("opponent")
        if pid is None or not opp:
            continue
        key = (pid, opp)
        all_time[key].append(row)
        if row.get("season") == CURRENT_SEASON:
            current[key].append(row)
        if row.get("player_name"):
            names[pid] = row["player_name"]

    total_pairs = len(all_time)
    print(f"Unique player/opponent pairs: {total_pairs}\n")

    buffer = []
    total_written = 0
    failures = []

    for idx, (key, games) in enumerate(all_time.items(), start=1):
        player_id, opponent = key
        player_name = names.get(player_id)
        try:
            buffer.append(
                build_record(player_id, player_name, opponent, ALL_TIME_LABEL, games)
            )
            # Only emit a current-season row when the pair actually met this season.
            cur_games = current.get(key)
            if cur_games:
                buffer.append(
                    build_record(
                        player_id, player_name, opponent, CURRENT_SEASON, cur_games
                    )
                )

            if idx % 100 == 0 or idx == total_pairs:
                print(f"[{idx}/{total_pairs}] pairs processed")

            while len(buffer) >= BATCH_SIZE:
                chunk, buffer = buffer[:BATCH_SIZE], buffer[BATCH_SIZE:]
                total_written += upsert_batch(chunk)
                print(f"    -> upserted batch, total written: {total_written}")

        except Exception as exc:  # noqa: BLE001 - keep going on any failure
            failures.append((f"{player_name} vs {opponent}", str(exc)))
            print(f"FAILED ({player_name} vs {opponent}): {exc}")
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
    print(f"DONE. Total head-to-head records written: {total_written}")
    if failures:
        print(f"\n{len(failures)} pair computations failed:")
        for label, err in failures:
            print(f"  - {label}: {err}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()

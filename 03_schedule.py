"""
Pull the full NBA schedule (played AND upcoming games) into Supabase.

    python 03_schedule.py

Uses nba_api's ScheduleLeagueV2 -- the league's actual schedule feed -- which
covers every game of the season in one call: completed games (with final
scores), live games, and future games that haven't tipped yet. That last part
matters: the engine's next-game auto-detection reads rows with status
'upcoming', and the nightly refresh gate checks the schedule for yesterday's
games, so future games MUST exist in the table ahead of time. (The previous
version of this script built the schedule from LeagueGameLog, which only
returns games already played -- so 'upcoming' rows never existed and the
refresh gate deadlocked.)

Preseason and All-Star games are skipped. Records are upserted into
nba_schedule keyed on game_id.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY (service_role) in .env
"""

import os
import traceback
from datetime import datetime

from dotenv import load_dotenv
from supabase import create_client, Client

from nba_api.stats.endpoints import scheduleleaguev2


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
SEASON = current_season()   # e.g. '2025-26'
TABLE_NAME = "nba_schedule"
ON_CONFLICT = "game_id"
BATCH_SIZE = 100          # rows per Supabase upsert
API_TIMEOUT = 30          # seconds before an nba_api request times out

# game_id prefixes by game type: 001 preseason, 002 regular season, 003
# All-Star, 004 playoffs, 005 play-in. We keep real games only.
KEEP_GAME_ID_PREFIXES = ("002", "004", "005")

# ScheduleLeagueV2 gameStatus codes: 1 = scheduled, 2 = live, 3 = final.
STATUS_FINAL = 3

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
def to_int(value):
    """Safely coerce a stat to int (the feed returns floats / None / '')."""
    if value is None or value == "":
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def parse_game_date(raw) -> str | None:
    """gameDateEst '2026-06-10T00:00:00Z' -> '2026-06-10' (the US/EST game day)."""
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def upsert_batch(batch):
    """Upsert a batch of records into Supabase. Returns count written."""
    if not batch:
        return 0
    supabase.table(TABLE_NAME).upsert(batch, on_conflict=ON_CONFLICT).execute()
    return len(batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Pulling {SEASON} schedule (completed + upcoming) via ScheduleLeagueV2\n")

    failures = []
    try:
        sched = scheduleleaguev2.ScheduleLeagueV2(season=SEASON, timeout=API_TIMEOUT)
        df = sched.get_data_frames()[0]   # one row per game, both teams flattened
        print(f"Feed returned {len(df)} games")
    except Exception as exc:  # noqa: BLE001 - report and bail; nothing to write
        traceback.print_exc()
        raise SystemExit(f"FAILED to fetch schedule: {exc}")

    records = []
    for _, row in df.iterrows():
        game_id = str(row.get("gameId") or "")
        if not game_id or not game_id.startswith(KEEP_GAME_ID_PREFIXES):
            continue   # preseason / All-Star / junk

        completed = to_int(row.get("gameStatus")) == STATUS_FINAL
        home_score = to_int(row.get("homeTeam_score")) if completed else None
        away_score = to_int(row.get("awayTeam_score")) if completed else None

        records.append({
            "game_id": game_id,
            "game_date": parse_game_date(row.get("gameDateEst")),
            "home_team": (row.get("homeTeam_teamTricode") or "").strip() or None,
            "away_team": (row.get("awayTeam_teamTricode") or "").strip() or None,
            "season": SEASON,
            "status": "completed" if completed else "upcoming",
            "home_score": home_score,
            "away_score": away_score,
        })

    n_done = sum(1 for r in records if r["status"] == "completed")
    print(f"Keeping {len(records)} games ({n_done} completed, "
          f"{len(records) - n_done} upcoming)")

    # Batch-upsert the assembled games.
    total_written = 0
    buffer = records
    while buffer:
        chunk, buffer = buffer[:BATCH_SIZE], buffer[BATCH_SIZE:]
        try:
            total_written += upsert_batch(chunk)
            print(f"    -> upserted batch, total written: {total_written}")
        except Exception as exc:  # noqa: BLE001 - keep going on any failure
            failures.append(("upsert", str(exc)))
            print(f"FAILED upsert batch: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"DONE. Total games written: {total_written} (of {len(records)} assembled)")
    if failures:
        print(f"\n{len(failures)} step(s) failed:")
        for where, err in failures:
            print(f"  - {where}: {err}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()

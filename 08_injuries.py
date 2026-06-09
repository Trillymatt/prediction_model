"""
Pull the current NBA injury report into Supabase.

    python 08_injuries.py

The injury report is a volatile, sometimes-unavailable feed. If the endpoint
errors out or returns nothing, this prints a warning and exits 0 -- it never
crashes. Whatever rows come back are inserted into nba_injuries tagged with
today's date as game_date.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY (service_role) in .env
"""

import os
import time
import traceback
from datetime import date

from dotenv import load_dotenv
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TABLE_NAME = "nba_injuries"
BATCH_SIZE = 100          # rows per Supabase insert
API_DELAY = 0.6           # seconds between API calls (rate-limit guard)
API_TIMEOUT = 30          # seconds before an nba_api request times out

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
def clean(value):
    """Strip/collapse whitespace from a string; pass through non-strings."""
    if value is None:
        return None
    if isinstance(value, str):
        return " ".join(value.split()) or None
    return value


def first_present(row: dict, keys):
    """Return the first non-empty value among `keys` in a row, else None."""
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def fetch_injury_rows():
    """Fetch the injury report rows, tolerating an unavailable endpoint.

    nba_api ships an InjuryReport endpoint under stats.endpoints; it is flaky and
    occasionally absent depending on the installed version, so the import itself
    is guarded. Returns a list of raw row dicts (possibly empty).
    """
    try:
        from nba_api.stats.endpoints import injuryreport
    except ImportError:
        print("WARNING: nba_api has no InjuryReport endpoint in this install.")
        return []

    report = injuryreport.InjuryReport(timeout=API_TIMEOUT)
    data = report.get_normalized_dict()
    # Grab whichever result set actually carries rows.
    for key, rows in data.items():
        if rows:
            return rows
    return []


def build_record(row: dict, game_date: str) -> dict:
    """Map an injury-report row into the nba_injuries schema (whitespace-clean)."""
    return {
        "player_id": first_present(row, ["PLAYER_ID", "PlayerID", "player_id"]),
        "player_name": clean(
            first_present(row, ["PLAYER_NAME", "PlayerName", "PLAYER", "player_name"])
        ),
        "team": clean(
            first_present(row, ["TEAM", "TEAM_ABBREVIATION", "TeamName", "team"])
        ),
        "status": clean(
            first_present(row, ["STATUS", "CURRENT_STATUS", "Status", "status"])
        ),
        "reason": clean(
            first_present(row, ["REASON", "COMMENT", "Reason", "reason", "DESCRIPTION"])
        ),
        "game_date": game_date,
    }


def insert_batch(batch):
    """Insert a batch of records into Supabase. Returns count written."""
    if not batch:
        return 0
    supabase.table(TABLE_NAME).insert(batch).execute()
    return len(batch)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    today = date.today().isoformat()
    print(f"Pulling NBA injury report for {today} ...")

    try:
        rows = fetch_injury_rows()
    except Exception as exc:  # noqa: BLE001 - never crash on a flaky feed
        print(f"WARNING: injury report endpoint failed: {exc}")
        traceback.print_exc()
        print("Exiting gracefully without writing anything.")
        return
    finally:
        time.sleep(API_DELAY)

    if not rows:
        print("WARNING: injury report returned no data. Exiting gracefully.")
        return

    print(f"Injury report rows: {len(rows)}\n")

    buffer = []
    total_written = 0
    failures = []

    for idx, row in enumerate(rows, start=1):
        try:
            record = build_record(row, today)
            # Skip rows with no identifiable player.
            if not record["player_name"] and record["player_id"] is None:
                continue
            buffer.append(record)

            while len(buffer) >= BATCH_SIZE:
                chunk, buffer = buffer[:BATCH_SIZE], buffer[BATCH_SIZE:]
                total_written += insert_batch(chunk)
                print(f"    -> inserted batch, total written: {total_written}")

        except Exception as exc:  # noqa: BLE001 - keep going on any failure
            failures.append((f"row {idx}", str(exc)))
            print(f"FAILED (row {idx}): {exc}")
            traceback.print_exc()

    if buffer:
        try:
            total_written += insert_batch(buffer)
            print(f"    -> inserted final batch, total written: {total_written}")
        except Exception as exc:  # noqa: BLE001
            failures.append(("final batch", str(exc)))
            print(f"FAILED final batch: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"DONE. Total injury records written: {total_written}")
    if failures:
        print(f"\n{len(failures)} row inserts failed:")
        for label, err in failures:
            print(f"  - {label}: {err}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()

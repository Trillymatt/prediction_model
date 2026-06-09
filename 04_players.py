"""
Pull the current-season active player roster into Supabase.

    python 04_players.py

Uses CommonAllPlayers (current season only) to get the active player list, then
CommonPlayerInfo per player for the bio details. This table drives frontend
autocomplete, so every string is whitespace-stripped before insert. Records are
upserted into nba_players keyed on player_id.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY (service_role) in .env
"""

import os
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv
from supabase import create_client, Client

from nba_api.stats.endpoints import commonallplayers, commonplayerinfo


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
TABLE_NAME = "nba_players"
ON_CONFLICT = "player_id"
BATCH_SIZE = 100          # rows per Supabase upsert
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
    """Strip surrounding/extra whitespace from a string; pass through non-strings."""
    if value is None:
        return None
    if isinstance(value, str):
        # Collapse internal runs of whitespace and trim the ends.
        return " ".join(value.split()) or None
    return value


def to_int(value):
    """Safely coerce a value to int (nba_api returns floats / None / '')."""
    if value is None or value == "":
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def age_from_birthdate(raw):
    """Compute current age in years from a CommonPlayerInfo BIRTHDATE.

    BIRTHDATE comes back like '1998-02-28T00:00:00'. Returns None if unparseable.
    """
    if not raw:
        return None
    try:
        born = datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    today = datetime.now().date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def upsert_batch(batch):
    """Upsert a batch of records into Supabase. Returns count written."""
    if not batch:
        return 0
    supabase.table(TABLE_NAME).upsert(batch, on_conflict=ON_CONFLICT).execute()
    return len(batch)


def fetch_active_players():
    """Return list of {'id', 'name'} for current-season active players."""
    allplayers = commonallplayers.CommonAllPlayers(
        is_only_current_season=1,
        league_id="00",
        season=SEASON,
        timeout=API_TIMEOUT,
    )
    rows = allplayers.get_normalized_dict().get("CommonAllPlayers", [])
    return [
        {"id": row.get("PERSON_ID"), "name": clean(row.get("DISPLAY_FIRST_LAST"))}
        for row in rows
        if row.get("PERSON_ID")
    ]


def build_record(info: dict) -> dict:
    """Map a CommonPlayerInfo row into the nba_players schema (whitespace-clean)."""
    full_name = clean(info.get("DISPLAY_FIRST_LAST"))
    if not full_name:
        first = clean(info.get("FIRST_NAME")) or ""
        last = clean(info.get("LAST_NAME")) or ""
        full_name = clean(f"{first} {last}")
    return {
        "player_id": info.get("PERSON_ID"),
        "player_name": full_name,
        "team": clean(info.get("TEAM_NAME")),
        "position": clean(info.get("POSITION")),
        "age": age_from_birthdate(info.get("BIRTHDATE")),
        "height": clean(info.get("HEIGHT")),
        "weight": clean(info.get("WEIGHT")),
        "jersey_number": clean(info.get("JERSEY")),
        "status": clean(info.get("ROSTERSTATUS")),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Fetching active players for {SEASON} ...")
    try:
        active_players = fetch_active_players()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to fetch active player list: {exc}")
        traceback.print_exc()
        return
    time.sleep(API_DELAY)

    total_players = len(active_players)
    print(f"Active players: {total_players}\n")

    buffer = []
    total_written = 0
    failures = []

    for idx, player in enumerate(active_players, start=1):
        player_id = player["id"]
        player_name = player["name"]

        try:
            print(f"[{idx}/{total_players}] {player_name} ...", end=" ", flush=True)

            info = commonplayerinfo.CommonPlayerInfo(
                player_id=player_id,
                timeout=API_TIMEOUT,
            )
            rows = info.get_normalized_dict().get("CommonPlayerInfo", [])
            if not rows:
                print("no info returned, skipping")
                continue

            record = build_record(rows[0])
            # Fall back to the roster name if CommonPlayerInfo gave us nothing.
            if not record["player_name"]:
                record["player_name"] = player_name
            buffer.append(record)
            print("ok")

            while len(buffer) >= BATCH_SIZE:
                chunk, buffer = buffer[:BATCH_SIZE], buffer[BATCH_SIZE:]
                total_written += upsert_batch(chunk)
                print(f"    -> upserted batch, total written: {total_written}")

        except Exception as exc:  # noqa: BLE001 - keep going on any failure
            failures.append((player_name, str(exc)))
            print(f"FAILED: {exc}")
            traceback.print_exc()

        finally:
            time.sleep(API_DELAY)  # always wait between API calls

    if buffer:
        try:
            total_written += upsert_batch(buffer)
            print(f"    -> upserted final batch, total written: {total_written}")
        except Exception as exc:  # noqa: BLE001 - keep going on any failure
            failures.append(("final batch", str(exc)))
            print(f"FAILED final batch: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"DONE. Total players written: {total_written}")
    if failures:
        print(f"\n{len(failures)} player pulls failed:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()

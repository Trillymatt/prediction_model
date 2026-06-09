"""
Pull NBA player game logs and load them into Supabase.

Two modes:
  * Backfill (one-time):   python 01_nba_data_pull.py --full
        Pulls all SEASONS for every active player. Use this once to seed
        an empty table.

  * Update (default):      python 01_nba_data_pull.py
        Pulls only the current season, and only games newer than the most
        recent game already stored. Fast, safe to run repeatedly (e.g. daily).

Both regular season and playoff games are pulled (tagged via season_type).

Either way, records are UPSERTed on (player_id, game_date), so re-running never
creates duplicates -- existing rows are updated, new games are added.
(Requires a unique constraint on (player_id, game_date); see README/SQL.)

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY (service_role) in .env
"""

import os
import time
import argparse
import traceback
from datetime import datetime

from dotenv import load_dotenv
from supabase import create_client, Client

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]   # used by --full backfill
SEASON_TYPES = ["Regular Season", "Playoffs"]            # pulled for each season
TABLE_NAME = "nba_player_game_logs"
ON_CONFLICT = "player_id,game_date"           # matches the table's unique constraint
BATCH_SIZE = 100          # rows per Supabase upsert
API_DELAY = 0.6           # seconds between player API calls (rate-limit guard)
API_TIMEOUT = 30          # seconds before an nba_api request times out

# Resolve .env relative to this file so it works from any working directory
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit(
        "Missing credentials. Set SUPABASE_URL and SUPABASE_KEY in your .env file."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Optional columns (prop stats the table may not have yet). If they exist in
# Supabase, we map them from the API; if not, we leave them out of the upsert
# so it doesn't error. Add them with:
#   ALTER TABLE nba_player_game_logs
#     ADD COLUMN fouls int, ADD COLUMN oreb int, ADD COLUMN dreb int;
# then re-pull history once (python 01_nba_data_pull.py --full) to backfill.
OPTIONAL_COLUMNS = {"fouls": "PF", "oreb": "OREB", "dreb": "DREB"}


def _optional_columns_present() -> bool:
    """True if the optional columns exist in nba_player_game_logs."""
    try:
        supabase.table(TABLE_NAME).select(",".join(OPTIONAL_COLUMNS)).limit(1).execute()
        return True
    except Exception:  # noqa: BLE001 - columns absent
        return False


HAS_OPTIONAL_COLUMNS = _optional_columns_present()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def current_season() -> str:
    """Return the in-progress NBA season label, e.g. '2025-26'.

    The NBA season spans two calendar years and tips off in October, so months
    Oct-Dec belong to the season starting that year; Jan-Sep to the prior one.
    """
    today = datetime.now()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def parse_matchup(matchup: str):
    """Turn a MATCHUP string into (team, opponent, home_away).

    "GSW vs. LAL" -> ("GSW", "LAL", "HOME")
    "GSW @ LAL"   -> ("GSW", "LAL", "AWAY")
    """
    if not matchup:
        return None, None, None
    if " vs. " in matchup:
        team, opponent = matchup.split(" vs. ")
        return team.strip(), opponent.strip(), "HOME"
    if " @ " in matchup:
        team, opponent = matchup.split(" @ ")
        return team.strip(), opponent.strip(), "AWAY"
    return matchup.strip(), None, None


def parse_game_date(raw: str):
    """'OCT 25, 2022' -> ISO date string '2022-10-25' (or None)."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw.title(), "%b %d, %Y").date().isoformat()
    except ValueError:
        return None


def to_int(value):
    """Safely coerce a stat to int (nba_api returns floats / None / '')."""
    if value is None or value == "":
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def build_record(row: dict, player_name: str, season: str, season_type: str) -> dict:
    """Map a PlayerGameLog row into our table schema."""
    team, opponent, home_away = parse_matchup(row.get("MATCHUP"))
    extra = (
        {col: to_int(row.get(api_key)) for col, api_key in OPTIONAL_COLUMNS.items()}
        if HAS_OPTIONAL_COLUMNS else {}
    )
    return {
        **extra,
        "player_id": row.get("Player_ID"),
        "player_name": player_name,
        "game_date": parse_game_date(row.get("GAME_DATE")),
        "season": season,
        "season_type": season_type,
        "team": team,
        "opponent": opponent,
        "home_away": home_away,
        "win_loss": row.get("WL"),
        "minutes_played": to_int(row.get("MIN")),
        "points": to_int(row.get("PTS")),
        "rebounds": to_int(row.get("REB")),
        "assists": to_int(row.get("AST")),
        "steals": to_int(row.get("STL")),
        "blocks": to_int(row.get("BLK")),
        "turnovers": to_int(row.get("TOV")),
        "fg_made": to_int(row.get("FGM")),
        "fg_attempted": to_int(row.get("FGA")),
        "three_made": to_int(row.get("FG3M")),
        "three_attempted": to_int(row.get("FG3A")),
        "ft_made": to_int(row.get("FTM")),
        "ft_attempted": to_int(row.get("FTA")),
    }


def latest_stored_date():
    """Most recent game_date already in the table (ISO str), or None if empty."""
    res = (
        supabase.table(TABLE_NAME)
        .select("game_date")
        .order("game_date", desc=True)
        .limit(1)
        .execute()
    )
    if res.data and res.data[0].get("game_date"):
        return res.data[0]["game_date"]
    return None


def iso_to_api_date(iso_date: str) -> str:
    """'2025-01-15' -> '01/15/2025' (the format nba_api's date_from expects)."""
    return datetime.strptime(iso_date, "%Y-%m-%d").strftime("%m/%d/%Y")


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
    parser = argparse.ArgumentParser(description="Pull NBA game logs into Supabase.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Backfill all SEASONS for every player (use once to seed the table). "
             "Without this flag, runs in update mode (current season, new games only).",
    )
    args = parser.parse_args()

    if args.full:
        seasons = SEASONS
        date_from = None
        print(f"FULL BACKFILL mode. Seasons: {', '.join(seasons)}")
    else:
        seasons = [current_season()]
        date_from = latest_stored_date()
        if date_from:
            print(f"UPDATE mode. Season: {seasons[0]} | only games on/after {date_from}")
        else:
            print(f"UPDATE mode. Season: {seasons[0]} | table empty, pulling whole season")

    active_players = players.get_active_players()
    total_players = len(active_players)
    print(f"Active players: {total_players}\n")

    buffer = []           # accumulates records until we hit BATCH_SIZE
    total_written = 0
    failures = []

    for idx, player in enumerate(active_players, start=1):
        player_id = player["id"]
        player_name = player["full_name"]

        for season in seasons:
            for season_type in SEASON_TYPES:
                try:
                    print(
                        f"[{idx}/{total_players}] {player_name} - {season} {season_type} ...",
                        end=" ",
                        flush=True,
                    )

                    log = playergamelog.PlayerGameLog(
                        player_id=player_id,
                        season=season,
                        season_type_all_star=season_type,
                        date_from_nullable=iso_to_api_date(date_from) if date_from else "",
                        timeout=API_TIMEOUT,
                    )
                    rows = log.get_normalized_dict().get("PlayerGameLog", [])
                    print(f"{len(rows)} games")

                    for row in rows:
                        buffer.append(build_record(row, player_name, season, season_type))

                        while len(buffer) >= BATCH_SIZE:
                            chunk, buffer = buffer[:BATCH_SIZE], buffer[BATCH_SIZE:]
                            total_written += upsert_batch(chunk)
                            print(f"    -> upserted batch, total written: {total_written}")

                except Exception as exc:  # noqa: BLE001 - keep going on any failure
                    failures.append((player_name, f"{season} {season_type}", str(exc)))
                    print(f"FAILED: {exc}")
                    traceback.print_exc()

                finally:
                    time.sleep(API_DELAY)  # always wait between API calls

    # Flush whatever is left over in the buffer
    if buffer:
        total_written += upsert_batch(buffer)
        print(f"    -> upserted final batch, total written: {total_written}")

    # Summary
    print("\n" + "=" * 50)
    print(f"DONE. Total records written: {total_written}")
    if failures:
        print(f"\n{len(failures)} player/season pulls failed:")
        for name, season, err in failures:
            print(f"  - {name} ({season}): {err}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()

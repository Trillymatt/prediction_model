"""
Pull NBA team stats and load them into Supabase.

    python 02_team_stats.py

Uses nba_api's LeagueDashTeamStats (Base + Advanced measure types) to gather
per-team season stats for 2022-23, 2023-24 and 2024-25, then upserts them into
nba_team_stats keyed on (team_id, season).

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

from nba_api.stats.endpoints import leaguedashteamstats


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
# Past seasons for team-strength history, plus the in-progress season so the
# table always reflects the current year. De-duped in case current is already listed.
SEASONS = list(dict.fromkeys(["2022-23", "2023-24", "2024-25", current_season()]))
SEASON_TYPES = ["Regular Season", "Playoffs"]   # both pulled and stored separately
TABLE_NAME = "nba_team_stats"
ON_CONFLICT = "team_id,season,season_type"
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
def to_int(value):
    """Safely coerce a stat to int (nba_api returns floats / None / '')."""
    if value is None or value == "":
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def to_float(value):
    """Safely coerce a stat to float (nba_api returns strings / None / '')."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_team_stats(season: str, season_type: str, measure_type: str) -> dict:
    """Return {team_id: row_dict} for a season + season type + measure type.

    measure_type is one of 'Base', 'Advanced', 'Opponent'. 'Opponent' is what
    surfaces OPP_PTS (opponent points per game); 'Advanced' carries pace and the
    offensive/defensive ratings. season_type is 'Regular Season' or 'Playoffs'.
    """
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense=measure_type,
        per_mode_detailed="PerGame",
        timeout=API_TIMEOUT,
    )
    rows = stats.get_normalized_dict().get("LeagueDashTeamStats", [])
    return {row.get("TEAM_ID"): row for row in rows}


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
    print(f"Pulling team stats for seasons: {', '.join(SEASONS)}\n")

    buffer = []
    total_written = 0
    failures = []

    for season in SEASONS:
        for season_type in SEASON_TYPES:
            try:
                print(
                    f"[{season} {season_type}] fetching Base + Advanced + Opponent ...",
                    flush=True,
                )

                base = fetch_team_stats(season, season_type, "Base")
                time.sleep(API_DELAY)
                advanced = fetch_team_stats(season, season_type, "Advanced")
                time.sleep(API_DELAY)
                opponent = fetch_team_stats(season, season_type, "Opponent")
                time.sleep(API_DELAY)

                # Early in a season (or for season types with no games yet) the
                # endpoint returns zero teams -- just skip, nothing to write.
                print(f"[{season} {season_type}] {len(base)} teams")

                for team_id, base_row in base.items():
                    adv_row = advanced.get(team_id, {})
                    opp_row = opponent.get(team_id, {})
                    buffer.append({
                        "team_id": team_id,
                        "team_name": (base_row.get("TEAM_NAME") or "").strip(),
                        "season": season,
                        "season_type": season_type,
                        "wins": to_int(base_row.get("W")),
                        "losses": to_int(base_row.get("L")),
                        "points_per_game": to_float(base_row.get("PTS")),
                        "opponent_points_per_game": to_float(opp_row.get("OPP_PTS")),
                        "pace": to_float(adv_row.get("PACE")),
                        "offensive_rating": to_float(adv_row.get("OFF_RATING")),
                        "defensive_rating": to_float(adv_row.get("DEF_RATING")),
                    })

                    while len(buffer) >= BATCH_SIZE:
                        chunk, buffer = buffer[:BATCH_SIZE], buffer[BATCH_SIZE:]
                        total_written += upsert_batch(chunk)
                        print(f"    -> upserted batch, total written: {total_written}")

            except Exception as exc:  # noqa: BLE001 - keep going on any failure
                failures.append((f"{season} {season_type}", str(exc)))
                print(f"FAILED ({season} {season_type}): {exc}")
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
    print(f"DONE. Total team-season records written: {total_written}")
    if failures:
        print(f"\n{len(failures)} season pulls failed:")
        for season, err in failures:
            print(f"  - {season}: {err}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()

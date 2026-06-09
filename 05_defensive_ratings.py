"""
Compute Defense-vs-Position (points allowed by position) into Supabase.

    python 05_defensive_ratings.py

NOTE ON SOURCE: the originally-specified nba_api endpoint (LeagueDashPtDefend)
does NOT expose points allowed by position -- it only reports a *defender's*
field-goal defense (D_FGM/D_FGA/D_FG_PCT). So instead we compute the real metric
the table wants from data already in Supabase:

  For each defending team, take every opposing player's points, bucket them by
  that player's position (looked up from nba_players), sum per team+position, and
  divide by the number of games that team played. That yields true "points
  allowed per game" to Guards / Forwards / Centers.

Computed for the current season, split by season type (Regular Season / Playoffs).
Upserted into nba_defensive_ratings on (team_id, season, position, season_type).

CAVEAT: positions are only known for players currently in nba_players. Points
scored by players without a stored position bucket into position 'Unknown'.

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

from nba_api.stats.static import teams


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
PLAYERS_TABLE = "nba_players"
TABLE_NAME = "nba_defensive_ratings"
ON_CONFLICT = "team_id,season,position,season_type"
SEASON = current_season()   # e.g. '2025-26'; includes regular season + playoffs
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
def position_bucket(position) -> str:
    """Normalize a CommonPlayerInfo position to a primary bucket G / F / C.

    'Guard' -> 'G', 'Forward-Center' -> 'F' (primary, the part before the dash),
    'Center' -> 'C'. Anything missing/unrecognized -> 'Unknown'.
    """
    if not position:
        return "Unknown"
    primary = position.split("-")[0].strip().lower()
    return {"guard": "G", "forward": "F", "center": "C"}.get(primary, "Unknown")


def to_float(value):
    """Safely coerce a stat to float (returns None / '' tolerant)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_player_positions() -> dict:
    """Return {player_id: position_bucket} from nba_players."""
    res = supabase.table(PLAYERS_TABLE).select("player_id,position").execute()
    return {
        r["player_id"]: position_bucket(r.get("position"))
        for r in (res.data or [])
        if r.get("player_id") is not None
    }


def build_team_maps():
    """Return (abbrev -> team_id, team_id -> team_name) from nba_api static data."""
    abbrev_to_id = {}
    id_to_name = {}
    for t in teams.get_teams():
        abbrev_to_id[t["abbreviation"]] = t["id"]
        id_to_name[t["id"]] = t["full_name"]
    return abbrev_to_id, id_to_name


def fetch_season_logs():
    """Pull this season's game-log rows, paging past PostgREST's 1000-row cap."""
    columns = "player_id,opponent,points,game_date,season,season_type"
    rows = []
    start = 0
    while True:
        res = (
            supabase.table(SOURCE_TABLE)
            .select(columns)
            .eq("season", SEASON)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Computing Defense-vs-Position for {SEASON}\n")

    try:
        positions = load_player_positions()
        print(f"Loaded positions for {len(positions)} players")
        abbrev_to_id, id_to_name = build_team_maps()
        print(f"Loaded {len(id_to_name)} NBA teams")
        logs = fetch_season_logs()
        print(f"Season game-log rows: {len(logs)}\n")
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED to load source data: {exc}")
        traceback.print_exc()
        return

    # Accumulators:
    #   points_allowed[(team_id, position, season_type)] = total points conceded
    #   games[(team_id, season_type)]                    = set of game_dates defended
    points_allowed = defaultdict(float)
    games = defaultdict(set)
    unmapped_opponents = set()

    for row in logs:
        opp = (row.get("opponent") or "").strip()
        team_id = abbrev_to_id.get(opp)
        if team_id is None:
            if opp:
                unmapped_opponents.add(opp)
            continue

        season_type = row.get("season_type") or "Regular Season"
        pos = positions.get(row.get("player_id"), "Unknown")
        pts = to_float(row.get("points"))
        game_date = row.get("game_date")

        if pts is not None:
            points_allowed[(team_id, pos, season_type)] += pts
        if game_date:
            games[(team_id, season_type)].add(game_date)

    if unmapped_opponents:
        print(f"NOTE: {len(unmapped_opponents)} opponent codes had no team match: "
              f"{', '.join(sorted(unmapped_opponents))}")

    # Build one row per (team, position, season_type): points conceded / games played.
    buffer = []
    total_written = 0
    failures = []

    for (team_id, pos, season_type), total in points_allowed.items():
        try:
            game_count = len(games[(team_id, season_type)]) or 1
            buffer.append({
                "team_id": team_id,
                "team_name": id_to_name.get(team_id),
                "season": SEASON,
                "season_type": season_type,
                "position": pos,
                "points_allowed_per_game": round(total / game_count, 2),
            })

            while len(buffer) >= BATCH_SIZE:
                chunk, buffer = buffer[:BATCH_SIZE], buffer[BATCH_SIZE:]
                total_written += upsert_batch(chunk)
                print(f"    -> upserted batch, total written: {total_written}")

        except Exception as exc:  # noqa: BLE001 - keep going on any failure
            failures.append((f"team {team_id} {pos} {season_type}", str(exc)))
            print(f"FAILED (team {team_id} {pos} {season_type}): {exc}")
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
    print(f"DONE. Total defensive-rating records written: {total_written}")
    if failures:
        print(f"\n{len(failures)} writes failed:")
        for label, err in failures:
            print(f"  - {label}: {err}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()

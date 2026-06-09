"""
Validate the data loaded into nba_player_game_logs.

Run this after 01_nba_data_pull.py finishes (or any time) to sanity-check the
table: row counts, per-season breakdown, null checks, date range, and a
spot-check of one known player's per-season averages.

    python 02_validate_nba.py
    python 02_validate_nba.py --player "Stephen Curry"
"""

import os
import argparse

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

TABLE_NAME = "nba_player_game_logs"
SEASONS = ["2022-23", "2023-24", "2024-25"]

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Missing SUPABASE_URL / SUPABASE_KEY in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def count_rows(**filters) -> int:
    """Exact row count, optionally filtered by equality on columns."""
    q = supabase.table(TABLE_NAME).select("id", count="exact")
    for col, val in filters.items():
        q = q.eq(col, val)
    return q.limit(1).execute().count


def count_nulls(column: str) -> int:
    return (
        supabase.table(TABLE_NAME)
        .select("id", count="exact")
        .is_(column, "null")
        .limit(1)
        .execute()
        .count
    )


def date_extreme(desc: bool):
    res = (
        supabase.table(TABLE_NAME)
        .select("game_date")
        .order("game_date", desc=desc)
        .limit(1)
        .execute()
    )
    return res.data[0]["game_date"] if res.data else None


def fetch_player(name: str) -> pd.DataFrame:
    """Pull all rows for one player (players have < ~300 games, well under the
    1000-row default page limit)."""
    res = supabase.table(TABLE_NAME).select("*").eq("player_name", name).execute()
    return pd.DataFrame(res.data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", default="LeBron James",
                        help="Player name to spot-check (default: LeBron James)")
    args = parser.parse_args()

    print("=" * 60)
    print("NBA GAME LOG VALIDATION")
    print("=" * 60)

    # --- Total + per-season counts ---
    total = count_rows()
    print(f"\nTotal rows: {total:,}")
    if total == 0:
        print("Table is empty -- run 01_nba_data_pull.py --full first.")
        return

    print("\nRows per season:")
    accounted = 0
    for s in SEASONS:
        c = count_rows(season=s)
        accounted += c
        print(f"  {s}: {c:,}")
    other = total - accounted
    if other:
        print(f"  (other seasons): {other:,}")

    # --- Null checks on fields that should never be null ---
    print("\nNull checks (should all be 0):")
    for col in ["player_id", "game_date", "season", "points", "minutes_played"]:
        n = count_nulls(col)
        flag = "" if n == 0 else "  <-- INVESTIGATE"
        print(f"  null {col}: {n:,}{flag}")

    # --- Date range ---
    print(f"\nDate range: {date_extreme(desc=False)}  ->  {date_extreme(desc=True)}")

    # --- Spot check one player ---
    print("\n" + "-" * 60)
    print(f"SPOT CHECK: {args.player}")
    print("-" * 60)
    df = fetch_player(args.player)
    if df.empty:
        print("No rows found for that player (check spelling).")
        return

    df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"Total games: {len(df)}")
    print(f"Most recent game: {df['game_date'].max().date()}")

    print("\nPer-season averages:")
    agg = (
        df.groupby("season")
        .agg(games=("points", "size"),
             ppg=("points", "mean"),
             rpg=("rebounds", "mean"),
             apg=("assists", "mean"),
             mpg=("minutes_played", "mean"))
        .round(1)
        .sort_index()
    )
    print(agg.to_string())

    print("\nMost recent 5 games:")
    recent = df.sort_values("game_date", ascending=False).head(5)
    cols = ["game_date", "season", "team", "opponent", "home_away",
            "win_loss", "minutes_played", "points", "rebounds", "assists"]
    print(recent[cols].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Nightly refresh: re-pull data and retrain the model -- but ONLY if a game was
actually played, so off-nights cost nothing.

    python refresh.py            # gated: skips if no game yesterday
    python refresh.py --force    # run the whole pipeline regardless

Meant to be run from cron once a night (see the README / setup below). The guard
checks nba_schedule for any game dated "yesterday"; if there were none, there's
nothing new to ingest and it exits early.

When it does run, it executes the pipeline in dependency order:
    01 game logs -> 02 team stats -> 03 schedule -> 04 players ->
    05 defense-vs-pos -> 06 averages -> 07 head-to-head -> 08 injuries ->
    10 rebuild training data -> 11 retrain model

Each step's output is streamed; a failing step is logged but doesn't abort the
rest (the data scripts are all idempotent upserts, so a partial run is safe).

Cron example (run 4am daily):
    0 4 * * * cd /path/to/money_from_a_baby && /usr/bin/python3 refresh.py >> refresh.log 2>&1
"""

import sys
import time
import subprocess
from datetime import date, timedelta

import os
from dotenv import load_dotenv
from supabase import create_client, Client

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Missing SUPABASE_URL / SUPABASE_KEY in .env")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Pipeline, in dependency order. (04 players before 05; 01 before 05/06/07/10;
# 10 before 11; 12 before 13.)
PIPELINE = [
    "01_nba_data_pull.py",
    "02_team_stats.py",
    "03_schedule.py",
    "04_players.py",
    "05_defensive_ratings.py",
    "06_player_averages.py",
    "07_head_to_head.py",
    "08_injuries.py",
    "10_build_training_data.py",
    "11_train_model.py",
    "12_build_game_training_data.py",
    "13_train_game_model.py",
]


def games_were_played(on_day: date) -> bool:
    """True if nba_schedule has any game dated `on_day` (a proxy for 'a game was played').

    03_schedule.py loads the FULL season schedule (upcoming games included), so
    future game dates exist in the table ahead of time and this check stays
    accurate even across skipped nights.
    """
    res = (
        supabase.table("nba_schedule")
        .select("game_id")
        .eq("game_date", on_day.isoformat())
        .limit(1)
        .execute()
    )
    return bool(res.data)


def schedule_is_stale() -> bool:
    """True if nba_schedule holds no future games -- the table needs reseeding.

    Guards against the deadlock that bit the v1 gate: the old schedule script
    only stored already-played games, so once a night was skipped the table
    never saw a new date again and the gate skipped forever. If there's nothing
    upcoming, we can't trust 'no games yesterday' and should refresh anyway.
    (The offseason also triggers this; that costs one no-op pipeline run per
    night, which the idempotent upserts make harmless.)
    """
    res = (
        supabase.table("nba_schedule")
        .select("game_id")
        .gte("game_date", date.today().isoformat())
        .limit(1)
        .execute()
    )
    return not res.data


def run_step(script: str) -> bool:
    """Run one pipeline script, streaming its output. Returns True on success."""
    print(f"\n{'=' * 60}\n>>> {script}\n{'=' * 60}", flush=True)
    result = subprocess.run([sys.executable, os.path.join(HERE, script)])
    ok = result.returncode == 0
    if not ok:
        print(f"!! {script} exited with code {result.returncode}", flush=True)
    return ok


def main():
    force = "--force" in sys.argv
    yesterday = date.today() - timedelta(days=1)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] refresh starting (yesterday = {yesterday})")

    if not force and not games_were_played(yesterday):
        if schedule_is_stale():
            print("Schedule table has no future games -- reseeding via full run.")
        else:
            print("No games found for yesterday -- nothing to refresh. Exiting.")
            return

    print("Games detected (or --force). Running pipeline.\n")
    failures = []
    for script in PIPELINE:
        if not run_step(script):
            failures.append(script)

    print(f"\n{'=' * 60}")
    if failures:
        print(f"DONE with {len(failures)} failed step(s): {', '.join(failures)}")
    else:
        print("DONE. All steps succeeded; data + model refreshed.")


if __name__ == "__main__":
    main()

"""
Pull the current NFL injury report into Supabase (nfl_injuries).

    python 33_nfl_injuries.py
    python 33_nfl_injuries.py --check     # fetch + parse only, print a sample

The football sibling of 08_injuries.py. Uses ESPN's public league-wide injuries
endpoint (site.api.espn.com -- no key needed), which groups every team's injury
report together. Rows are inserted tagged with today's date as game_date, and
keyed by ESPN athlete id so the projection engine can join them to a player the
same way the NBA side does.

Like the NBA injury puller, this is defensive: a flaky/absent feed prints a
warning and exits 0 rather than crashing the nightly pipeline.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import time
import argparse
import traceback
from datetime import date

import requests

import nfl_common as nc


TABLE_NAME = nc.INJURIES_TABLE
API_TIMEOUT = 30
BATCH_SIZE = 200

# League-wide report (all teams in one call). If ESPN changes the shape we fall
# back to whatever list of injury entries we can find.
INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
)


def clean(value):
    if value is None:
        return None
    if isinstance(value, str):
        return " ".join(value.split()) or None
    return value


def fetch_injuries():
    resp = requests.get(
        INJURIES_URL,
        timeout=API_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (prediction-model nfl injury sync)"},
    )
    resp.raise_for_status()
    return resp.json() or {}


def parse_injuries(payload: dict, game_date: str):
    """ESPN league injuries payload -> nfl_injuries rows.

    Shape (as of 2026): payload['injuries'] is a list of team blocks, each with
    a team display name and an 'injuries' list of athlete entries. We read the
    athlete id/name/position and the status + best-available reason string.
    """
    rows = []
    for team_block in payload.get("injuries") or []:
        team = nc.normalize_team(
            team_block.get("displayName")
            or (team_block.get("team") or {}).get("displayName")
        )
        for entry in team_block.get("injuries") or []:
            athlete = entry.get("athlete") or {}
            pid = athlete.get("id")
            name = clean(athlete.get("displayName") or athlete.get("fullName"))
            if pid is None and not name:
                continue
            position = ((athlete.get("position") or {}).get("abbreviation")
                        or (athlete.get("position") or {}).get("name"))
            status = clean(entry.get("status") or (entry.get("type") or {}).get("description"))
            reason = clean(
                entry.get("longComment") or entry.get("shortComment")
                or (entry.get("details") or {}).get("type")
                or entry.get("detail")
            )
            try:
                pid = int(pid) if pid is not None else None
            except (TypeError, ValueError):
                pid = None
            rows.append({
                "player_id": pid,
                "player_name": name,
                "team": team,
                "position": position,
                "status": status,
                "reason": reason,
                "game_date": game_date,
            })
    return rows


def insert_batch(batch):
    if not batch:
        return 0
    nc.supabase.table(TABLE_NAME).insert(batch).execute()
    return len(batch)


def main():
    parser = argparse.ArgumentParser(description="Sync the NFL injury report.")
    parser.add_argument("--check", action="store_true",
                        help="Fetch + parse only; print a sample, write nothing.")
    args = parser.parse_args()

    today = date.today().isoformat()
    print(f"Pulling NFL injury report for {today} ...")

    try:
        payload = fetch_injuries()
        rows = parse_injuries(payload, today)
    except Exception as exc:  # noqa: BLE001 - never crash on a flaky feed
        print(f"WARNING: injury endpoint failed: {exc}")
        traceback.print_exc()
        print("Exiting gracefully without writing anything.")
        return

    rows = [r for r in rows if r["player_name"] or r["player_id"] is not None]
    print(f"Injury rows parsed: {len(rows)}\n")

    if args.check:
        for r in rows[:12]:
            print(f"  {r['player_name']} ({r['position']}) - {r['team']}: "
                  f"{r['status']} — {r['reason']}")
        print("\n--check: nothing written.")
        return

    if not rows:
        print("No injuries returned. Exiting gracefully.")
        return

    total, buf, failures = 0, rows, []
    while buf:
        chunk, buf = buf[:BATCH_SIZE], buf[BATCH_SIZE:]
        try:
            total += insert_batch(chunk)
        except Exception as exc:  # noqa: BLE001
            failures.append(str(exc))
            print(f"FAILED insert batch: {exc}")
            traceback.print_exc()
        time.sleep(0.2)

    print("\n" + "=" * 50)
    print(f"DONE. Injury records written: {total}")
    if failures:
        print(f"{len(failures)} batch(es) failed (create nfl_injuries per NFL_SETUP.md).")


if __name__ == "__main__":
    main()

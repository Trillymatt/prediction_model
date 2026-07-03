"""
Pull NFL team rosters into Supabase (nfl_players).

    python 31_nfl_rosters.py
    python 31_nfl_rosters.py --check     # fetch + parse only, print a sample

This is the "player directory" ingestion: it's what powers autocomplete and
the per-game roster view, but it carries no stats -- box scores / game logs
(the piece props need) come with the next pipeline stage, see NFL_SETUP.md.

Uses ESPN's public team + roster APIs (site.api.espn.com -- no key needed):
first the team list (to get each team's ESPN id, since URL slugs don't
always match our abbreviations), then one roster call per team. Rosters
change slowly (waivers/trades), so this is cheap to run nightly alongside
the schedule puller.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import time
import argparse
import traceback

import requests

import nfl_common as nc

TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
ROSTER_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"
)
TABLE_NAME = nc.PLAYERS_TABLE
ON_CONFLICT = "player_id"
BATCH_SIZE = 200
API_TIMEOUT = 30
REQUEST_PAUSE = 0.4


def fetch_teams():
    """[(espn_team_id, canonical_team_name), ...] for all 32 teams."""
    resp = requests.get(
        TEAMS_URL,
        params={"limit": 50},
        timeout=API_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (prediction-model roster sync)"},
    )
    resp.raise_for_status()
    leagues = (resp.json() or {}).get("sports", [{}])[0].get("leagues", [{}])
    entries = (leagues[0] if leagues else {}).get("teams", [])
    out = []
    for e in entries:
        t = e.get("team") or {}
        name = nc.normalize_team(t.get("displayName"))
        if t.get("id") and name:
            out.append((t["id"], name))
    return out


def fetch_roster(team_id):
    resp = requests.get(
        ROSTER_URL.format(team_id=team_id),
        timeout=API_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (prediction-model roster sync)"},
    )
    resp.raise_for_status()
    return (resp.json() or {}).get("athletes", [])


def parse_group(group: dict, team_name: str):
    """One position-group entry (offense/defense/specialTeam) -> records."""
    out = []
    for a in group.get("items") or []:
        try:
            pid = int(a["id"])
        except (KeyError, TypeError, ValueError):
            continue
        name = a.get("fullName") or a.get("displayName")
        if not name:
            continue
        pos = ((a.get("position") or {}).get("abbreviation") or "").strip() or None
        out.append({
            "player_id": pid,
            "player_name": name,
            "team": team_name,
            "position": pos,
        })
    return out


def upsert_batch(batch):
    if not batch:
        return 0
    nc.supabase.table(TABLE_NAME).upsert(batch, on_conflict=ON_CONFLICT).execute()
    return len(batch)


def main():
    parser = argparse.ArgumentParser(description="Sync NFL team rosters.")
    parser.add_argument("--check", action="store_true",
                        help="Fetch + parse only; print a sample, write nothing.")
    args = parser.parse_args()

    print("Fetching NFL team list...")
    teams = fetch_teams()
    print(f"Found {len(teams)} teams\n")

    records = {}
    failures = []
    for team_id, team_name in teams:
        try:
            groups = fetch_roster(team_id)
        except Exception as exc:  # noqa: BLE001 - one bad team shouldn't kill the run
            failures.append((team_name, str(exc)))
            print(f"  !! {team_name}: {exc}")
            continue
        count = 0
        for group in groups:
            for rec in parse_group(group, team_name):
                records[rec["player_id"]] = rec
                count += 1
        print(f"  {team_name}: {count} players")
        time.sleep(REQUEST_PAUSE)

    rows = list(records.values())
    print(f"\nAssembled {len(rows)} players across {len(teams)} teams")

    if args.check:
        for r in rows[:10]:
            print(f"  {r['player_id']} {r['player_name']} ({r['position']}) - {r['team']}")
        print("\n--check: nothing written.")
        return

    total_written = 0
    buffer = rows
    while buffer:
        chunk, buffer = buffer[:BATCH_SIZE], buffer[BATCH_SIZE:]
        try:
            total_written += upsert_batch(chunk)
        except Exception as exc:  # noqa: BLE001 - keep going on any failure
            failures.append(("upsert", str(exc)))
            print(f"FAILED upsert batch: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"DONE. Total players written: {total_written} (of {len(rows)} assembled)")
    if failures:
        print(f"\n{len(failures)} step(s) failed:")
        for where, err in failures[:20]:
            print(f"  - {where}: {err}")


if __name__ == "__main__":
    main()

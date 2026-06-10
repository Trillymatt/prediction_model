"""
Pull international soccer fixtures + results into Supabase (soccer_schedule).

    # Normal nightly run: refresh a window around today (default 7 back / 40 ahead)
    python 20_soccer_schedule.py

    # Backfill history for Elo/form ratings (one-time before the World Cup):
    python 20_soccer_schedule.py --backfill 2024-01-01

    # Sanity-check the ESPN feed without writing anything:
    python 20_soccer_schedule.py --check

The soccer sibling of 03_schedule.py. Uses ESPN's public scoreboard API
(site.api.espn.com -- no key needed) for the FIFA World Cup, every confed's
qualifiers, friendlies, and the major continental tournaments, so the Elo /
form ratings see each national team's real recent results, not just WC games.

Rows are upserted into soccer_schedule keyed on match_id (ESPN's event id),
with status 'upcoming' / 'completed' matching the NBA conventions, so the
engines and the nightly-refresh gate work identically.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import sys
import time
import argparse
import traceback
from datetime import date, timedelta, datetime

import requests

import soccer_common as sc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TABLE_NAME = sc.SCHEDULE_TABLE
ON_CONFLICT = "match_id"
BATCH_SIZE = 100
API_TIMEOUT = 30
REQUEST_PAUSE = 0.6          # seconds between ESPN calls -- be polite
CHUNK_DAYS = 30              # days per scoreboard request when backfilling

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
)

# ESPN league slug -> competition name (shared with 21 via soccer_common).
LEAGUES = sc.LEAGUES


# ---------------------------------------------------------------------------
# ESPN fetch + parse
# ---------------------------------------------------------------------------
def fetch_scoreboard(league: str, start: date, end: date):
    """One scoreboard call for a league + date range. Returns the events list."""
    params = {
        "dates": f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}",
        "limit": 500,
    }
    resp = requests.get(
        SCOREBOARD_URL.format(league=league),
        params=params,
        timeout=API_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (prediction-model schedule sync)"},
    )
    resp.raise_for_status()
    return (resp.json() or {}).get("events", [])


def parse_event(event: dict, competition: str):
    """One ESPN event -> a soccer_schedule record (or None if malformed)."""
    try:
        comp = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors") or []
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            return None

        status_obj = ((event.get("status") or {}).get("type") or {})
        completed = bool(status_obj.get("completed"))
        state = status_obj.get("state")  # 'pre' | 'in' | 'post'
        status = "completed" if completed else ("live" if state == "in" else "upcoming")

        raw_date = str(event.get("date") or "")           # e.g. 2026-06-11T20:00Z
        match_date = raw_date[:10] or None
        match_time = raw_date[11:16] or None              # HH:MM UTC

        def score(c):
            try:
                return int(float(c.get("score")))
            except (TypeError, ValueError):
                return None

        season = (event.get("season") or {}).get("year")
        return {
            "match_id": int(event["id"]),
            "match_date": match_date,
            "match_time": match_time,
            "competition": competition,
            "season": str(season) if season else (match_date or "")[:4] or None,
            "home_team": sc.normalize_team(
                ((home.get("team") or {}).get("displayName"))
            ) or None,
            "away_team": sc.normalize_team(
                ((away.get("team") or {}).get("displayName"))
            ) or None,
            "status": status,
            "home_score": score(home) if completed else None,
            "away_score": score(away) if completed else None,
        }
    except (KeyError, TypeError, ValueError):
        return None


def date_chunks(start: date, end: date):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def upsert_batch(batch):
    if not batch:
        return 0
    sc.supabase.table(TABLE_NAME).upsert(batch, on_conflict=ON_CONFLICT).execute()
    return len(batch)


def main():
    parser = argparse.ArgumentParser(description="Sync international soccer schedule.")
    parser.add_argument("--backfill", metavar="YYYY-MM-DD", default=None,
                        help="Pull everything from this date to today+60 (one-time).")
    parser.add_argument("--days-back", type=int, default=7,
                        help="Days of recent results to refresh (default 7).")
    parser.add_argument("--days-ahead", type=int, default=40,
                        help="Days of upcoming fixtures to load (default 40).")
    parser.add_argument("--check", action="store_true",
                        help="Fetch + parse only; print a sample, write nothing.")
    args = parser.parse_args()

    if args.backfill:
        try:
            start = datetime.strptime(args.backfill, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit("--backfill must be YYYY-MM-DD")
        end = date.today() + timedelta(days=60)
    else:
        start = date.today() - timedelta(days=args.days_back)
        end = date.today() + timedelta(days=args.days_ahead)

    print(f"Syncing soccer schedule {start} -> {end} "
          f"across {len(LEAGUES)} competitions\n")

    records, failures = {}, []
    for league, competition in LEAGUES.items():
        league_count = 0
        for c_start, c_end in date_chunks(start, end):
            try:
                events = fetch_scoreboard(league, c_start, c_end)
            except Exception as exc:  # noqa: BLE001 - a bad slug shouldn't kill the run
                failures.append((f"{league} {c_start}", str(exc)))
                print(f"  !! {league} {c_start}->{c_end}: {exc}")
                continue
            for ev in events:
                rec = parse_event(ev, competition)
                if rec and rec["home_team"] and rec["away_team"]:
                    records[rec["match_id"]] = rec   # dedupe across chunks
                    league_count += 1
            time.sleep(REQUEST_PAUSE)
        print(f"  {competition}: {league_count} matches")

    rows = list(records.values())
    n_done = sum(1 for r in rows if r["status"] == "completed")
    print(f"\nAssembled {len(rows)} matches ({n_done} completed, "
          f"{len(rows) - n_done} upcoming/live)")

    if args.check:
        for r in rows[:10]:
            print(f"  {r['match_date']} {r['home_team']} vs {r['away_team']} "
                  f"[{r['competition']}] {r['status']} "
                  f"{r['home_score']}-{r['away_score']}")
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
    print(f"DONE. Total matches written: {total_written} (of {len(rows)} assembled)")
    if failures:
        print(f"\n{len(failures)} step(s) failed:")
        for where, err in failures[:20]:
            print(f"  - {where}: {err}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()

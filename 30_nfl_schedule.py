"""
Pull the NFL schedule + results into Supabase (nfl_schedule).

    # Normal nightly run: refresh a window around today (default 7 back / 30 ahead)
    python 30_nfl_schedule.py

    # Load a whole season (schedules are published in the spring):
    python 30_nfl_schedule.py --backfill 2026-08-01 --days-ahead 200

    # Sanity-check the ESPN feed without writing anything:
    python 30_nfl_schedule.py --check

The football sibling of 20_soccer_schedule.py. Uses ESPN's public NFL
scoreboard API (site.api.espn.com -- no key needed) for preseason, regular
season and playoffs. Rows are upserted into nfl_schedule keyed on game_id
(ESPN's event id), with status 'upcoming' / 'live' / 'completed' matching the
conventions on the NBA/soccer sides, so the same refresh gate and boards work.

This is stage 1 of the NFL pipeline (see NFL_SETUP.md for the table SQL and
the roadmap: stats ingestion and projection engines come next).

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import time
import argparse
import traceback
from datetime import date, timedelta, datetime, timezone

import requests

import nfl_common as nc

try:
    from zoneinfo import ZoneInfo
    EASTERN = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001 - no tz database => fall back to UTC dates
    EASTERN = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TABLE_NAME = nc.SCHEDULE_TABLE
ON_CONFLICT = "game_id"
BATCH_SIZE = 100
API_TIMEOUT = 30
REQUEST_PAUSE = 0.6          # seconds between ESPN calls -- be polite
CHUNK_DAYS = 30              # days per scoreboard request

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
)

# ESPN season.type -> our season_type label.
SEASON_TYPES = {1: "preseason", 2: "regular", 3: "playoffs", 4: "offseason"}


# ---------------------------------------------------------------------------
# ESPN fetch + parse
# ---------------------------------------------------------------------------
def fetch_scoreboard(start: date, end: date):
    """One scoreboard call for a date range. Returns the events list."""
    params = {
        "dates": f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}",
        "limit": 500,
    }
    resp = requests.get(
        SCOREBOARD_URL,
        params=params,
        timeout=API_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (prediction-model schedule sync)"},
    )
    resp.raise_for_status()
    return (resp.json() or {}).get("events", [])


def parse_event(event: dict):
    """One ESPN event -> an nfl_schedule record (or None if malformed)."""
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

        # ESPN dates are UTC; store the US/Eastern game day + kickoff so a
        # Sunday-night game doesn't land on Monday (same convention as the
        # NBA and soccer sides).
        raw_date = str(event.get("date") or "")
        game_date = raw_date[:10] or None
        game_time = raw_date[11:16] or None
        if EASTERN is not None:
            try:
                dt_utc = datetime.strptime(
                    raw_date[:16], "%Y-%m-%dT%H:%M"
                ).replace(tzinfo=timezone.utc)
                local = dt_utc.astimezone(EASTERN)
                game_date = local.date().isoformat()
                game_time = local.strftime("%H:%M")      # HH:MM Eastern
            except ValueError:
                pass

        def score(c):
            try:
                return int(float(c.get("score")))
            except (TypeError, ValueError):
                return None

        season = event.get("season") or {}
        week = (event.get("week") or {}).get("number")
        return {
            "game_id": int(event["id"]),
            "game_date": game_date,
            "game_time": game_time,
            "season": str(season.get("year")) if season.get("year") else
                      (game_date or "")[:4] or None,
            "season_type": SEASON_TYPES.get(season.get("type"), "regular"),
            "week": int(week) if week is not None else None,
            "home_team": nc.normalize_team(
                ((home.get("team") or {}).get("displayName"))
            ) or None,
            "away_team": nc.normalize_team(
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
    nc.supabase.table(TABLE_NAME).upsert(batch, on_conflict=ON_CONFLICT).execute()
    return len(batch)


def main():
    parser = argparse.ArgumentParser(description="Sync the NFL schedule.")
    parser.add_argument("--backfill", metavar="YYYY-MM-DD", default=None,
                        help="Pull everything from this date forward (one-time).")
    parser.add_argument("--days-back", type=int, default=7,
                        help="Days of recent results to refresh (default 7).")
    parser.add_argument("--days-ahead", type=int, default=30,
                        help="Days of upcoming games to load (default 30).")
    parser.add_argument("--check", action="store_true",
                        help="Fetch + parse only; print a sample, write nothing.")
    args = parser.parse_args()

    if args.backfill:
        try:
            start = datetime.strptime(args.backfill, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit("--backfill must be YYYY-MM-DD")
        end = date.today() + timedelta(days=max(args.days_ahead, 60))
    else:
        start = date.today() - timedelta(days=args.days_back)
        end = date.today() + timedelta(days=args.days_ahead)

    print(f"Syncing NFL schedule {start} -> {end}\n")

    records, failures = {}, []
    for c_start, c_end in date_chunks(start, end):
        try:
            events = fetch_scoreboard(c_start, c_end)
        except Exception as exc:  # noqa: BLE001 - one bad chunk shouldn't kill the run
            failures.append((f"scoreboard {c_start}", str(exc)))
            print(f"  !! {c_start}->{c_end}: {exc}")
            continue
        for ev in events:
            rec = parse_event(ev)
            if rec and rec["home_team"] and rec["away_team"]:
                records[rec["game_id"]] = rec   # dedupe across chunks
        time.sleep(REQUEST_PAUSE)

    rows = list(records.values())
    n_done = sum(1 for r in rows if r["status"] == "completed")
    print(f"Assembled {len(rows)} games ({n_done} completed, "
          f"{len(rows) - n_done} upcoming/live)")

    if args.check:
        for r in rows[:10]:
            print(f"  {r['game_date']} {r['game_time']} "
                  f"{r['away_team']} @ {r['home_team']} "
                  f"[{r['season_type']} wk {r['week']}] {r['status']} "
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
    print(f"DONE. Total games written: {total_written} (of {len(rows)} assembled)")
    if failures:
        print(f"\n{len(failures)} step(s) failed:")
        for where, err in failures[:20]:
            print(f"  - {where}: {err}")


if __name__ == "__main__":
    main()

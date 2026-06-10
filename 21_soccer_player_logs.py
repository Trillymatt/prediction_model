"""
Pull per-player match stats for completed internationals into Supabase.

    # Normal nightly run: matches completed in the last few days
    python 21_soccer_player_logs.py

    # Backfill after seeding the schedule (one-time before the World Cup):
    python 21_soccer_player_logs.py --backfill 2024-01-01

    # One match, verbose (for debugging the feed):
    python 21_soccer_player_logs.py --match-id 740123 --check

The soccer sibling of 01_nba_data_pull.py. For every completed match in
soccer_schedule it calls ESPN's public match-summary endpoint and writes one
row per player who appeared into soccer_player_match_logs. It also maintains
the soccer_players directory table (id/name/team/position) that powers the
frontend's autocomplete -- see SOCCER_SETUP.md for the one-time SQL.

What ESPN provides per player: goals, assists, shots, shots on target,
fouls, cards, saves, starter/sub. It does NOT provide xG/xA or key passes --
those columns stay NULL unless you load them from another source (the
engines handle NULLs fine). Minutes are taken from the feed when present,
otherwise estimated (starter 90 / sub 30) and that estimate is good enough
for the per-90 rates the engine uses.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import time
import argparse
import traceback
from datetime import date, timedelta, datetime

import requests

import soccer_common as sc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOGS_TABLE = sc.LOGS_TABLE
PLAYERS_TABLE = sc.PLAYERS_TABLE
API_TIMEOUT = 30
REQUEST_PAUSE = 0.6
BATCH_SIZE = 200

SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/summary"
)

# ESPN stat name (lowercased) -> our column. Several spellings are listed per
# column because the feed has renamed stats over time.
STAT_MAP = {
    "totalgoals": "goals", "goals": "goals", "goalstotal": "goals",
    "goalassists": "assists", "assists": "assists",
    "totalshots": "shots", "shots": "shots", "shotattempts": "shots",
    "shotsontarget": "shots_on_target", "shotsongoal": "shots_on_target",
    "yellowcards": "yellow_cards", "redcards": "red_cards",
    "minutes": "minutes_played", "minutesplayed": "minutes_played",
}

# Stats whose columns may not exist in soccer_player_match_logs yet (see
# SOCCER_SETUP.md). Probed once at startup -- if you add the columns, they
# start filling automatically on the next run.
OPTIONAL_STAT_MAP = {
    "foulscommitted": "fouls_committed",
    "foulssuffered": "fouls_suffered",
    "saves": "saves",
    "totalpasses": "passes", "passes": "passes",
}


def optional_columns_present():
    """Which optional log columns actually exist in Supabase right now.

    Shared probe (soccer_common.optional_log_columns) so the puller and the
    projection engine can never disagree about which columns are live.
    """
    try:
        return sc.optional_log_columns()
    except Exception:  # noqa: BLE001 - probe failure => write base columns only
        return set()


# ---------------------------------------------------------------------------
# ESPN fetch + parse
# ---------------------------------------------------------------------------
def fetch_summary(league: str, event_id):
    resp = requests.get(
        SUMMARY_URL.format(league=league),
        params={"event": event_id},
        timeout=API_TIMEOUT,
        headers={"User-Agent": "Mozilla/5.0 (prediction-model log sync)"},
    )
    resp.raise_for_status()
    return resp.json() or {}


def to_int(value):
    if value in (None, "", "-"):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def player_stats_dict(entry: dict) -> dict:
    """Flatten a roster entry's stats list into {lower_name: value}."""
    out = {}
    for s in entry.get("stats") or []:
        name = str(s.get("name") or s.get("abbreviation") or "").lower()
        if name:
            out[name] = s.get("value", s.get("displayValue"))
    return out


def appeared(entry: dict, stats: dict) -> bool:
    """Did this rostered player actually get on the pitch?"""
    if entry.get("starter"):
        return True
    sub = entry.get("subbedIn")
    if isinstance(sub, dict):
        return bool(sub.get("didSub", True))
    if sub:
        return True
    apps = to_int(stats.get("appearances"))
    return bool(apps)


def parse_match_rosters(summary: dict, match: dict, optional_cols: set):
    """ESPN summary -> (log_rows, player_rows) for one completed match."""
    logs, players = [], []
    rosters = summary.get("rosters") or []
    teams = {}
    for r in rosters:
        side = r.get("homeAway")
        if side in ("home", "away"):
            teams[side] = sc.normalize_team(
                ((r.get("team") or {}).get("displayName"))
            )

    for r in rosters:
        side = r.get("homeAway")
        if side not in ("home", "away"):
            continue
        team = teams.get(side)
        opponent = teams.get("away" if side == "home" else "home")
        for entry in r.get("roster") or []:
            athlete = entry.get("athlete") or {}
            pid = to_int(athlete.get("id"))
            name = athlete.get("displayName")
            if not pid or not name:
                continue
            stats = player_stats_dict(entry)
            if not appeared(entry, stats):
                continue

            row = {
                "player_id": pid,
                "player_name": name,
                "match_id": match["match_id"],
                "match_date": match["match_date"],
                "competition": match.get("competition"),
                "season": match.get("season"),
                "team": team,
                "opponent": opponent,
                "home_away": side.upper(),
                "goals": 0, "assists": 0, "shots": 0, "shots_on_target": 0,
                "yellow_cards": 0, "red_cards": 0,
                "minutes_played": None,
                "xg": None, "xa": None, "key_passes": None,
            }
            for espn_name, col in STAT_MAP.items():
                if espn_name in stats:
                    val = to_int(stats[espn_name])
                    row[col] = val if col == "minutes_played" else (val or 0)
            for espn_name, col in OPTIONAL_STAT_MAP.items():
                if col in optional_cols and espn_name in stats:
                    row[col] = to_int(stats[espn_name])
            if row["minutes_played"] is None:
                row["minutes_played"] = 90 if entry.get("starter") else 30

            position = ((entry.get("position") or {}).get("abbreviation")
                        or (athlete.get("position") or {}).get("abbreviation"))
            players.append({
                "player_id": pid,
                "player_name": name,
                "team": team,
                "position": position,
            })
            logs.append(row)
    return logs, players


# ---------------------------------------------------------------------------
# Supabase writes
# ---------------------------------------------------------------------------
_warned_no_upsert = False


def write_logs(rows):
    """Upsert log rows on (player_id, match_id); fall back to dedupe+insert.

    The clean path needs the match_id column + unique index from
    SOCCER_SETUP.md. Without it we check which (player_id, match_id) pairs
    already exist and insert only the new ones -- slower but never duplicates.
    """
    global _warned_no_upsert
    if not rows:
        return 0
    try:
        sc.supabase.table(LOGS_TABLE).upsert(
            rows, on_conflict="player_id,match_id"
        ).execute()
        return len(rows)
    except Exception:  # noqa: BLE001 - missing column/constraint => fallback
        if not _warned_no_upsert:
            print("  (!) Upsert on (player_id, match_id) unavailable -- run the "
                  "SQL in SOCCER_SETUP.md. Falling back to dedupe-insert.")
            _warned_no_upsert = True

    slim = [{k: v for k, v in r.items() if k != "match_id"} for r in rows]
    try:
        existing = (
            sc.supabase.table(LOGS_TABLE)
            .select("player_id")
            .eq("match_date", rows[0]["match_date"])
            .eq("competition", rows[0]["competition"] or "")
            .execute()
        ).data or []
    except Exception:  # noqa: BLE001
        existing = []
    seen = {e.get("player_id") for e in existing}
    fresh = [r for r in slim if r["player_id"] not in seen]
    if fresh:
        sc.supabase.table(LOGS_TABLE).insert(fresh).execute()
    return len(fresh)


_warned_no_players_table = False


def write_players(rows):
    """Upsert the soccer_players directory; tolerate the table being absent."""
    global _warned_no_players_table
    if not rows:
        return
    dedup = {r["player_id"]: r for r in rows}
    try:
        sc.supabase.table(PLAYERS_TABLE).upsert(
            list(dedup.values()), on_conflict="player_id"
        ).execute()
    except Exception:  # noqa: BLE001 - table missing => autocomplete falls back
        if not _warned_no_players_table:
            print("  (!) soccer_players table not found -- player autocomplete "
                  "will fall back to the logs table. See SOCCER_SETUP.md.")
            _warned_no_players_table = True


def matches_to_process(since: date, match_id=None):
    """Completed matches from soccer_schedule we should pull rosters for."""
    filters = [("eq", "status", "completed"),
               ("gte", "match_date", since.isoformat())]
    if match_id:
        filters = [("eq", "match_id", match_id)]
    return sc.fetch_all(
        sc.SCHEDULE_TABLE,
        "match_id,match_date,competition,season,home_team,away_team,status",
        filters=filters,
        order_col="match_date",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sync soccer player match logs.")
    parser.add_argument("--days-back", type=int, default=4,
                        help="Process matches completed in the last N days (default 4).")
    parser.add_argument("--backfill", metavar="YYYY-MM-DD", default=None,
                        help="Process every completed match since this date.")
    parser.add_argument("--match-id", type=int, default=None,
                        help="Process a single match id.")
    parser.add_argument("--check", action="store_true",
                        help="Parse + print only; write nothing.")
    args = parser.parse_args()

    if args.backfill:
        try:
            since = datetime.strptime(args.backfill, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit("--backfill must be YYYY-MM-DD")
    else:
        since = date.today() - timedelta(days=args.days_back)

    matches = matches_to_process(since, match_id=args.match_id)
    print(f"{len(matches)} completed match(es) to process since {since}\n")
    optional_cols = optional_columns_present()
    if optional_cols:
        print(f"Optional columns active: {', '.join(sorted(optional_cols))}\n")

    failures = []
    total_logs = 0
    for i, match in enumerate(matches, 1):
        slug = sc.COMPETITION_TO_SLUG.get(match.get("competition"))
        if not slug:
            continue   # competition we don't know how to query
        label = (f"{match.get('match_date')} {match.get('home_team')} vs "
                 f"{match.get('away_team')}")
        try:
            summary = fetch_summary(slug, match["match_id"])
            logs, players = parse_match_rosters(summary, match, optional_cols)
        except Exception as exc:  # noqa: BLE001 - one bad match shouldn't kill the run
            failures.append((label, str(exc)))
            print(f"  !! {label}: {exc}")
            time.sleep(REQUEST_PAUSE)
            continue

        if args.check:
            print(f"  {label}: {len(logs)} player rows")
            for row in logs[:6]:
                print(f"     {row['player_name']} ({row['team']}) "
                      f"{row['minutes_played']}min {row['goals']}g {row['assists']}a "
                      f"{row['shots']}sh {row['shots_on_target']}sot")
        else:
            try:
                written = write_logs(logs)
                write_players(players)
                total_logs += written
                print(f"  [{i}/{len(matches)}] {label}: {written} rows")
            except Exception as exc:  # noqa: BLE001
                failures.append((label, str(exc)))
                print(f"  !! {label} write failed: {exc}")
                traceback.print_exc()
        time.sleep(REQUEST_PAUSE)

    print("\n" + "=" * 50)
    print(f"DONE. Player log rows written: {total_logs}")
    if failures:
        print(f"\n{len(failures)} match(es) failed:")
        for where, err in failures[:20]:
            print(f"  - {where}: {err}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()

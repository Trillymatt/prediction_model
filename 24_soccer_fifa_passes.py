"""
Enrich soccer player logs with passes attempted from FIFA's official API.

    # Normal nightly run: World Cup matches completed in the last few days
    python 24_soccer_fifa_passes.py

    # Backfill every completed World Cup match since a date:
    python 24_soccer_fifa_passes.py --backfill 2026-06-11

    # One match, verbose (for debugging the feed / name matching):
    python 24_soccer_fifa_passes.py --match-id 760415 --check

ESPN's international match-summary feed (21_soccer_player_logs.py) carries no
pass stats, so the `passes` column sits NULL and the projection engine keeps
the stat hidden. This script fills it for FIFA World Cup matches from FIFA's
public match-data API and the engine picks it up automatically.

Why FIFA and not FBref: FBref lost its Opta data in the 2025 licensing split
and now serves only basic match stats -- match reports, competition passing
pages and player matchlogs were all checked (June 2026) and the passing
columns are empty sitewide, including retroactively for Euro 2024 / WC 2022.
FIFA's own feed is the authoritative source for the World Cup:

    api.fifa.com/api/v3/calendar/matches      match ids per date window
    api.fifa.com/api/v3/live/football/...     lineups (player id -> name)
    fdh-api.fifa.com/v1/stats/match/{ifes}/players.json
                                              per-player stats incl. Passes

Coverage is World Cup finals only -- qualifiers and friendlies have no stats
on the data hub (verified: WCQ returns 404, friendlies carry no IFES id).
That is fine: passes props are a World Cup market and history accumulates
with every completed match. Tackles/key passes are NOT in FIFA's feed, so
those columns stay NULL.

Players are matched ESPN-row <-> FIFA-stat by name. FIFA strips accents and
upper-cases surnames ("Cesar MONTES"); ESPN keeps accents ("César Montes"),
so both sides are accent-folded before comparing (same problem the engine's
accent-insensitive search solves, see 22_soccer_projections.py).

Setup: nothing new -- uses the same .env / requests / supabase stack as 21.
"""

import re
import time
import argparse
import unicodedata
import traceback
from datetime import date, timedelta, datetime

import requests

import soccer_common as sc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOGS_TABLE = sc.LOGS_TABLE
API_TIMEOUT = 30
REQUEST_PAUSE = 1.0           # polite gap between FIFA API calls

FIFA_WC_COMPETITION = "17"    # FIFA World Cup on api.fifa.com
CALENDAR_URL = "https://api.fifa.com/api/v3/calendar/matches"
LIVE_URL = "https://api.fifa.com/api/v3/live/football/{comp}/{season}/{stage}/{match}"
FDH_PLAYERS_URL = "https://fdh-api.fifa.com/v1/stats/match/{ifes}/players.json"

HEADERS = {"User-Agent": "Mozilla/5.0 (prediction-model pass-stats sync; "
                         "mattknorman@gmail.com)"}

# FIFA stat id -> our log column. Passes = attempted (PassesCompleted exists
# too but books price attempts). Extend here if FIFA adds more useful ids.
FIFA_STAT_MAP = {
    "Passes": "passes",
}


def fold_name(name) -> str:
    """Accent-fold + lowercase a player name for cross-feed comparison."""
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", ascii_only.lower().replace("-", " ")).strip()


# ---------------------------------------------------------------------------
# FIFA fetch + parse
# ---------------------------------------------------------------------------
def fifa_get(url, params=None):
    resp = requests.get(url, params=params, timeout=API_TIMEOUT, headers=HEADERS)
    resp.raise_for_status()
    time.sleep(REQUEST_PAUSE)
    return resp.json() or {}


def fifa_calendar(since: date, until: date) -> dict:
    """Completed FIFA WC matches keyed by (team-pair frozenset) per date.

    Returns {(iso_date, frozenset({home, away})): match_record}. Teams are
    normalized through sc.normalize_team so they join with soccer_schedule
    (FIFA says "Korea Republic" / "IR Iran"; ESPN says "South Korea" /
    "Iran" -- the alias table covers both spellings).
    """
    data = fifa_get(CALENDAR_URL, params={
        "idCompetition": FIFA_WC_COMPETITION,
        "from": f"{since.isoformat()}T00:00:00Z",
        "to": f"{until.isoformat()}T23:59:59Z",
        "count": 500, "language": "en",
    })
    index = {}
    for m in data.get("Results") or []:
        if m.get("MatchStatus") != 0:      # 0 = full time
            continue
        home = (((m.get("Home") or {}).get("TeamName")) or [{}])[0].get("Description")
        away = (((m.get("Away") or {}).get("TeamName")) or [{}])[0].get("Description")
        d = sc.parse_date(m.get("Date"))
        if not home or not away or not d:
            continue
        pair = frozenset({sc.normalize_team(home).lower(),
                          sc.normalize_team(away).lower()})
        # FIFA dates are UTC; ESPN's schedule date can sit a day either side.
        for offset in (-1, 0, 1):
            index.setdefault(((d + timedelta(days=offset)).isoformat(), pair), m)
    return index


def fifa_player_passes(match: dict) -> dict:
    """One FIFA match -> {folded player name: {column: value}}.

    Joins the data-hub stats file (player id -> stat list) with the live
    endpoint's lineups (player id -> display name). Ids are shared between
    the two feeds; "-1" is FIFA's team-total bucket and is skipped.
    """
    ifes = (match.get("Properties") or {}).get("IdIFES")
    if not ifes:
        raise LookupError("match has no IdIFES (no data-hub stats)")
    stats = fifa_get(FDH_PLAYERS_URL.format(ifes=ifes))
    live = fifa_get(LIVE_URL.format(
        comp=match["IdCompetition"], season=match["IdSeason"],
        stage=match["IdStage"], match=match["IdMatch"]), params={"language": "en"})

    names = {}
    for side in ("HomeTeam", "AwayTeam"):
        for p in (live.get(side) or {}).get("Players") or []:
            name = ((p.get("PlayerName")) or [{}])[0].get("Description")
            if p.get("IdPlayer") and name:
                names[str(p["IdPlayer"])] = name

    out = {}
    for pid, stat_list in stats.items():
        if pid == "-1" or pid not in names:
            continue
        by_id = {s[0]: s[1] for s in stat_list if isinstance(s, (list, tuple)) and s}
        row = {col: by_id.get(fifa_id) for fifa_id, col in FIFA_STAT_MAP.items()
               if by_id.get(fifa_id) is not None}
        if row:
            out[fold_name(names[pid])] = row
    return out


def match_player(folded_fifa: dict, espn_name: str):
    """Find the FIFA stat row for one ESPN player name, or None.

    Tried in order, requiring a unique hit at each step:
      1. exact accent-folded match
      2. same tokens in any order (Korean names flip order between feeds)
      3. token subset (one feed carries an extra middle/second surname)
    """
    folded = fold_name(espn_name)
    if folded in folded_fifa:
        return folded_fifa[folded]
    tokens = set(folded.split())
    if not tokens:
        return None
    same = [k for k in folded_fifa if set(k.split()) == tokens]
    if len(same) == 1:
        return folded_fifa[same[0]]
    subset = [k for k in folded_fifa
              if set(k.split()) <= tokens or tokens <= set(k.split())]
    if len(subset) == 1:
        return folded_fifa[subset[0]]
    return None


# ---------------------------------------------------------------------------
# Supabase access
# ---------------------------------------------------------------------------
def matches_to_process(since: date, match_id=None):
    """Completed World Cup matches from soccer_schedule."""
    filters = [("eq", "status", "completed"),
               ("gte", "match_date", since.isoformat())]
    if match_id:
        filters = [("eq", "match_id", match_id)]
    rows = sc.fetch_all(
        sc.SCHEDULE_TABLE,
        "match_id,match_date,competition,season,home_team,away_team,status",
        filters=filters, order_col="match_date",
    )
    return [r for r in rows if sc.is_world_cup(r.get("competition"))]


def log_rows_for_match(match_id):
    res = (
        sc.supabase.table(LOGS_TABLE)
        .select("player_id,player_name,passes")
        .eq("match_id", match_id)
        .execute()
    )
    return res.data or []


def write_passes(match_id, updates):
    """Set pass counts on existing (player_id, match_id) log rows."""
    for player_id, row in updates:
        (sc.supabase.table(LOGS_TABLE)
         .update(row)
         .eq("player_id", player_id)
         .eq("match_id", match_id)
         .execute())
    return len(updates)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fill soccer_player_match_logs.passes from FIFA's API.")
    parser.add_argument("--days-back", type=int, default=4,
                        help="Process matches completed in the last N days (default 4).")
    parser.add_argument("--backfill", metavar="YYYY-MM-DD", default=None,
                        help="Process every completed WC match since this date.")
    parser.add_argument("--match-id", type=int, default=None,
                        help="Process a single soccer_schedule match id.")
    parser.add_argument("--check", action="store_true",
                        help="Parse + print only; write nothing.")
    parser.add_argument("--force", action="store_true",
                        help="Re-write matches whose passes are already filled.")
    args = parser.parse_args()

    if args.backfill:
        try:
            since = datetime.strptime(args.backfill, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit("--backfill must be YYYY-MM-DD")
    else:
        since = date.today() - timedelta(days=args.days_back)

    if "passes" not in sc.optional_log_columns():
        raise SystemExit("soccer_player_match_logs has no passes column -- "
                         "run the SQL in SOCCER_SETUP.md first.")

    matches = matches_to_process(since, match_id=args.match_id)
    print(f"{len(matches)} completed World Cup match(es) since {since}\n")
    if not matches:
        return

    fifa_index = fifa_calendar(since - timedelta(days=1),
                               date.today() + timedelta(days=1))
    print(f"FIFA calendar: {len(set(m['IdMatch'] for m in fifa_index.values()))} "
          f"completed match(es) in window\n")

    failures, total = [], 0
    for i, match in enumerate(matches, 1):
        label = (f"{match.get('match_date')} {match.get('home_team')} vs "
                 f"{match.get('away_team')}")
        try:
            logs = log_rows_for_match(match["match_id"])
            if not logs:
                print(f"  -- {label}: no log rows yet (run 21 first); skipping")
                continue
            if not args.force and all(r.get("passes") is not None for r in logs):
                continue   # already enriched

            pair = frozenset({sc.normalize_team(match["home_team"]).lower(),
                              sc.normalize_team(match["away_team"]).lower()})
            fifa_match = fifa_index.get((str(match["match_date"])[:10], pair))
            if fifa_match is None:
                failures.append((label, "no FIFA calendar match for date+teams"))
                print(f"  !! {label}: not found in FIFA calendar")
                continue

            folded_fifa = fifa_player_passes(fifa_match)
            updates, unmatched = [], []
            for r in logs:
                row = match_player(folded_fifa, r.get("player_name"))
                if row is not None:
                    updates.append((r["player_id"], row))
                else:
                    unmatched.append(r.get("player_name"))

            if args.check:
                print(f"  {label}: {len(updates)}/{len(logs)} players matched")
                for pid, row in updates[:8]:
                    name = next(r["player_name"] for r in logs if r["player_id"] == pid)
                    print(f"     {name}: {row}")
            else:
                written = write_passes(match["match_id"], updates)
                total += written
                print(f"  [{i}/{len(matches)}] {label}: {written}/{len(logs)} "
                      f"rows updated")
            if unmatched:
                print(f"     (no FIFA name match: {', '.join(unmatched[:6])}"
                      f"{' ...' if len(unmatched) > 6 else ''})")
        except Exception as exc:  # noqa: BLE001 - one bad match shouldn't kill the run
            failures.append((label, str(exc)))
            print(f"  !! {label}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 50)
    print(f"DONE. Log rows enriched with passes: {total}")
    if failures:
        print(f"\n{len(failures)} match(es) failed:")
        for where, err in failures[:20]:
            print(f"  - {where}: {err}")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()

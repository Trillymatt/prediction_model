"""
Shared plumbing for the NFL side of the tool.

The football sibling of soccer_common.py: the Supabase client, table names,
team-name normalization (ESPN display names vs. common short forms), and the
Eastern-time helpers the schedule/boards need. Imported only by the 3x NFL
scripts and api.py's NFL endpoints -- the NBA and soccer sides are untouched.

The NFL pipeline is being built in stages (see NFL_SETUP.md):
  1. Schedule ingestion (30_nfl_schedule.py)      <- done, this release
  2. Team + player stats ingestion                <- next
  3. Projection engines (game outcome, props)     <- after data flows
"""

import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from supabase import create_client, Client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCHEDULE_TABLE = "nfl_schedule"
PAGE_SIZE = 1000                          # PostgREST page cap, same as elsewhere

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(HERE, ".env")
load_dotenv(ENV_PATH)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit(
        "Missing credentials. Set SUPABASE_URL and SUPABASE_KEY in your .env file."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Team-name normalization
# ---------------------------------------------------------------------------
# Canonical names are ESPN's displayName ("Kansas City Chiefs"). The map below
# lets the API accept abbreviations and city/nickname short forms the way the
# NBA side accepts "LAL".
TEAMS = {
    "ARI": "Arizona Cardinals",    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",     "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",   "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",       "DEN": "Denver Broncos",
    "DET": "Detroit Lions",        "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",       "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars", "KC":  "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
    "LV":  "Las Vegas Raiders",    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",    "NE":  "New England Patriots",
    "NO":  "New Orleans Saints",   "NYG": "New York Giants",
    "NYJ": "New York Jets",        "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",  "SEA": "Seattle Seahawks",
    "SF":  "San Francisco 49ers",  "TB":  "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",     "WSH": "Washington Commanders",
}

_ALIASES = {name.lower(): name for name in TEAMS.values()}
_ALIASES.update({abbr.lower(): name for abbr, name in TEAMS.items()})
# Nicknames alone ("chiefs") and common alternate abbreviations.
_ALIASES.update({name.rsplit(" ", 1)[-1].lower(): name for name in TEAMS.values()})
_ALIASES.update({
    "niners": "San Francisco 49ers",
    "jags": "Jacksonville Jaguars",
    "jac": "Jacksonville Jaguars",
    "was": "Washington Commanders",
    "wft": "Washington Commanders",
    "oak": "Las Vegas Raiders",
    "sd": "Los Angeles Chargers",
})

NAME_TO_ABBR = {name: abbr for abbr, name in TEAMS.items()}


def normalize_team(name) -> str:
    """Map any spelling of an NFL team to its canonical (ESPN) name."""
    if not name:
        return ""
    cleaned = str(name).strip()
    return _ALIASES.get(cleaned.lower(), cleaned)


def team_abbr(name) -> str:
    """Canonical name/alias -> abbreviation ('' if unknown)."""
    return NAME_TO_ABBR.get(normalize_team(name), "")


def same_team(a, b) -> bool:
    return normalize_team(a).lower() == normalize_team(b).lower()


# ---------------------------------------------------------------------------
# Game timing (Eastern) -- same conventions as the soccer side: schedule rows
# store game_date / game_time in US/Eastern, so slate boundaries are evaluated
# in Eastern, not the server's clock.
# ---------------------------------------------------------------------------
try:
    from zoneinfo import ZoneInfo
    EASTERN = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001 - no tz database -> fall back to UTC
    EASTERN = None


def now_eastern():
    if EASTERN is not None:
        return datetime.now(EASTERN).replace(tzinfo=None)
    return datetime.utcnow()


def today_eastern():
    return now_eastern().date()


def parse_date(raw):
    """ISO 'YYYY-MM-DD...' -> date (None if unparseable)."""
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def game_datetime(row):
    """A row's kickoff as a naive Eastern datetime (start of day if untimed)."""
    d = parse_date(row.get("game_date"))
    if d is None:
        return None
    raw_time = (row.get("game_time") or "").strip()
    hh = mm = 0
    if raw_time:
        try:
            parts = raw_time.split(":")
            hh, mm = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            hh = mm = 0
    return datetime(d.year, d.month, d.day, hh, mm)


def game_has_started(row, grace_minutes=0, now=None):
    """True once a game's kickoff (Eastern) is in the past. Rows with no
    kickoff time only 'start' once their whole Eastern day is over, so a
    missing time never hides a game early."""
    kickoff = game_datetime(row)
    if kickoff is None:
        return False
    now = now or now_eastern()
    if not (row.get("game_time") or "").strip():
        return now.date() > kickoff.date()
    return now >= kickoff + timedelta(minutes=grace_minutes)


# ---------------------------------------------------------------------------
# Supabase access
# ---------------------------------------------------------------------------
def fetch_all(table: str, columns: str, filters=None, order_col=None):
    """Select every row of a table, paging past the PostgREST 1000-row cap.
    Same contract as soccer_common.fetch_all (including the `id` tiebreaker
    so page boundaries can't shuffle ties)."""
    rows = []
    start = 0
    while True:
        q = supabase.table(table).select(columns)
        for f in (filters or []):
            q = getattr(q, f[0])(*f[1:])
        if order_col:
            q = q.order(order_col, desc=False)
        q = q.order("id", desc=False)
        res = q.range(start, start + PAGE_SIZE - 1).execute()
        page = res.data or []
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return rows


def upcoming_games(days: int = 30) -> list:
    """Upcoming NFL games in the next `days` days, soonest first.

    Time-based like the soccer boards: a game that has already kicked off
    today is dropped while the rest of the slate stays up.
    """
    today = today_eastern()
    horizon = (today + timedelta(days=days)).isoformat()
    rows = fetch_all(
        SCHEDULE_TABLE,
        "game_id,game_date,game_time,season,season_type,week,home_team,away_team",
        filters=[("eq", "status", "upcoming"),
                 ("gte", "game_date", today.isoformat()),
                 ("lte", "game_date", horizon)],
        order_col="game_date",
    )
    now = now_eastern()
    rows = [g for g in rows if not game_has_started(g, now=now)]
    rows.sort(key=lambda g: (g.get("game_date") or "", g.get("game_time") or ""))
    return rows

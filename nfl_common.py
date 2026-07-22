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
import math
import time
from datetime import datetime, timedelta, date

from dotenv import load_dotenv
from supabase import create_client, Client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCHEDULE_TABLE = "nfl_schedule"
PLAYERS_TABLE = "nfl_players"
LOGS_TABLE = "nfl_player_game_logs"       # per-player box scores (stage 2b)
INJURIES_TABLE = "nfl_injuries"           # ESPN team injury reports (stage 2b)
PAGE_SIZE = 1000                          # PostgREST page cap, same as elsewhere

# Season ratings. NFL home-field advantage is worth roughly this many points;
# it's removed before computing opponent-adjusted strength so a team isn't
# credited for playing at home. SRS is solved by fixed-point iteration.
HFA_POINTS = 2.0
SRS_ITERS = 30
# Early in the year a team's own margins are noisy, so ratings are shrunk toward
# the prior-season rating (or league average) until this many games are played.
RATING_PRIOR_GAMES = 6
# Season types that count toward ratings/form (preseason is meaningless).
RATED_SEASON_TYPES = ("regular", "playoffs")

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


# ---------------------------------------------------------------------------
# Players (roster directory -- powers autocomplete + the per-game roster view;
# stats/props come with the next pipeline stage, see NFL_SETUP.md)
# ---------------------------------------------------------------------------
# Same accent-insensitive substring approach as the NBA/soccer sides. Each
# side keeps its own copy so the endpoints stay independent (see api.py).
_ACCENT_CLASSES = {
    "a": "aàáâãäåā", "c": "cçćč", "e": "eèéêëē", "g": "gğ", "i": "iìíîïıī",
    "n": "nñń", "o": "oòóôõöø", "s": "sšş", "u": "uùúûü", "y": "yýÿ", "z": "zžź",
}


def _accent_regex(query: str) -> str:
    import re
    out = []
    for ch in query:
        cls = _ACCENT_CLASSES.get(ch.lower())
        out.append(f"[{cls}]" if cls else re.escape(ch))
    return "".join(out)


def search_players(query: str, limit: int = 10) -> list:
    """Autocomplete against nfl_players. Returns [] below 2 chars or if the
    table isn't set up yet (caller decides how to surface that)."""
    query = (query or "").strip()
    if len(query) < 2:
        return []
    res = (
        supabase.table(PLAYERS_TABLE)
        .select("player_id,player_name,team,position")
        .filter("player_name", "imatch", _accent_regex(query))
        .order("player_name")
        .limit(limit)
        .execute()
    )
    return res.data or []


# Rough offense -> defense -> specialists ordering so a roster reads like a
# depth chart instead of an alphabetical dump.
_POSITION_RANK = {
    "QB": 0, "RB": 1, "FB": 2, "WR": 3, "TE": 4,
    "T": 5, "G": 5, "C": 5, "OL": 5, "OT": 5, "OG": 5,
    "DE": 6, "DT": 6, "NT": 6, "DL": 6,
    "LB": 7, "OLB": 7, "ILB": 7, "MLB": 7,
    "CB": 8, "S": 8, "SS": 8, "FS": 8, "DB": 8,
    "K": 9, "P": 9, "LS": 9,
}


def _position_rank(pos):
    return _POSITION_RANK.get((pos or "").upper(), 99)


def roster_for_game(home, away) -> dict:
    """Both teams' rosters for a matchup -- the NFL twin of
    multi_props.nba_roster() / soccer_roster(), minus the stat-based
    ranking (no player game logs yet). Players are ordered offense -> defense
    -> specialists so the list reads like a depth chart."""
    home_n, away_n = normalize_team(home), normalize_team(away)
    if not home_n or not away_n:
        raise ValueError(f"Unknown team in '{home}'/'{away}'.")

    res = (
        supabase.table(PLAYERS_TABLE)
        .select("player_id,player_name,team,position")
        .in_("team", [home_n, away_n])
        .execute()
    )
    by_team = {home_n: [], away_n: []}
    for r in res.data or []:
        if r.get("team") in by_team:
            by_team[r["team"]].append(r)

    teams = []
    for team, side, opp in ((home_n, "home", away_n), (away_n, "away", home_n)):
        players = sorted(
            by_team[team],
            key=lambda p: (_position_rank(p.get("position")), p.get("player_name") or ""),
        )
        teams.append({
            "abbr": team_abbr(team) or team,
            "side": side,
            "opponent": opp,
            "players": players,
        })
    return {"home_team": home_n, "away_team": away_n, "teams": teams}


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


# ---------------------------------------------------------------------------
# Season + schedule access (cached) -- the football twin of soccer_common's
# schedule cache. Every rating/form read shares one fetch so the API isn't
# re-paging the whole schedule table on each request.
# ---------------------------------------------------------------------------
def current_season() -> str:
    """The in-progress NFL season label (its starting year, e.g. '2026').

    The NFL season runs Sep -> Feb, labeled by the starting year, so January /
    February playoff games belong to the previous calendar year's season.
    """
    today = today_eastern()
    return str(today.year if today.month >= 3 else today.year - 1)


SCHEDULE_CACHE_TTL = 120.0
_schedule_cache = {"rows": None, "at": 0.0}


def fetch_schedule_rows(force=False):
    """Every nfl_schedule row, oldest-first, cached for SCHEDULE_CACHE_TTL."""
    now = time.monotonic()
    if (not force and _schedule_cache["rows"] is not None
            and now - _schedule_cache["at"] < SCHEDULE_CACHE_TTL):
        return _schedule_cache["rows"]
    rows = fetch_all(
        SCHEDULE_TABLE,
        "game_id,game_date,game_time,season,season_type,week,home_team,away_team,"
        "status,home_score,away_score",
        order_col="game_date",
    )
    _schedule_cache.update(rows=rows, at=now)
    return rows


def _num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(total, count):
    return total / count if count else None


def completed_team_games(schedule_rows, team, seasons=None):
    """A team's completed rated games (regular + playoffs), oldest-first.

    Each entry: {date, season, opponent, scored, allowed, home, won, tied,
    margin_neutral} where margin_neutral removes home-field advantage so the
    value reflects true strength regardless of venue.
    """
    team = normalize_team(team)
    out = []
    for g in schedule_rows:
        if g.get("status") != "completed":
            continue
        if (g.get("season_type") or "regular") not in RATED_SEASON_TYPES:
            continue
        if seasons is not None and str(g.get("season")) not in seasons:
            continue
        hs, as_ = _num(g.get("home_score")), _num(g.get("away_score"))
        if hs is None or as_ is None:
            continue
        home, away = normalize_team(g.get("home_team")), normalize_team(g.get("away_team"))
        if team == home:
            scored, allowed, opp, is_home = hs, as_, away, True
        elif team == away:
            scored, allowed, opp, is_home = as_, hs, home, False
        else:
            continue
        margin = scored - allowed
        # Neutralize home field so strength isn't inflated by playing at home.
        margin_neutral = margin - (HFA_POINTS if is_home else -HFA_POINTS)
        out.append({
            "date": parse_date(g.get("game_date")),
            "season": str(g.get("season")),
            "opponent": opp,
            "scored": scored,
            "allowed": allowed,
            "home": is_home,
            "won": scored > allowed,
            "tied": scored == allowed,
            "margin_neutral": margin_neutral,
        })
    return out


def _srs_ratings(schedule_rows, seasons):
    """Simple Rating System over the given seasons: each team's rating is its
    average neutral-site point margin plus its average opponent's rating,
    solved by fixed-point iteration. Positive = better than an average team."""
    games = {}
    for g in schedule_rows:
        if g.get("status") != "completed":
            continue
        if (g.get("season_type") or "regular") not in RATED_SEASON_TYPES:
            continue
        if str(g.get("season")) not in seasons:
            continue
        hs, as_ = _num(g.get("home_score")), _num(g.get("away_score"))
        if hs is None or as_ is None:
            continue
        home, away = normalize_team(g.get("home_team")), normalize_team(g.get("away_team"))
        if not home or not away:
            continue
        margin = (hs - as_) - HFA_POINTS   # from home team's view, HFA removed
        games.setdefault(home, []).append((away, margin))
        games.setdefault(away, []).append((home, -margin))
    if not games:
        return {}, {}

    avg_margin = {t: sum(m for _, m in gl) / len(gl) for t, gl in games.items()}
    ratings = dict(avg_margin)
    for _ in range(SRS_ITERS):
        new = {}
        for t, gl in games.items():
            sos = sum(ratings.get(opp, 0.0) for opp, _ in gl) / len(gl)
            new[t] = avg_margin[t] + sos
        # Re-center on zero so ratings stay interpretable as "vs average team".
        mean_r = sum(new.values()) / len(new)
        ratings = {t: r - mean_r for t, r in new.items()}
    return ratings, {t: len(gl) for t, gl in games.items()}


_ratings_cache = {"key": None, "value": None}


def team_ratings(schedule_rows=None, season=None):
    """Opponent-adjusted team strength (points vs an average team), computed
    from results and blended with last season's rating as a prior early in the
    year -- so it's meaningful in Week 1 and sharpens every week as games come
    in (no retrain needed). This is the strength-of-schedule backbone: because
    a rating already accounts for who you played, harder schedules stop
    flattering weak teams and stop punishing strong ones.
    """
    schedule_rows = schedule_rows if schedule_rows is not None else fetch_schedule_rows()
    season = season or current_season()
    key = (id(schedule_rows), len(schedule_rows), season)
    if _ratings_cache["key"] == key:
        return _ratings_cache["value"]

    cur, games_played = _srs_ratings(schedule_rows, {season})
    prior, _ = _srs_ratings(schedule_rows, {str(int(season) - 1)})

    out = {}
    for team in set(cur) | set(prior):
        n = games_played.get(team, 0)
        cur_r = cur.get(team, 0.0)
        prior_r = prior.get(team, 0.0)
        # Shrink toward the prior until the team has a real sample this year.
        w = min(n, RATING_PRIOR_GAMES) / RATING_PRIOR_GAMES
        out[team] = w * cur_r + (1 - w) * prior_r
    _ratings_cache.update(key=key, value=out)
    return out


def rating_gap(home, away, schedule_rows=None):
    """Home team's projected point edge before the game model: rating gap plus
    home-field advantage. Used as the game model's anchor and its Week-1 fallback."""
    ratings = team_ratings(schedule_rows)
    return (ratings.get(normalize_team(home), 0.0)
            - ratings.get(normalize_team(away), 0.0) + HFA_POINTS)


def team_form(schedule_rows, team, season=None):
    """Season-to-date + rolling scoring/allowed/win form for a team this season,
    mirroring the NBA game model's team feature block (schedule-derived)."""
    season = season or current_season()
    results = completed_team_games(schedule_rows, team, seasons={season})
    n = len(results)
    recent, last5 = results[-10:], results[-5:]
    ppg = _avg(sum(r["scored"] for r in results), n)
    papg = _avg(sum(r["allowed"] for r in results), n)
    return {
        "season_games": n,
        "season_ppg": ppg,
        "season_papg": papg,
        "season_net": (ppg - papg) if (ppg is not None and papg is not None) else None,
        "season_win_pct": _avg(sum(1.0 for r in results if r["won"]), n),
        "l5_ppg": _avg(sum(r["scored"] for r in last5), len(last5)),
        "l5_papg": _avg(sum(r["allowed"] for r in last5), len(last5)),
        "l3_ppg": _avg(sum(r["scored"] for r in results[-3:]), len(results[-3:])),
    }


def days_rest(schedule_rows, team, game_date, cap=14):
    """Days since a team's previous completed game (capped), or None."""
    gd = parse_date(game_date) if not isinstance(game_date, date) else game_date
    results = completed_team_games(schedule_rows, team)
    if not results or gd is None or results[-1]["date"] is None:
        return None
    return min((gd - results[-1]["date"]).days, cap)


def strength_of_schedule(team, schedule_rows=None, season=None):
    """A team's average opponent rating across its games this season -- both the
    ones it has played and the ones still to come. Positive = a harder slate.
    This is the "who has the harder season" read, straight off the ratings."""
    schedule_rows = schedule_rows if schedule_rows is not None else fetch_schedule_rows()
    season = season or current_season()
    ratings = team_ratings(schedule_rows, season)
    team_n = normalize_team(team)
    played, remaining = [], []
    for g in schedule_rows:
        if (g.get("season_type") or "regular") not in RATED_SEASON_TYPES:
            continue
        if str(g.get("season")) != season:
            continue
        home, away = normalize_team(g.get("home_team")), normalize_team(g.get("away_team"))
        if team_n == home:
            opp = away
        elif team_n == away:
            opp = home
        else:
            continue
        r = ratings.get(opp, 0.0)
        (played if g.get("status") == "completed" else remaining).append(r)
    allg = played + remaining
    return {
        "played_sos": round(sum(played) / len(played), 2) if played else None,
        "remaining_sos": round(sum(remaining) / len(remaining), 2) if remaining else None,
        "full_sos": round(sum(allg) / len(allg), 2) if allg else None,
        "games_remaining": len(remaining),
    }


# ---------------------------------------------------------------------------
# Injuries (most-recent status per player, like NBA's injury_status)
# ---------------------------------------------------------------------------
def injury_status(player_id):
    """The most recent injury row for a player, or None. Tolerates a missing
    table so a fresh setup never errors."""
    try:
        res = (
            supabase.table(INJURIES_TABLE)
            .select("status,reason,game_date")
            .eq("player_id", player_id)
            .order("game_date", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return rows[0] if rows else None
    except Exception:  # noqa: BLE001 - table missing / RLS => no status
        return None

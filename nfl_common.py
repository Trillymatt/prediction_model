"""
Shared plumbing for the NFL side of the tool.

The football sibling of soccer_common.py: the Supabase client, table names,
team-name normalization (ESPN display names vs. common short forms), and the
Eastern-time helpers the schedule/boards need. Imported only by the 3x NFL
scripts and api.py's NFL endpoints -- the NBA and soccer sides are untouched.

The NFL pipeline is being built in stages (see NFL_SETUP.md):
  1. Schedule ingestion (30_nfl_schedule.py)              <- done
  2. Roster directory (31_nfl_rosters.py)                 <- done
  3. Game-outcome model: Elo + a points model, this file  <- done, this release
  4. Player-prop projections                              <- next

This module now also carries the team-strength engine (Elo ratings + a
recency-weighted points model) that 32_nfl_game_projections.py turns into a
full game projection -- see project_matchup() below.
"""

import os
import math
import time
from datetime import date, datetime, timedelta

from dotenv import load_dotenv
from supabase import create_client, Client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCHEDULE_TABLE = "nfl_schedule"
PLAYERS_TABLE = "nfl_players"
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


# ---------------------------------------------------------------------------
# Full schedule access (cached) -- the Elo/points model needs every completed
# game, not just the upcoming slate upcoming_games() returns.
# ---------------------------------------------------------------------------
SCHEDULE_CACHE_TTL = 120.0    # seconds; mirrors soccer_common's twin
_schedule_cache = {"rows": None, "at": 0.0}


def fetch_schedule_rows(force=False):
    """All nfl_schedule rows, oldest first (cached)."""
    now_mono = time.monotonic()
    if (not force and _schedule_cache["rows"] is not None
            and now_mono - _schedule_cache["at"] < SCHEDULE_CACHE_TTL):
        return _schedule_cache["rows"]
    rows = fetch_all(
        SCHEDULE_TABLE,
        "game_id,game_date,game_time,season,season_type,week,home_team,away_team,"
        "status,home_score,away_score",
        order_col="game_date",
    )
    _schedule_cache.update(rows=rows, at=now_mono)
    return rows


# ---------------------------------------------------------------------------
# Team strength: Elo ratings + a recency-weighted points model
# ---------------------------------------------------------------------------
# Unlike the soccer side, there's no researched priors file -- every team
# starts neutral (1500) and Elo is built entirely from nfl_schedule results.
# Backfilling a season or two of history before kickoff (see NFL_SETUP.md)
# gives the model real signal from Week 1 instead of starting blind.
#
# The constants below are standard NFL-analytics priors (in the spirit of
# 538's original NFL Elo methodology), not yet backtested against real
# results -- see the soccer side's 25_soccer_backtest.py for the kind of
# harness this should eventually get once a season's worth of games exist
# to validate against.
ELO_DEFAULT = 1500
ELO_K = 20.0                     # rating points moved per game at even odds
ELO_HOME_BONUS = 48              # home-field edge, in Elo points
ELO_SEASON_REGRESSION = 1 / 3.0  # fraction reverted toward 1500 at each new season
ELO_POINTS_PER_ELO = 25.0        # Elo points per 1 point of point-spread

BASE_TOTAL_POINTS = 44.0         # combined points/game prior (recent NFL average)
TOTAL_MIN, TOTAL_MAX = 30.0, 65.0
FORM_WINDOW = 10                 # games of scoring history considered "recent"
FORM_DAMPENING = 0.4             # how hard recent scoring moves the total
HOME_FIELD_POINTS = 1.5          # home-field edge, in points, for the stats side
STATS_ELO_BLEND = 0.5            # weight on the stats-implied margin vs. Elo's

SIGMA_MARGIN_PRIOR = 13.5        # ~1 std of NFL game margins
SIGMA_TOTAL_PRIOR = 10.0         # ~1 std of NFL game totals
THIN_DATA_GAMES = 3              # below this many logged games, widen sigma

_elo_cache = {"key": None, "ratings": None}


def normal_cdf(z: float) -> float:
    """Standard-normal CDF Phi(z) via the error function (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _mov_multiplier(margin: int, elo_diff: float) -> float:
    """538's margin-of-victory multiplier: a blowout moves the rating less
    when the winner was already a big favorite (avoids double-counting what
    the Elo gap already explains)."""
    return math.log(abs(margin) + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))


def elo_ratings(schedule_rows) -> dict:
    """Current Elo per team, built entirely from completed nfl_schedule
    results. Ratings regress ELO_SEASON_REGRESSION toward 1500 at each
    season boundary so they carry over year to year without getting stuck
    on a team that overhauled its roster. Memoized per schedule snapshot,
    like the soccer side's twin."""
    key = (id(schedule_rows), len(schedule_rows))
    if _elo_cache["key"] == key:
        return _elo_cache["ratings"]

    completed = [g for g in schedule_rows
                 if g.get("status") == "completed"
                 and g.get("home_score") is not None
                 and g.get("away_score") is not None]
    completed.sort(key=lambda g: (g.get("game_date") or "", g.get("game_id") or 0))

    ratings = {}
    last_season = None
    for g in completed:
        season = g.get("season")
        if last_season is not None and season != last_season:
            for team in ratings:
                ratings[team] += (ELO_DEFAULT - ratings[team]) * ELO_SEASON_REGRESSION
        last_season = season

        home = normalize_team(g.get("home_team"))
        away = normalize_team(g.get("away_team"))
        if not home or not away:
            continue
        rh = ratings.get(home, ELO_DEFAULT)
        ra = ratings.get(away, ELO_DEFAULT)

        hs, as_ = g["home_score"], g["away_score"]
        elo_diff = (rh + ELO_HOME_BONUS) - ra
        expected_home = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        result_home = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        margin = hs - as_
        mult = _mov_multiplier(margin, elo_diff) if margin else 1.0
        delta = ELO_K * mult * (result_home - expected_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta

    _elo_cache.update(key=key, ratings=ratings)
    return ratings


def team_recent_games(schedule_rows, team, window=FORM_WINDOW):
    """This team's last `window` completed games (any season, oldest first).
    Each entry: {date, opponent, scored, allowed, won, season}. A rolling
    window (not a season-only one) means Week 1 of a new season still has
    signal from the tail of the last one."""
    team = normalize_team(team)
    out = []
    for g in schedule_rows:
        if g.get("status") != "completed":
            continue
        hs, as_ = g.get("home_score"), g.get("away_score")
        if hs is None or as_ is None:
            continue
        home = normalize_team(g.get("home_team"))
        away = normalize_team(g.get("away_team"))
        if team == home:
            scored, allowed, opp = float(hs), float(as_), away
        elif team == away:
            scored, allowed, opp = float(as_), float(hs), home
        else:
            continue
        out.append({
            "date": parse_date(g.get("game_date")),
            "opponent": opp,
            "scored": scored,
            "allowed": allowed,
            "won": scored > allowed,
            "season": g.get("season"),
        })
    out.sort(key=lambda r: r["date"] or date.min)
    return out[-window:]


def league_scoring_average(schedule_rows, window_days=400):
    """Average combined points per team per game across recent completed
    games. Falls back to the BASE_TOTAL_POINTS prior if there's no data yet
    (e.g. a brand-new setup before any results have been backfilled)."""
    cutoff = today_eastern() - timedelta(days=window_days)
    total, n = 0.0, 0
    for g in schedule_rows:
        if g.get("status") != "completed":
            continue
        hs, as_ = g.get("home_score"), g.get("away_score")
        if hs is None or as_ is None:
            continue
        d = parse_date(g.get("game_date"))
        if d is None or d < cutoff:
            continue
        total += hs + as_
        n += 1
    if n == 0:
        return BASE_TOTAL_POINTS / 2.0
    return (total / n) / 2.0   # per team per game


def _rest_points(days_rest):
    """Point value of extra/short rest -- a small, well-known scheduling
    edge (a bye week, a short week off Thursday football), not a headline
    driver of the projection."""
    if days_rest is None:
        return 0.0
    if days_rest >= 13:
        return 1.5     # coming off a bye
    if days_rest <= 5:
        return -1.0    # short week
    return 0.0


def project_matchup(home, away, schedule_rows=None, game_date=None) -> dict:
    """The shared math behind an NFL game projection: Elo (whole-body team
    strength) blended with a recency-weighted points model (current scoring
    form), plus a small rest adjustment. Returns every intermediate number
    so the factor cards can show exactly why.

    Degrades gracefully with no data: a team with zero games on file just
    gets a neutral (1500 Elo, league-average scoring) profile instead of an
    error, so the projection always returns something -- just a low-
    confidence, near-even one until results start flowing in.
    """
    if schedule_rows is None:
        schedule_rows = fetch_schedule_rows()
    home, away = normalize_team(home), normalize_team(away)

    ratings = elo_ratings(schedule_rows)
    elo_home = ratings.get(home, ELO_DEFAULT)
    elo_away = ratings.get(away, ELO_DEFAULT)
    elo_diff = (elo_home + ELO_HOME_BONUS) - elo_away
    elo_win_expectancy = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
    elo_margin = elo_diff / ELO_POINTS_PER_ELO

    home_games = team_recent_games(schedule_rows, home)
    away_games = team_recent_games(schedule_rows, away)

    def avg(games, key):
        vals = [g[key] for g in games]
        return sum(vals) / len(vals) if vals else None

    home_pf, home_pa = avg(home_games, "scored"), avg(home_games, "allowed")
    away_pf, away_pa = avg(away_games, "scored"), avg(away_games, "allowed")

    base = league_scoring_average(schedule_rows)

    def dampened(value):
        """value/base as a multiplier, pulled toward 1 by FORM_DAMPENING (no
        data => neutral, i.e. exactly the league baseline)."""
        if value is None or not base:
            return 1.0
        return 1.0 + FORM_DAMPENING * (value / base - 1.0)

    # Stats-implied score: each side's attack rate meets the other's
    # leakiness, scaled off the league baseline (the "four factors" style
    # approach -- same shape as the soccer engine's openness calc, in points
    # instead of goals).
    home_expected = base * dampened(home_pf) * dampened(away_pa)
    away_expected = base * dampened(away_pf) * dampened(home_pa)
    stats_margin = (home_expected - away_expected) + HOME_FIELD_POINTS
    stats_total = home_expected + away_expected

    # Blend Elo's margin (whole-season team strength) with the stats margin
    # (current scoring form) so neither signal dominates alone.
    margin = STATS_ELO_BLEND * stats_margin + (1.0 - STATS_ELO_BLEND) * elo_margin
    total = min(max(stats_total, TOTAL_MIN), TOTAL_MAX)

    # Rest: computed against the game being projected, not between past
    # games -- a bye week or short week only matters for the game right
    # after it.
    gdate = parse_date(game_date) if game_date else None
    home_rest = away_rest = None
    if gdate:
        if home_games and home_games[-1]["date"]:
            home_rest = (gdate - home_games[-1]["date"]).days
        if away_games and away_games[-1]["date"]:
            away_rest = (gdate - away_games[-1]["date"]).days
    margin += _rest_points(home_rest) - _rest_points(away_rest)

    # Thin data (a team with few/no logged games) means we're guessing --
    # widen the spread instead of pretending to be confident.
    thin = len(home_games) < THIN_DATA_GAMES or len(away_games) < THIN_DATA_GAMES
    sigma_margin = SIGMA_MARGIN_PRIOR * (1.3 if thin else 1.0)
    sigma_total = SIGMA_TOTAL_PRIOR * (1.3 if thin else 1.0)

    return {
        "home": home, "away": away,
        "elo_home": round(elo_home), "elo_away": round(elo_away),
        "elo_margin": elo_margin, "elo_win_expectancy": elo_win_expectancy,
        "stats_margin": stats_margin, "stats_total": stats_total,
        "projected_margin": margin, "projected_total": total,
        "sigma_margin": sigma_margin, "sigma_total": sigma_total,
        "home_pf": home_pf, "home_pa": home_pa,
        "away_pf": away_pf, "away_pa": away_pa,
        "home_games_n": len(home_games), "away_games_n": len(away_games),
        "home_rest": home_rest, "away_rest": away_rest,
        "league_avg_points": base, "thin_data": thin,
    }


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

"""
Shared plumbing for the soccer (World Cup) side of the tool.

Everything soccer-specific that more than one script needs lives here:
the Supabase client, table names, team-name normalization (ESPN, FIFA and
the priors file all spell countries differently), the team-priors knowledge
base, Elo ratings that update as results come in, and the Poisson math the
projection engines are built on.

The NBA side is untouched -- this module is imported only by the 2x soccer
scripts and api.py's soccer endpoints.
"""

import os
import math
import time
from datetime import datetime, date, timedelta

from dotenv import load_dotenv
from supabase import create_client, Client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCHEDULE_TABLE = "soccer_schedule"
LOGS_TABLE = "soccer_player_match_logs"
PLAYERS_TABLE = "soccer_players"          # optional -- see SOCCER_SETUP.md
PAGE_SIZE = 1000                          # PostgREST page cap, same as NBA side

HERE = os.path.dirname(os.path.abspath(__file__))
PRIORS_PATH = os.path.join(HERE, "soccer_team_priors.json")

# Elo settings (World Football Elo Ratings conventions). Priors carry a
# snapshot rating per team; completed matches AFTER the snapshot date update
# it, so ratings adapt automatically as the World Cup progresses.
ELO_DEFAULT = 1500            # unrated team (rare -- priors cover the WC field)
ELO_K_WORLD_CUP = 60          # World Cup finals weight
ELO_K_QUALIFIER = 40          # qualifiers / continental tournaments
ELO_K_FRIENDLY = 25           # friendlies
ELO_HOME_BONUS = 80           # host-nation edge (USA/Mexico/Canada at WC 2026)

# Goal model settings. International matches average ~2.6 goals. The two
# lambdas split the expected total by the Elo win expectancy through an
# odds-ratio curve: ratio = (W / (1-W))^RHO. RHO=0.65 reproduces market-like
# three-way prices across the range (a 65% Elo favourite ~52/25/23, a 90%
# favourite ~75/17/8) instead of collapsing underdogs to zero.
BASE_TOTAL_GOALS = 2.6
SPLIT_RHO = 0.65
WIN_EXPECTANCY_CLAMP = 0.97               # cap W so the split stays finite
TOTAL_MIN, TOTAL_MAX = 1.6, 4.4           # clamp on expected total goals
LAMBDA_FLOOR = 0.15                       # no team's xG expectation hits zero
FORM_DAMPENING = 0.45                     # how hard recent GF/GA moves the total
FORM_WINDOW = 15                          # completed matches per team for form

WC_HOSTS = {"United States", "Mexico", "Canada"}
WORLD_CUP_NAMES = ("fifa world cup", "world cup")

# ESPN league slug -> human-readable competition name stored in the tables.
# Shared by 20 (scoreboard) and 21 (match summaries). Unknown slugs are
# skipped with a warning, so this list can be extended freely.
LEAGUES = {
    "fifa.world": "FIFA World Cup",
    "fifa.worldq.uefa": "World Cup Qualifying - UEFA",
    "fifa.worldq.conmebol": "World Cup Qualifying - CONMEBOL",
    "fifa.worldq.concacaf": "World Cup Qualifying - CONCACAF",
    "fifa.worldq.caf": "World Cup Qualifying - CAF",
    "fifa.worldq.afc": "World Cup Qualifying - AFC",
    "fifa.worldq.ofc": "World Cup Qualifying - OFC",
    "fifa.friendly": "International Friendly",
    "uefa.nations": "UEFA Nations League",
    "uefa.euro": "UEFA European Championship",
    "conmebol.america": "Copa América",
    "concacaf.gold": "CONCACAF Gold Cup",
    "caf.nations": "Africa Cup of Nations",
    "afc.asiancup": "AFC Asian Cup",
}
COMPETITION_TO_SLUG = {v: k for k, v in LEAGUES.items()}

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
# Canonical names follow ESPN's spelling (what the pull scripts ingest), so
# the schedule, the logs, and the priors file all join cleanly. Every alias
# is lowercase.
TEAM_ALIASES = {
    "usa": "United States", "united states": "United States",
    "usmnt": "United States", "united states of america": "United States",
    "south korea": "South Korea", "korea republic": "South Korea",
    "korea": "South Korea",
    "czechia": "Czechia", "czech republic": "Czechia",
    "turkiye": "Türkiye", "türkiye": "Türkiye", "turkey": "Türkiye",
    "ivory coast": "Ivory Coast", "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "cape verde": "Cape Verde", "cabo verde": "Cape Verde",
    "cape verde islands": "Cape Verde",
    "dr congo": "DR Congo", "congo dr": "DR Congo",
    "democratic republic of the congo": "DR Congo", "congo": "DR Congo",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia": "Bosnia and Herzegovina",
    "iran": "Iran", "ir iran": "Iran", "islamic republic of iran": "Iran",
    "curacao": "Curaçao", "curaçao": "Curaçao",
    "netherlands": "Netherlands", "holland": "Netherlands",
    "saudi arabia": "Saudi Arabia", "ksa": "Saudi Arabia",
    "new zealand": "New Zealand",
    "scotland": "Scotland", "england": "England", "wales": "Wales",
    "northern ireland": "Northern Ireland", "republic of ireland": "Ireland",
    "ireland": "Ireland",
    "uae": "United Arab Emirates", "united arab emirates": "United Arab Emirates",
}


def normalize_team(name) -> str:
    """Map any spelling of a national team to its canonical (ESPN) name."""
    if not name:
        return ""
    cleaned = str(name).strip()
    return TEAM_ALIASES.get(cleaned.lower(), cleaned)


def same_team(a, b) -> bool:
    return normalize_team(a).lower() == normalize_team(b).lower()


# ---------------------------------------------------------------------------
# Small helpers (mirrors of the NBA-side utilities)
# ---------------------------------------------------------------------------
def parse_date(raw):
    """ISO 'YYYY-MM-DD...' -> date (None if unparseable)."""
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def mean(values):
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def normal_cdf(z: float) -> float:
    """Standard-normal CDF Phi(z) via the error function (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * lam ** k / math.factorial(k)


def poisson_line_probs(line: float, lam: float):
    """(p_over, p_push, p_under) for X ~ Poisson(lam) against a betting line.

    A 0.5 line means 'at least 1', a 1.5 line 'at least 2', etc. On a
    whole-number line, landing exactly on it is a PUSH (books refund it), so
    that mass belongs to neither side -- counting it for the under was a bug.
    """
    lam = max(lam, 0.0)
    if line == math.floor(line):
        k = int(line)
        p_push = poisson_pmf(k, lam)
        p_under = sum(poisson_pmf(i, lam) for i in range(k))
    else:
        p_push = 0.0
        p_under = sum(poisson_pmf(i, lam) for i in range(math.ceil(line)))
    p_over = max(0.0, 1.0 - p_under - p_push)
    return p_over, p_push, min(p_under, 1.0)


def poisson_p_over(line: float, lam: float) -> float:
    """P(X strictly beats the line); pushes excluded. See poisson_line_probs."""
    return poisson_line_probs(line, lam)[0]


def fetch_all(table: str, columns: str, filters=None, order_col=None):
    """Select every row of a table, paging past the PostgREST 1000-row cap.

    `filters` is a list of (method, args...) tuples applied to the query,
    e.g. [("eq", "status", "completed")]. Always adds the primary key `id`
    as an ordering tiebreaker: order_col values like match_date aren't
    unique, and without a total order Postgres may shuffle ties between
    page requests, silently dropping/duplicating rows at page boundaries.
    """
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
# Team priors (the researched knowledge base)
# ---------------------------------------------------------------------------
_priors_cache = None


def load_priors() -> dict:
    """Load soccer_team_priors.json once.

    Returns {"meta": {...}, "teams": {canonical_name: profile_dict}}. If the
    file is missing the engines still run -- every team just starts from the
    default Elo with no scouting notes.
    """
    global _priors_cache
    if _priors_cache is None:
        try:
            import json
            with open(PRIORS_PATH) as f:
                raw = json.load(f)
            teams = {normalize_team(t["team"]): t for t in raw.get("teams", [])}
            _priors_cache = {"meta": raw.get("meta", {}), "teams": teams}
        except (OSError, ValueError):
            _priors_cache = {"meta": {}, "teams": {}}
    return _priors_cache


def team_profile(team: str) -> dict:
    return load_priors()["teams"].get(normalize_team(team), {})


# ---------------------------------------------------------------------------
# Optional log columns (shared by the puller and the props engine)
# ---------------------------------------------------------------------------
OPTIONAL_LOG_COLUMNS = ("passes", "tackles", "saves",
                        "fouls_committed", "fouls_suffered")
_optional_cols_cache = None


def optional_log_columns() -> set:
    """Which optional soccer_player_match_logs columns exist right now.

    Probed once per process: one combined select when all columns exist (the
    common case after running SOCCER_SETUP.md), falling back to per-column
    probes so a partial migration still activates what's there.
    """
    global _optional_cols_cache
    if _optional_cols_cache is not None:
        return _optional_cols_cache
    try:
        sc_cols = ",".join(OPTIONAL_LOG_COLUMNS)
        supabase.table(LOGS_TABLE).select(sc_cols).limit(1).execute()
        _optional_cols_cache = set(OPTIONAL_LOG_COLUMNS)
        return _optional_cols_cache
    except Exception:  # noqa: BLE001 - at least one column missing
        pass
    present = set()
    for col in OPTIONAL_LOG_COLUMNS:
        try:
            supabase.table(LOGS_TABLE).select(col).limit(1).execute()
            present.add(col)
        except Exception:  # noqa: BLE001 - column absent
            continue
    _optional_cols_cache = present
    return present


# ---------------------------------------------------------------------------
# Schedule access + live Elo
# ---------------------------------------------------------------------------
SCHEDULE_CACHE_TTL = 120.0    # seconds; data only changes on pipeline runs
_schedule_cache = {"rows": None, "at": 0.0}


def fetch_schedule_rows(force=False):
    """All soccer_schedule rows, oldest first.

    Cached for SCHEDULE_CACHE_TTL: every soccer endpoint needs these rows,
    and the table only changes when the refresh pipeline runs -- without the
    cache each API request would re-page the whole table out of Supabase.
    """
    now = time.monotonic()
    if (not force and _schedule_cache["rows"] is not None
            and now - _schedule_cache["at"] < SCHEDULE_CACHE_TTL):
        return _schedule_cache["rows"]
    rows = fetch_all(
        SCHEDULE_TABLE,
        "match_id,match_date,match_time,competition,season,home_team,away_team,"
        "status,home_score,away_score",
        order_col="match_date",
    )
    _schedule_cache.update(rows=rows, at=now)
    return rows


def is_world_cup(competition) -> bool:
    comp = (competition or "").lower()
    return any(w in comp for w in WORLD_CUP_NAMES) and "qualif" not in comp


def _k_factor(competition) -> float:
    comp = (competition or "").lower()
    if is_world_cup(comp):
        return ELO_K_WORLD_CUP
    if "friendly" in comp:
        return ELO_K_FRIENDLY
    return ELO_K_QUALIFIER


def home_elo_bonus(home_team, competition) -> int:
    """The home side's Elo edge for one match.

    World Cup 2026 is mostly neutral-venue: only the three hosts get the
    bonus. Outside the WC the nominal home side is assumed genuinely at home.
    """
    if is_world_cup(competition):
        return ELO_HOME_BONUS if normalize_team(home_team) in WC_HOSTS else 0
    return ELO_HOME_BONUS


_elo_cache = {"key": None, "ratings": None}


def elo_ratings(schedule_rows) -> dict:
    """Current Elo per team: researched snapshot + updates from results.

    Teams start at their priors snapshot rating (or ELO_DEFAULT). Every
    completed match dated after the snapshot then moves both teams the
    standard Elo way, with the K-factor scaled by competition importance and
    margin of victory. This is what makes the model adapt as the World Cup
    goes on -- a team over-performing its scouting report gains rating with
    every result the nightly refresh ingests.

    Memoized per schedule snapshot (the cached rows object), so repeated
    calls within one request/TTL window don't replay the whole history.
    """
    # len() in the key catches in-place appends to a cached/shared list.
    key = (id(schedule_rows), len(schedule_rows))
    if _elo_cache["key"] == key:
        return _elo_cache["ratings"]
    priors = load_priors()
    snapshot = parse_date(priors["meta"].get("elo_snapshot_date")) or date(2026, 6, 1)
    ratings = {
        name: (p.get("elo") or ELO_DEFAULT) for name, p in priors["teams"].items()
    }

    for g in schedule_rows:
        if g.get("status") != "completed":
            continue
        # Strictly after the snapshot: matches ON the snapshot date are
        # already baked into the researched ratings (replaying them would
        # double-count).
        d = parse_date(g.get("match_date"))
        if d is None or d <= snapshot:
            continue
        hs, as_ = g.get("home_score"), g.get("away_score")
        if hs is None or as_ is None:
            continue
        home = normalize_team(g.get("home_team"))
        away = normalize_team(g.get("away_team"))
        if not home or not away:
            continue
        rh = ratings.get(home, ELO_DEFAULT)
        ra = ratings.get(away, ELO_DEFAULT)
        bonus = home_elo_bonus(home, g.get("competition"))

        expected_home = 1.0 / (1.0 + 10 ** (-((rh + bonus) - ra) / 400.0))
        result_home = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        margin = abs(hs - as_)
        g_mult = 1.0 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8.0)
        k = _k_factor(g.get("competition")) * g_mult
        delta = k * (result_home - expected_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    _elo_cache.update(key=key, ratings=ratings)
    return ratings


def team_recent_form(schedule_rows, team, window=FORM_WINDOW):
    """Last `window` completed matches for a team, oldest first.

    Returns a list of {date, opponent, scored, allowed, won, competition}.
    """
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
            "date": parse_date(g.get("match_date")),
            "opponent": opp,
            "scored": scored,
            "allowed": allowed,
            "won": scored > allowed,
            "drawn": scored == allowed,
            "competition": g.get("competition"),
        })
    return out[-window:]


def expected_goals(home, away, schedule_rows=None, competition="FIFA World Cup"):
    """Expected goals for each side of a match -- the heart of both engines.

    Three ingredients, all explainable:
      1. Elo gap (researched snapshot + live results) -> win expectancy W,
         which SPLITS the goals between the sides via an odds-ratio curve.
         Hosts get a home-edge bonus at the World Cup (true home sides get
         it elsewhere).
      2. Recent form (last FORM_WINDOW internationals) -> goals scored /
         conceded rates vs the international average, dampened, which sets
         how OPEN the match is (the expected total).
      3. The international scoring baseline (league_scoring_average).

    Returns a dict with lambda_home/lambda_away plus every intermediate
    number, so the factor cards can show exactly why.
    """
    if schedule_rows is None:
        schedule_rows = fetch_schedule_rows()
    home, away = normalize_team(home), normalize_team(away)

    ratings = elo_ratings(schedule_rows)
    elo_home = ratings.get(home, ELO_DEFAULT)
    elo_away = ratings.get(away, ELO_DEFAULT)

    world_cup = is_world_cup(competition)
    home_bonus = home_elo_bonus(home, competition)

    diff = (elo_home + home_bonus) - elo_away
    win_expectancy = 1.0 / (1.0 + 10 ** (-diff / 400.0))

    base = league_scoring_average(schedule_rows)  # goals per team per match
    home_form = team_recent_form(schedule_rows, home)
    away_form = team_recent_form(schedule_rows, away)

    def rate(form, key):
        vals = [m[key] for m in form]
        return (sum(vals) / len(vals)) if vals else None

    h_gf, h_ga = rate(home_form, "scored"), rate(home_form, "allowed")
    a_gf, a_ga = rate(away_form, "scored"), rate(away_form, "allowed")

    def dampened(value):
        """value/base as a multiplier, pulled toward 1 by FORM_DAMPENING."""
        if value is None or not base:
            return 1.0
        return 1.0 + FORM_DAMPENING * (value / base - 1.0)

    # How open is this match? Each side's attack rate meets the other's
    # leakiness; their average scales the baseline total.
    openness_home = dampened(h_gf) * dampened(a_ga)
    openness_away = dampened(a_gf) * dampened(h_ga)
    total = 2.0 * base * (openness_home + openness_away) / 2.0
    total = min(max(total, TOTAL_MIN), TOTAL_MAX)

    # Split the total by Elo: odds-ratio curve keeps underdogs alive while
    # still pricing big favourites like the market does.
    w = min(max(win_expectancy, 1.0 - WIN_EXPECTANCY_CLAMP), WIN_EXPECTANCY_CLAMP)
    ratio = (w / (1.0 - w)) ** SPLIT_RHO
    lambda_home = max(LAMBDA_FLOOR, total * ratio / (1.0 + ratio))
    lambda_away = max(LAMBDA_FLOOR, total / (1.0 + ratio))

    return {
        "home": home, "away": away,
        "lambda_home": lambda_home, "lambda_away": lambda_away,
        "elo_home": round(elo_home), "elo_away": round(elo_away),
        "elo_home_bonus": home_bonus,
        "elo_win_expectancy": win_expectancy,
        "expected_margin": lambda_home - lambda_away,
        "league_avg_goals": base,
        "home_gf": h_gf, "home_ga": h_ga,
        "away_gf": a_gf, "away_ga": a_ga,
        "home_form_n": len(home_form), "away_form_n": len(away_form),
        "home_form": home_form, "away_form": away_form,
        "world_cup": world_cup,
    }


def league_scoring_average(schedule_rows, window_days=730):
    """Average goals per team per match across recent completed matches."""
    cutoff = date.today() - timedelta(days=window_days)
    total, n = 0.0, 0
    for g in schedule_rows:
        if g.get("status") != "completed":
            continue
        hs, as_ = g.get("home_score"), g.get("away_score")
        if hs is None or as_ is None:
            continue
        d = parse_date(g.get("match_date"))
        if cutoff and (d is None or d < cutoff):
            continue
        total += hs + as_
        n += 1
    return (total / (2 * n)) if n else BASE_TOTAL_GOALS / 2.0

"""
FastAPI backend for the Money From a Baby projection tool.

    uvicorn api:app --reload --port 8000

Thin HTTP layer over the projection engine in 09_projections.py. The React
frontend hits these endpoints; everything that matters happens in
project_player(). (09_projections.py can't be imported normally because its name
starts with a digit, so we load it via importlib and reuse its functions and its
already-configured Supabase client.)

Endpoints
---------
  GET /api/health                       liveness check
  GET /api/stats                        list of supported stats
  GET /api/players?q=<text>             player autocomplete
  GET /api/project?player=&stat=&...    projection + confidence for a line
  GET /api/games                        upcoming games (next 10 days)
  GET /api/game?home=&away=&...         game outcome: win prob + projected score
  GET /api/picks?sport=nba|soccer       today's "My Picks" board (cached daily)

Soccer (World Cup) -- same shapes, three-way outcomes:
  GET /api/soccer/stats                 supported soccer stats
  GET /api/soccer/players?q=<text>      player autocomplete
  GET /api/soccer/project?player=&...   player prop projection + line grade
  GET /api/soccer/games?days=           upcoming matches (WC first)
  GET /api/soccer/game?home=&away=      match outcome: win/draw/win + goals

NFL (schedule + roster directory for now -- projections are being built, see
NFL_SETUP.md):
  GET /api/nfl/games?days=              upcoming games
  GET /api/nfl/players?q=<text>         player autocomplete
  GET /api/nfl/roster?home=&away=       both rosters for a matchup

Fantasy advisor (Sleeper / ESPN leagues -- see FANTASY_SETUP.md):
  GET  /api/fantasy/sleeper/leagues?username=   a Sleeper user's leagues
  POST /api/fantasy/league              connect a league, list its teams
  POST /api/fantasy/analyze             start/sit, needs, trades, buy/sell, waivers
  POST /api/fantasy/trade               grade a specific trade
  GET  /api/fantasy/odds?week=          spreads / totals / implied team totals
  GET  /api/fantasy/players?q=          player search (parlay builder)
  GET  /api/fantasy/player?name=&team=  one player's projection this week (NFL tab)
  POST /api/fantasy/parlay              grade prop + game-line parlay legs
  GET  /api/fantasy/props?event_id=     graded prop board (needs ODDS_API_KEY)

Setup:
    pip install -r requirements.txt
    # needs the same .env (SUPABASE_URL / SUPABASE_KEY) as the scripts
"""

import os
import re
import functools
import importlib.util

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Player names are stored with their real accents ("Nikola Jokić", "Luka
# Dončić"); Postgres ilike treats é/e, ć/c, etc. as distinct, so a plain-ASCII
# autocomplete query would miss them. We map each base letter to a regex class
# of its accented variants and match case-insensitively (PostgREST ~* via
# .filter("imatch")) instead of ilike. The soccer engine carries its own copy
# so the two endpoints stay independent.
_ACCENT_CLASSES = {
    "a": "aàáâãäåā", "c": "cçćč", "e": "eèéêëē", "g": "gğ", "i": "iìíîïıī",
    "n": "nñń", "o": "oòóôõöø", "s": "sšş", "u": "uùúûü", "y": "yýÿ", "z": "zžź",
}


def _accent_regex(query: str) -> str:
    """Build an accent-insensitive substring regex from a (partial) name."""
    out = []
    for ch in query:
        cls = _ACCENT_CLASSES.get(ch.lower())
        out.append(f"[{cls}]" if cls else re.escape(ch))
    return "".join(out)


def _load_numbered(module_name, filename):
    """Import a pipeline script whose filename starts with a digit."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(here, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Player-prop engine + game-outcome engine (names start with digits).
HERE = os.path.dirname(os.path.abspath(__file__))
engine = _load_numbered("projection_engine", "09_projections.py")
game_engine = _load_numbered("game_projection_engine", "14_game_projections.py")

# Soccer engines. Loaded defensively: if anything soccer-side is broken or
# not set up yet, the NBA app keeps working and the soccer endpoints explain.
try:
    soccer_engine = _load_numbered(
        "soccer_projection_engine", "22_soccer_projections.py")
    soccer_game_engine = _load_numbered(
        "soccer_game_projection_engine", "23_soccer_game_projections.py")
    _soccer_load_error = None
except Exception as exc:  # noqa: BLE001 - soccer must never break NBA
    soccer_engine = soccer_game_engine = None
    _soccer_load_error = str(exc)

# NFL (schedule stage). Loaded defensively like soccer: a missing table or
# bad credentials must never break the NBA/soccer apps.
try:
    import nfl_common
    _nfl_load_error = None
except (Exception, SystemExit) as exc:  # noqa: BLE001 - NFL must never break the others
    nfl_common = None
    _nfl_load_error = str(exc)

# Daily "My Picks" boards (computed in the background, cached per day).
import daily_picks

# Bet-slip analyzer (screenshot -> per-leg hit probabilities). Imports plainly
# because its filename has no leading digit; it talks to the engines we pass in.
import slip_analysis
import llm_analysis
import multi_props

# Fantasy advisor: pure HTTP clients + math, no Supabase.
import time
import fantasy_engine
import fantasy_sources

daily_picks.init(
    nba=engine, nba_game=game_engine,
    soccer=soccer_engine, soccer_game=soccer_game_engine,
)


def _require_soccer():
    if soccer_engine is None or soccer_game_engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Soccer engine unavailable: {_soccer_load_error}",
        )


def _soccer_data_error(exc: Exception) -> HTTPException:
    """A Supabase/network failure at request time (e.g. soccer tables not
    created yet, RLS denying reads) -> a 503 with a hint instead of a raw 500."""
    return HTTPException(
        status_code=503,
        detail=f"Soccer data unavailable ({exc}). If this is a fresh setup, "
               f"run the SQL + backfill in SOCCER_SETUP.md.",
    )


app = FastAPI(title="Money From a Baby API", version="1.0")

# Dev-friendly CORS so the Vite dev server can call us directly if it isn't
# proxying. (The frontend also proxies /api -> here, which needs no CORS.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_daily_picks():
    """Start building today's pick boards so the first visitor doesn't wait."""
    daily_picks.warm()


@app.get("/api/picks")
def picks(sport: str = Query("nba", description="nba | soccer")):
    """Today's boards: the "My Picks" list (the model's most confident calls
    for the slate) plus the per-game board ("games": every slate game with its
    outcome projection and strongest player picks). Returns status=building
    while the daily build is running; the frontend polls until it's ready."""
    sport = sport.lower()
    if sport not in ("nba", "soccer"):
        raise HTTPException(status_code=400, detail="sport must be nba or soccer")
    if sport == "soccer":
        _require_soccer()
    return daily_picks.get_picks(sport)


@app.get("/api/health")
def health():
    """Liveness probe + whether the trained model is loaded."""
    return {"status": "ok", "model_loaded": engine.load_models() is not None}


@app.get("/api/stats")
def stats():
    """The stats the tool can project (sorted for a stable dropdown)."""
    return {"stats": sorted(engine.STAT_DEFS.keys())}


@app.get("/api/players")
def players(q: str = Query("", description="name fragment"),
            limit: int = Query(10, ge=1, le=25)):
    """Player autocomplete: case-insensitive name match against nba_players."""
    q = q.strip()
    if len(q) < 2:
        return {"players": []}
    res = (
        engine.supabase.table(engine.PLAYERS_TABLE)
        .select("player_id,player_name,team,position")
        .filter("player_name", "imatch", _accent_regex(q))
        .order("player_name")
        .limit(limit)
        .execute()
    )
    return {"players": res.data or []}


@app.get("/api/project")
def project(
    player: str = Query(..., description="exact player name from autocomplete"),
    stat: str = Query(..., description="one of /api/stats"),
    line: float | None = Query(None, description="the over/under line from your book"),
    opponent: str | None = Query(None, description="opponent abbrev; omit to auto-detect"),
    location: str = Query("auto", description="auto | home | away"),
    game_type: str = Query("auto", description="auto | regular | playoffs"),
):
    """Project a stat and (if a line is given) grade it. Wraps project_player()."""
    home_away = {"home": "HOME", "away": "AWAY"}.get(location.lower())
    season_type = {
        "regular": "Regular Season",
        "playoffs": "Playoffs",
    }.get(game_type.lower(), "auto")
    try:
        return engine.project_player(
            player_name=player,
            stat=stat,
            line=line,
            opponent=opponent or None,
            home_away=home_away,
            season_type=season_type,
        )
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/games")
def games(days: int = Query(10, ge=1, le=60)):
    """Upcoming games in the next `days` days (soonest first), for the picker."""
    return {"games": game_engine.upcoming_games(days=days)}


@app.get("/api/game")
def game(
    home: str = Query(..., description="home team abbrev, e.g. NYK"),
    away: str = Query(..., description="away team abbrev, e.g. SAS"),
    date: str | None = Query(None, description="game date YYYY-MM-DD; omit to auto-detect"),
    game_id: str | None = Query(None, description="schedule game_id, if known"),
    game_type: str = Query("auto", description="auto | regular | playoffs"),
):
    """Game outcome: win probability + projected score. Wraps project_game()."""
    season_type = {
        "regular": "Regular Season",
        "playoffs": "Playoffs",
    }.get(game_type.lower(), "auto")
    try:
        return game_engine.project_game(
            home=home, away=away, game_date=date, game_id=game_id,
            season_type=season_type,
        )
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# --- Roster + multi-prop (NBA) ----------------------------------------------
@app.get("/api/roster")
def roster(
    home: str = Query(..., description="home team abbrev, e.g. NYK"),
    away: str = Query(..., description="away team abbrev, e.g. BOS"),
):
    """Every player on both teams of a game, rotation players first. Powers the
    'whole game in one place' view -- pick a game, see both rosters, tap any
    player for the model's read instead of searching them one at a time."""
    try:
        return multi_props.nba_roster(engine, home, away)
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/player/projections")
def player_projections(
    player: str = Query(..., description="exact player name from autocomplete"),
    stats: str | None = Query(None, description="comma-separated stats; omit for defaults"),
    opponent: str | None = Query(None, description="opponent abbrev; omit to auto-detect"),
    location: str = Query("auto", description="auto | home | away"),
    game_type: str = Query("auto", description="auto | regular | playoffs"),
):
    """One player's projection across several stats at once ("what he's
    projected for"), so you see every number before picking which line to bet."""
    stat_list = [s.strip() for s in stats.split(",")] if stats else None
    try:
        return multi_props.player_projections(
            engine, player, stats=stat_list, opponent=opponent or None,
            location=None if location == "auto" else location, game_type=game_type,
        )
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/project-batch")
def project_batch(payload: dict = Body(..., description='{"props": [{player, stat, line, side?, opponent?, location?, game_type?}]}')):
    """Grade a hand-built list of props in one call and score them as a parlay.
    The typed counterpart to the slip scanner -- same per-leg + combined math."""
    props = payload.get("props") or []
    if not isinstance(props, list) or not props:
        raise HTTPException(status_code=400, detail="Send a non-empty 'props' list.")
    if len(props) > 25:
        raise HTTPException(status_code=400, detail="Too many props (max 25).")
    return multi_props.grade_batch(engine, props)


# --- Soccer (World Cup) ------------------------------------------------------
@app.get("/api/soccer/stats")
def soccer_stats():
    """The soccer stats the tool can project (sorted for a stable dropdown)."""
    _require_soccer()
    return {"stats": sorted(soccer_engine.STAT_DEFS.keys())}


@app.get("/api/soccer/players")
def soccer_players(q: str = Query("", description="name fragment"),
                   limit: int = Query(10, ge=1, le=25)):
    """Player autocomplete from soccer_players (or the logs as fallback)."""
    _require_soccer()
    try:
        return {"players": soccer_engine.search_players(q, limit=limit)}
    except Exception as exc:  # noqa: BLE001 - tables missing / RLS / network
        raise _soccer_data_error(exc)


@app.get("/api/soccer/project")
def soccer_project(
    player: str = Query(..., description="exact player name from autocomplete"),
    stat: str = Query(..., description="one of /api/soccer/stats"),
    line: float | None = Query(None, description="the over/under line from your book"),
    opponent: str | None = Query(None, description="opponent country; omit to auto-detect"),
):
    """Project a soccer stat and (if a line is given) grade it."""
    _require_soccer()
    try:
        return soccer_engine.project_soccer_player(
            player_name=player, stat=stat, line=line,
            opponent=opponent or None,
        )
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - tables missing / RLS / network
        raise _soccer_data_error(exc)


@app.get("/api/soccer/roster")
def soccer_roster(
    home: str = Query(..., description="home team, e.g. Mexico"),
    away: str = Query(..., description="away team, e.g. South Africa"),
):
    """Both squads for a match, most-used players first -- the soccer twin of
    /api/roster so a match shows every player in one place."""
    _require_soccer()
    try:
        return multi_props.soccer_roster(soccer_engine, home, away)
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - tables missing / RLS / network
        raise _soccer_data_error(exc)


@app.get("/api/soccer/player/projections")
def soccer_player_projections(
    player: str = Query(..., description="exact player name from autocomplete"),
    stats: str | None = Query(None, description="comma-separated stats; omit for defaults"),
    opponent: str | None = Query(None, description="opponent country; omit to auto-detect"),
):
    """One player's projection across several soccer stats at once."""
    _require_soccer()
    stat_list = [s.strip() for s in stats.split(",")] if stats else None
    try:
        return multi_props.player_projections(
            soccer_engine, player, stats=stat_list, opponent=opponent or None,
            sport="soccer",
        )
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - tables missing / RLS / network
        raise _soccer_data_error(exc)


@app.post("/api/soccer/project-batch")
def soccer_project_batch(payload: dict = Body(..., description='{"props": [{player, stat, line, side?, opponent?}]}')):
    """Grade a hand-built list of soccer props as a parlay (soccer twin of
    /api/project-batch)."""
    _require_soccer()
    props = payload.get("props") or []
    if not isinstance(props, list) or not props:
        raise HTTPException(status_code=400, detail="Send a non-empty 'props' list.")
    if len(props) > 25:
        raise HTTPException(status_code=400, detail="Too many props (max 25).")
    try:
        return multi_props.grade_batch(soccer_engine, props, sport="soccer")
    except Exception as exc:  # noqa: BLE001 - tables missing / RLS / network
        raise _soccer_data_error(exc)


@app.get("/api/soccer/games")
def soccer_games(days: int = Query(10, ge=1, le=60)):
    """Upcoming matches in the next `days` days (World Cup games first)."""
    _require_soccer()
    try:
        return {"games": soccer_game_engine.upcoming_games(days=days)}
    except Exception as exc:  # noqa: BLE001 - tables missing / RLS / network
        raise _soccer_data_error(exc)


@app.get("/api/soccer/game")
def soccer_game(
    home: str = Query(..., description="home team, e.g. Mexico"),
    away: str = Query(..., description="away team, e.g. South Africa"),
    date: str | None = Query(None, description="match date YYYY-MM-DD; omit to auto-detect"),
    match_id: int | None = Query(None, description="schedule match_id, if known"),
    total_line: float = Query(2.5, description="goals total line to grade"),
):
    """Match outcome: win/draw/win probabilities + projected goals."""
    _require_soccer()
    try:
        return soccer_game_engine.project_soccer_game(
            home=home, away=away, match_date=date, match_id=match_id,
            total_line=total_line,
        )
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - tables missing / RLS / network
        raise _soccer_data_error(exc)


# --- NFL (schedule stage -- projections come with the later pipeline stages) --
def _require_nfl():
    if nfl_common is None:
        raise HTTPException(
            status_code=503,
            detail=f"NFL side unavailable: {_nfl_load_error}",
        )


def _nfl_data_error(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=f"NFL data unavailable ({exc}). If this is a fresh setup, run "
               f"the SQL + schedule/roster load in NFL_SETUP.md.",
    )


@app.get("/api/nfl/games")
def nfl_games(days: int = Query(30, ge=1, le=250)):
    """Upcoming NFL games in the next `days` days (soonest first)."""
    _require_nfl()
    try:
        return {"games": nfl_common.upcoming_games(days=days)}
    except Exception as exc:  # noqa: BLE001 - table missing / RLS / network
        raise _nfl_data_error(exc)


@app.get("/api/nfl/players")
def nfl_players(q: str = Query("", description="name fragment"),
                 limit: int = Query(10, ge=1, le=25)):
    """Player autocomplete against nfl_players (roster directory; no stats
    yet -- see NFL_SETUP.md)."""
    _require_nfl()
    try:
        return {"players": nfl_common.search_players(q, limit=limit)}
    except Exception as exc:  # noqa: BLE001 - table missing / RLS / network
        raise _nfl_data_error(exc)


@app.get("/api/nfl/roster")
def nfl_roster(
    home: str = Query(..., description="home team, e.g. Kansas City Chiefs or KC"),
    away: str = Query(..., description="away team, e.g. Buffalo Bills or BUF"),
):
    """Both teams' rosters for a matchup, offense -> defense -> specialists.
    No per-player stats yet (that's the next pipeline stage) -- tapping a
    player is a placeholder in the frontend until then."""
    _require_nfl()
    try:
        return nfl_common.roster_for_game(home, away)
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - table missing / RLS / network
        raise _nfl_data_error(exc)


# --- Fantasy football advisor (Sleeper + ESPN leagues, Vegas lines) ---------
# No Supabase involved: everything comes from the platforms' public APIs (see
# FANTASY_SETUP.md). ESPN private-league cookies are sent in POST bodies, used
# for the one upstream call, and never stored.
_LEAGUE_TTL = 300
_league_cache = {}
# Values that end up in upstream URL paths are validated so a crafted input
# can't walk to another endpoint (e.g. "../" on the Odds API with our key).
_SEASON_RE = re.compile(r"^20\d\d$")
_SLEEPER_USER_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,39}$")
_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9]{1,64}$")


def _fantasy_error(exc: Exception) -> HTTPException:
    if isinstance(exc, fantasy_sources.SourceError):
        return HTTPException(status_code=exc.status, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


def _season(raw) -> str:
    season = str(raw or fantasy_sources.nfl_state()["season"]).strip()
    if not _SEASON_RE.match(season):
        raise HTTPException(status_code=400, detail="Season must be a year like 2026.")
    return season


def _week(raw):
    """None (= current week) or an int 1-22."""
    if raw in (None, "", 0, "0"):
        return None
    try:
        wk = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Week must be a number.") from None
    if not 1 <= wk <= 22:
        raise HTTPException(status_code=400, detail="Week must be between 1 and 22.")
    return wk


def _load_fantasy_league(body: dict):
    platform = (body.get("platform") or "").lower()
    league_id = str(body.get("league_id") or "").strip()
    if platform not in ("sleeper", "espn") or not league_id.isdigit():
        raise HTTPException(status_code=400,
                            detail="Send platform ('sleeper' or 'espn') and a numeric league_id.")
    season = _season(body.get("season"))
    s2, swid = body.get("espn_s2") or None, body.get("swid") or None
    key = (platform, league_id, season, hash((s2, swid)))
    now = time.time()
    hit = _league_cache.get(key)
    if hit and now - hit[0] < _LEAGUE_TTL and not body.get("refresh"):
        return hit[1]
    league = fantasy_engine.load_league(platform, league_id, season, espn_s2=s2, swid=swid)
    for k in [k for k, v in _league_cache.items() if now - v[0] >= _LEAGUE_TTL]:
        _league_cache.pop(k, None)
    _league_cache[key] = (now, league)
    return league


@app.get("/api/fantasy/sleeper/leagues")
def fantasy_sleeper_leagues(username: str = Query(..., min_length=1),
                            season: str = Query("", description="defaults to current")):
    """Every NFL league a Sleeper user is in this season."""
    username = username.strip()
    if not _SLEEPER_USER_RE.match(username):
        raise HTTPException(status_code=400, detail="That doesn't look like a Sleeper username.")
    try:
        return fantasy_sources.sleeper_user_leagues(username, _season(season))
    except fantasy_sources.SourceError as exc:
        raise _fantasy_error(exc)


@app.post("/api/fantasy/league")
def fantasy_league(body: dict = Body(..., description="{platform, league_id, season?, espn_s2?, swid?}")):
    """Connect a league: returns its settings and team list (pick yours)."""
    try:
        lg = _load_fantasy_league(body)
    except (fantasy_sources.SourceError, ValueError) as exc:
        raise _fantasy_error(exc)
    return {
        "platform": lg["platform"], "league_id": lg["league_id"], "name": lg["name"],
        "season": lg["season"], "slots": lg["slots"],
        "scoring": fantasy_engine._scoring_label(lg["scoring"]),
        "teams": [{"team_id": t["team_id"], "name": t["name"], "owner": t["owner"],
                   "owner_id": t.get("owner_id")} for t in lg["teams"]],
        "unmatched_players": lg.get("unmatched_players") or [],
    }


@app.post("/api/fantasy/analyze")
def fantasy_analyze(body: dict = Body(..., description="{platform, league_id, team_id, season?, week?, espn_s2?, swid?}")):
    """The full report for your team: start/sit (Vegas-adjusted), team needs
    across the league, trade ideas, buy-low / sell-high, and waiver targets."""
    try:
        lg = _load_fantasy_league(body)
        return fantasy_engine.analyze_league(lg, str(body.get("team_id") or ""),
                                             _week(body.get("week")))
    except (fantasy_sources.SourceError, ValueError) as exc:
        raise _fantasy_error(exc)


@app.post("/api/fantasy/trade")
def fantasy_trade(body: dict = Body(..., description="{platform, league_id, team_id, partner_id, give:[ids], get:[ids], ...}")):
    """Grade a specific trade -- one you're drafting or one you were sent."""
    give = list(dict.fromkeys(str(p) for p in body.get("give") or []))
    get = list(dict.fromkeys(str(p) for p in body.get("get") or []))
    if set(give) & set(get):
        raise HTTPException(status_code=400, detail="A player can't be on both sides of a trade.")
    if len(give) > 6 or len(get) > 6:
        raise HTTPException(status_code=400, detail="Up to 6 players per side.")
    try:
        lg = _load_fantasy_league(body)
        ctx = fantasy_engine.prepare(lg, _week(body.get("week")))
        needs = fantasy_engine.team_needs(lg, ctx["uni"], ctx["repl"], ctx["per_team"])
        return fantasy_engine.evaluate_trade(
            lg, ctx["uni"], needs, ctx["repl"], ctx["per_team"],
            str(body.get("team_id") or ""), str(body.get("partner_id") or ""), give, get)
    except (fantasy_sources.SourceError, ValueError) as exc:
        raise _fantasy_error(exc)


@app.get("/api/fantasy/odds")
def fantasy_odds(week: int = Query(0, ge=0, le=22)):
    """This week's NFL lines: spread, total, moneyline, implied team totals."""
    state = fantasy_sources.nfl_state()
    wk = week or state["week"]
    try:
        games = fantasy_sources.game_lines(state["season"], wk, state["season_type"])
    except fantasy_sources.SourceError as exc:
        raise _fantasy_error(exc)
    return {"season": state["season"], "week": wk, "games": games,
            "props_available": bool(fantasy_sources._odds_api_key())}


@app.get("/api/fantasy/players")
def fantasy_players(q: str = Query("", description="name fragment"),
                    limit: int = Query(10, ge=1, le=25)):
    """Autocomplete over Sleeper's NFL player database (for the parlay builder)."""
    try:
        return {"players": fantasy_sources.search_players(q, limit=limit)}
    except fantasy_sources.SourceError as exc:
        raise _fantasy_error(exc)


def _week_context(week):
    state = fantasy_sources.nfl_state()
    wk = _week(week) or state["week"]
    proj = fantasy_sources.sleeper_week_projections(state["season"], wk)
    try:
        games = fantasy_sources.game_lines(state["season"], wk, state["season_type"])
    except fantasy_sources.SourceError:
        games = []
    return proj, games


@app.post("/api/fantasy/parlay")
def fantasy_parlay(body: dict = Body(..., description="{legs:[...], week?}")):
    """Grade a parlay: player props (our projection vs your line) and game
    lines (spread/total/moneyline, priced off the market)."""
    legs = body.get("legs")
    try:
        fantasy_engine.validate_legs(legs)
        proj, games = _week_context(body.get("week"))
        return fantasy_engine.grade_parlay(legs, proj, fantasy_sources.sleeper_players(), games)
    except (fantasy_sources.SourceError, ValueError) as exc:
        raise _fantasy_error(exc)


@app.get("/api/fantasy/props")
def fantasy_props(event_id: str = Query(..., min_length=1), week: int = Query(0, ge=0, le=22)):
    """Every player-prop line for one game, graded by the model (needs ODDS_API_KEY)."""
    if not _EVENT_ID_RE.match(event_id):
        raise HTTPException(status_code=400, detail="Bad event id.")
    try:
        proj, games = _week_context(week)
        props = fantasy_sources.event_props(event_id)
        return {"props": fantasy_engine.grade_props_board(
            props, proj, fantasy_sources.sleeper_players(), games)}
    except fantasy_sources.SourceError as exc:
        raise _fantasy_error(exc)


# ESPN roster positions -> Sleeper fantasy positions.
_NFL_POS_TO_FANTASY = {"QB": "QB", "RB": "RB", "FB": "RB", "HB": "RB", "WR": "WR",
                       "TE": "TE", "K": "K", "PK": "K"}


@app.get("/api/fantasy/player")
def fantasy_player(name: str = Query(..., min_length=2, max_length=80),
                   team: str = Query("", max_length=40, description="full name or abbr"),
                   position: str = Query("", max_length=4),
                   week: int = Query(0, ge=0, le=22)):
    """This week's projection for one NFL player (powers the NFL tab's player
    taps): stat line, fantasy points in PPR/half/standard, Vegas context."""
    pos = position.strip().upper()
    if pos and pos not in _NFL_POS_TO_FANTASY:
        return {"available": False,
                "reason": "Projections cover fantasy positions only (QB, RB, WR, TE, K)."}
    try:
        players = fantasy_sources.sleeper_players()
        pid = fantasy_engine.find_player(players, name, fantasy_sources.team_abbr(team),
                                         _NFL_POS_TO_FANTASY.get(pos, ""))
        if not pid:
            return {"available": False,
                    "reason": f"Couldn't find {name} in Sleeper's player database."}
        state = fantasy_sources.nfl_state()
        wk = week or state["week"]
        proj, games = _week_context(wk)
        return {"available": True,
                **fantasy_engine.player_outlook(players, proj, games, pid, wk)}
    except fantasy_sources.SourceError as exc:
        raise _fantasy_error(exc)


# --- Bet-slip analyzer -------------------------------------------------------
# Accept up to ~10 MB; phone screenshots are well under this.
_MAX_SLIP_BYTES = 10 * 1024 * 1024
_ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/heic"}


@app.get("/api/slip/health")
def slip_health():
    """Whether the slip analyzer can run (needs a Gemini key for OCR/fallback)."""
    return {"llm_available": llm_analysis.available(), "model": llm_analysis.GEMINI_MODEL}


@app.post("/api/analyze-slip")
async def analyze_slip(image: UploadFile = File(..., description="bet-slip screenshot")):
    """Read a bet-slip/parlay screenshot and grade every leg.

    Soccer/NBA player props we track are graded with our own model; everything
    else (other sports, team markets, untracked players) falls back to Gemini.
    Returns {bet_type, book, legs:[...], parlay:{...}}."""
    mime = (image.content_type or "image/png").lower()
    if mime not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{mime}'. Use PNG, JPEG, WEBP or HEIC.",
        )
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > _MAX_SLIP_BYTES:
        raise HTTPException(status_code=400, detail="Image too large (max 10 MB).")
    # analyze() makes blocking Gemini + Supabase calls; run it off the event
    # loop so one slip doesn't freeze the rest of the API for several seconds.
    work = functools.partial(
        slip_analysis.analyze, data, mime,
        nba_engine=engine, soccer_engine=soccer_engine,
    )
    try:
        return await run_in_threadpool(work)
    except llm_analysis.LLMUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


# --- Static frontend (production) -------------------------------------------
# In production (Railway/Docker) the built React app lives in frontend/dist and
# is served by this same process, so one service runs everything. Mounted LAST
# so the /api routes above keep priority; html=True makes / serve index.html.
# In development this directory may not exist -- the Vite dev server handles
# the frontend and proxies /api here instead.
_dist = os.path.join(HERE, "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")

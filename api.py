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

# NFL. Loaded defensively like soccer: a missing table, missing models or bad
# credentials must never break the NBA/soccer apps. nfl_common carries the
# schedule/roster/ratings; the two engines add projections (props + game).
try:
    import nfl_common
    nfl_engine = _load_numbered("nfl_projection_engine", "38_nfl_projections.py")
    nfl_game_engine = _load_numbered("nfl_game_projection_engine", "39_nfl_game_projections.py")
    _nfl_load_error = None
except (Exception, SystemExit) as exc:  # noqa: BLE001 - NFL must never break the others
    nfl_common = nfl_engine = nfl_game_engine = None
    _nfl_load_error = str(exc)

# Daily "My Picks" boards (computed in the background, cached per day).
import daily_picks

# Bet-slip analyzer (screenshot -> per-leg hit probabilities). Imports plainly
# because its filename has no leading digit; it talks to the engines we pass in.
import slip_analysis
import llm_analysis
import multi_props

daily_picks.init(
    nba=engine, nba_game=game_engine,
    soccer=soccer_engine, soccer_game=soccer_game_engine,
    nfl=nfl_engine, nfl_game=nfl_game_engine,
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
    if sport not in ("nba", "soccer", "nfl"):
        raise HTTPException(status_code=400, detail="sport must be nba, nfl or soccer")
    if sport == "soccer":
        _require_soccer()
    if sport == "nfl":
        _require_nfl()
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


# --- NFL (full stack: schedule, rosters, props + game outcomes) -------------
def _require_nfl():
    if nfl_common is None or nfl_engine is None or nfl_game_engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"NFL side unavailable: {_nfl_load_error}",
        )


def _nfl_data_error(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=f"NFL data unavailable ({exc}). If this is a fresh setup, run "
               f"the SQL + schedule/roster/log load in NFL_SETUP.md.",
    )


@app.get("/api/nfl/games")
def nfl_games(days: int = Query(30, ge=1, le=250)):
    """Upcoming NFL games in the next `days` days (soonest first)."""
    _require_nfl()
    try:
        return {"games": nfl_common.upcoming_games(days=days)}
    except Exception as exc:  # noqa: BLE001 - table missing / RLS / network
        raise _nfl_data_error(exc)


@app.get("/api/nfl/stats")
def nfl_stats():
    """The NFL stats the tool can project (sorted for a stable dropdown)."""
    _require_nfl()
    return {"stats": sorted(nfl_engine.STAT_DEFS.keys())}


@app.get("/api/nfl/players")
def nfl_players(q: str = Query("", description="name fragment"),
                 limit: int = Query(10, ge=1, le=25)):
    """Player autocomplete against nfl_players (roster directory)."""
    _require_nfl()
    try:
        return {"players": nfl_common.search_players(q, limit=limit)}
    except Exception as exc:  # noqa: BLE001 - table missing / RLS / network
        raise _nfl_data_error(exc)


@app.get("/api/nfl/project")
def nfl_project(
    player: str = Query(..., description="exact player name from autocomplete"),
    stat: str = Query(..., description="one of /api/nfl/stats"),
    line: float | None = Query(None, description="the over/under line from your book"),
    opponent: str | None = Query(None, description="opponent team; omit to auto-detect"),
    location: str = Query("auto", description="auto | home | away"),
    game_type: str = Query("auto", description="auto | regular | playoffs"),
):
    """Project an NFL stat and (if a line is given) grade it."""
    _require_nfl()
    home_away = {"home": "HOME", "away": "AWAY"}.get(location.lower())
    season_type = {"regular": "regular", "playoffs": "playoffs"}.get(game_type.lower(), "auto")
    try:
        return nfl_engine.project_player(
            player_name=player, stat=stat, line=line, opponent=opponent or None,
            home_away=home_away, season_type=season_type,
        )
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - table missing / RLS / network
        raise _nfl_data_error(exc)


@app.get("/api/nfl/player/projections")
def nfl_player_projections(
    player: str = Query(..., description="exact player name from autocomplete"),
    stats: str | None = Query(None, description="comma-separated stats; omit for defaults"),
    opponent: str | None = Query(None, description="opponent team; omit to auto-detect"),
    location: str = Query("auto", description="auto | home | away"),
    game_type: str = Query("auto", description="auto | regular | playoffs"),
):
    """One player's projection across several NFL stats at once."""
    _require_nfl()
    stat_list = [s.strip() for s in stats.split(",")] if stats else None
    try:
        return multi_props.player_projections(
            nfl_engine, player, stats=stat_list, opponent=opponent or None,
            location=None if location == "auto" else location,
            game_type=game_type, sport="nfl",
        )
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - table missing / RLS / network
        raise _nfl_data_error(exc)


@app.post("/api/nfl/project-batch")
def nfl_project_batch(payload: dict = Body(..., description='{"props": [{player, stat, line, side?, opponent?, location?, game_type?}]}')):
    """Grade a hand-built list of NFL props as a parlay (NFL twin of
    /api/project-batch)."""
    _require_nfl()
    props = payload.get("props") or []
    if not isinstance(props, list) or not props:
        raise HTTPException(status_code=400, detail="Send a non-empty 'props' list.")
    if len(props) > 25:
        raise HTTPException(status_code=400, detail="Too many props (max 25).")
    try:
        return multi_props.grade_batch(nfl_engine, props, sport="nfl")
    except Exception as exc:  # noqa: BLE001 - table missing / RLS / network
        raise _nfl_data_error(exc)


@app.get("/api/nfl/roster")
def nfl_roster(
    home: str = Query(..., description="home team, e.g. Kansas City Chiefs or KC"),
    away: str = Query(..., description="away team, e.g. Buffalo Bills or BUF"),
):
    """Both teams' rosters for a matchup, featured players first. Tapping a
    player projects the right matchup (opponent + home/away carried per row)."""
    _require_nfl()
    try:
        return multi_props.nfl_roster(nfl_engine, home, away)
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - table missing / RLS / network
        raise _nfl_data_error(exc)


@app.get("/api/nfl/game")
def nfl_game(
    home: str = Query(..., description="home team, e.g. KC or Kansas City Chiefs"),
    away: str = Query(..., description="away team, e.g. BUF or Buffalo Bills"),
    date: str | None = Query(None, description="game date YYYY-MM-DD; omit to auto-detect"),
    game_id: int | None = Query(None, description="schedule game_id, if known"),
    game_type: str = Query("auto", description="auto | regular | playoffs"),
):
    """Game outcome: win / tie / loss, projected score, spread, total, team TDs."""
    _require_nfl()
    season_type = {"regular": "regular", "playoffs": "playoffs"}.get(game_type.lower(), "auto")
    try:
        return nfl_game_engine.project_game(
            home=home, away=away, game_date=date, game_id=game_id,
            season_type=season_type,
        )
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - table missing / RLS / network
        raise _nfl_data_error(exc)


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
        nba_engine=engine, soccer_engine=soccer_engine, nfl_engine=nfl_engine,
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

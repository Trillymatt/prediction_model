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

Setup:
    pip install -r requirements.txt
    # needs the same .env (SUPABASE_URL / SUPABASE_KEY) as the scripts
"""

import os
import re
import importlib.util

from fastapi import FastAPI, HTTPException, Query
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

# Daily "My Picks" boards (computed in the background, cached per day).
import daily_picks

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
    """Today's "My Picks" board: the model's most confident calls for the
    slate. Returns status=building while the daily board is being computed;
    the frontend polls until it's ready."""
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


# --- Static frontend (production) -------------------------------------------
# In production (Railway/Docker) the built React app lives in frontend/dist and
# is served by this same process, so one service runs everything. Mounted LAST
# so the /api routes above keep priority; html=True makes / serve index.html.
# In development this directory may not exist -- the Vite dev server handles
# the frontend and proxies /api here instead.
_dist = os.path.join(HERE, "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")

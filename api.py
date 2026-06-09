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

Setup:
    pip install -r requirements.txt
    # needs the same .env (SUPABASE_URL / SUPABASE_KEY) as the scripts
"""

import os
import importlib.util

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


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


app = FastAPI(title="Money From a Baby API", version="1.0")

# Dev-friendly CORS so the Vite dev server can call us directly if it isn't
# proxying. (The frontend also proxies /api -> here, which needs no CORS.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        .ilike("player_name", f"%{q}%")
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


# --- Static frontend (production) -------------------------------------------
# In production (Railway/Docker) the built React app lives in frontend/dist and
# is served by this same process, so one service runs everything. Mounted LAST
# so the /api routes above keep priority; html=True makes / serve index.html.
# In development this directory may not exist -- the Vite dev server handles
# the frontend and proxies /api here instead.
_dist = os.path.join(HERE, "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")

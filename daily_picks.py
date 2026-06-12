"""
Daily "My Picks": the model's most confident calls for today's slate.

For each sport we find who is actually playing today (falling back to the next
slate within a week if today is dark), keep only rotation-relevant players,
run the existing projection engines over them, and keep the claims the model
is most sure about:

  NBA     "25+ Points -- 84%"   (highest milestone cleared with >= 70% prob)
  Soccer  "To Score -- 52%", "No Goal -- 88%", "To Assist -- 41%"

Every pick carries the fully-graded projection (factors and all) for the
headline claim, plus graded projections for the player's other stats, so the
frontend can show "why" and "what else he's projected for" with zero extra
round-trips when a pick is tapped.

Building a board costs a few hundred Supabase reads, so it can't run per page
load: the first request of the day kicks off a background build (api.py also
warms both sports at startup) and the result is cached in-process for the rest
of the day. /api/picks returns {"status": "building"} until it's done and the
frontend polls.

CLI check:  python daily_picks.py nba|soccer
"""

import importlib.util
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
BOARD_SIZE = 6          # picks shown per sport
MAX_PER_TEAM = 3        # so a 2-team slate (Finals) still varies a little
MAX_WORKERS = 4         # parallel engine calls during a build

NBA_STATS = ("points", "rebounds", "assists")
NBA_STAT_NOUNS = {"points": "Points", "rebounds": "Rebounds", "assists": "Assists"}
# "X+" alt-line ladders. The headline is the HIGHEST rung the model clears
# with >= NBA_HEADLINE_MIN_P, so confidence stays meaningful instead of every
# star showing a trivial "10+ points -- 99%".
NBA_LADDERS = {
    "points": (10, 15, 20, 25, 30, 35, 40),
    "rebounds": (4, 6, 8, 10, 12, 14, 16),
    "assists": (2, 4, 6, 8, 10, 12),
}
NBA_HEADLINE_MIN_P = 0.70
NBA_MIN_L10_MINUTES = 24.0   # "players that actually play"
NBA_PLAYERS_PER_TEAM = 5

SOCCER_PLAYERS_PER_TEAM = 5
SOCCER_MIN_RECENT_MINUTES = 30.0
# Poisson anytime-scorer probabilities live in the 0.20-0.45 range (a 39%
# "to score" is a genuinely strong call -- books price top strikers around
# there), so these floors are about cutting junk, not demanding certainty.
TO_SCORE_MIN_P = 0.30
TO_ASSIST_MIN_P = 0.25
NO_GOAL_MIN_P = 0.78
NO_GOAL_MIN_PER90 = 0.30     # only fade players people would actually back
MAX_NO_GOAL_PICKS = 2        # keep the board from being all negatives


# ---------------------------------------------------------------------------
# Engine wiring. api.py injects its already-loaded engine modules via init();
# the CLI path loads them itself.
# ---------------------------------------------------------------------------
_engines = {"nba": None, "nba_game": None, "soccer": None, "soccer_game": None}


def init(nba=None, nba_game=None, soccer=None, soccer_game=None):
    for key, mod in (("nba", nba), ("nba_game", nba_game),
                     ("soccer", soccer), ("soccer_game", soccer_game)):
        if mod is not None:
            _engines[key] = mod


def _load_numbered(module_name, filename):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(here, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_engines(sport):
    if sport == "nba":
        if _engines["nba"] is None:
            _engines["nba"] = _load_numbered("projection_engine", "09_projections.py")
        if _engines["nba_game"] is None:
            _engines["nba_game"] = _load_numbered("game_projection_engine", "14_game_projections.py")
    else:
        if _engines["soccer"] is None:
            _engines["soccer"] = _load_numbered("soccer_projection_engine", "22_soccer_projections.py")
        if _engines["soccer_game"] is None:
            _engines["soccer_game"] = _load_numbered("soccer_game_projection_engine", "23_soccer_game_projections.py")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _cap_board(picks, is_no_goal=lambda p: False):
    """Top-of-board selection: confidence order, with per-team and
    no-goal caps so one team / one pick type can't fill the whole board."""
    board, team_counts, no_goal_count = [], {}, 0
    for p in picks:
        if len(board) >= BOARD_SIZE:
            break
        team = p.get("team")
        if team_counts.get(team, 0) >= MAX_PER_TEAM:
            continue
        if is_no_goal(p):
            if no_goal_count >= MAX_NO_GOAL_PICKS:
                continue
            no_goal_count += 1
        team_counts[team] = team_counts.get(team, 0) + 1
        board.append(p)
    return board


# ---------------------------------------------------------------------------
# NBA
# ---------------------------------------------------------------------------
def _grade_normal(nba, result, line):
    """Grade an ungraded NBA projection against `line`, mirroring the fields
    project_player() adds when the caller supplies a line."""
    g = dict(result)
    z = (result["projection"] - line) / result["sigma"]
    p_over = nba.normal_cdf(z)
    confidence = max(p_over, 1.0 - p_over)
    g.update({
        "line": line,
        "p_over": round(p_over, 4),
        "p_under": round(1.0 - p_over, 4),
        "recommendation": "OVER" if p_over >= 0.5 else "UNDER",
        "confidence": round(confidence, 4),
        "confidence_label": nba.confidence_label(confidence),
    })
    return g


def _best_milestone(projection, sigma, ladder, min_p, normal_cdf):
    """Highest rung of the ladder hit with probability >= min_p, or None.
    'm+' means clearing the X.5 line just below m."""
    best = None
    for m in ladder:
        p = normal_cdf((projection - (m - 0.5)) / sigma)
        if p >= min_p:
            best = (m, p)
    return best


def _nba_candidates(nba, team_abbrs):
    """Rotation-relevant players on the slate's teams: nba_players rosters
    filtered by last-10 minutes from nba_player_averages."""
    from nba_api.stats.static import teams as static_teams

    # nba_players.team stores nicknames ("Knicks"); accept full names too.
    wanted = set()
    for t in static_teams.get_teams():
        if t["abbreviation"] in team_abbrs:
            wanted.update((t["nickname"], t["full_name"]))
    if not wanted:
        return []

    res = (
        nba.supabase.table(nba.PLAYERS_TABLE)
        .select("player_id,player_name,team,position")
        .in_("team", sorted(wanted))
        .execute()
    )
    roster = res.data or []

    averages = {}
    ids = [r["player_id"] for r in roster]
    for chunk in _chunks(ids, 150):
        ares = (
            nba.supabase.table("nba_player_averages")
            .select("player_id,last_10_minutes,season_avg_points")
            .in_("player_id", chunk)
            .execute()
        )
        for a in ares.data or []:
            averages[a["player_id"]] = a

    by_team = {}
    for r in roster:
        l10_minutes = (averages.get(r["player_id"]) or {}).get("last_10_minutes") or 0
        if l10_minutes < NBA_MIN_L10_MINUTES:
            continue
        r["_l10_minutes"] = l10_minutes
        by_team.setdefault(r["team"], []).append(r)

    out = []
    for players in by_team.values():
        players.sort(key=lambda r: r["_l10_minutes"], reverse=True)
        out.extend(players[:NBA_PLAYERS_PER_TEAM])
    return out


def _nba_player_pick(nba, cand, slate_date):
    """One player -> a pick dict (headline + all graded predictions), or None
    if he's hurt / has no confident milestone."""
    inj = nba.injury_status(cand["player_id"])
    status = ((inj or {}).get("status") or "").lower()
    if "out" in status or "doubt" in status:
        return None

    results = {}
    for stat in NBA_STATS:
        r = _project_with_retry(nba.project_player, cand["player_name"], stat)
        if r and r.get("sigma"):
            results[stat] = r
    if not results:
        return None

    headline = None  # (stat, milestone, p)
    for stat, r in results.items():
        hit = _best_milestone(r["projection"], r["sigma"],
                              NBA_LADDERS[stat], NBA_HEADLINE_MIN_P, nba.normal_cdf)
        if hit and (headline is None or hit[1] > headline[2]):
            headline = (stat, hit[0], hit[1])
    if headline is None:
        return None

    stat, milestone, p = headline
    predictions = [_grade_normal(nba, results[stat], milestone - 0.5)]
    # The other stats, graded at the highest rung the model at least leans
    # over (or the lowest rung as an honest UNDER read).
    for other, r in results.items():
        if other == stat:
            continue
        hit = _best_milestone(r["projection"], r["sigma"],
                              NBA_LADDERS[other], 0.5, nba.normal_cdf)
        m = hit[0] if hit else NBA_LADDERS[other][0]
        predictions.append(_grade_normal(nba, r, m - 0.5))

    r0 = results[stat]
    return {
        "sport": "nba",
        "player_id": cand["player_id"],
        "player_name": r0["player_name"],
        "team": r0["team"],
        "position": r0["position"],
        "opponent": r0["opponent"],
        "home_away": r0["home_away"],
        "game_date": slate_date,
        "stat": stat,
        "headline": f"{milestone}+ {NBA_STAT_NOUNS[stat]}",
        "direction": "OVER",
        "line": milestone - 0.5,
        "probability": round(p, 4),
        "predictions": predictions,
    }


def _build_nba():
    nba, nba_game = _engines["nba"], _engines["nba_game"]
    games = nba_game.upcoming_games(days=7)
    if not games:
        return {"slate_date": None, "picks": [],
                "note": "No NBA games in the next week."}
    slate_date = games[0]["game_date"]
    slate = [g for g in games if g["game_date"] == slate_date]
    team_abbrs = {g["home_team"] for g in slate} | {g["away_team"] for g in slate}

    candidates = _nba_candidates(nba, team_abbrs)
    nba.load_models()  # warm the model cache before threads race to load it
    picks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for res in pool.map(
            lambda c: _swallow(_nba_player_pick, nba, c, slate_date), candidates
        ):
            if res:
                picks.append(res)

    picks.sort(key=lambda p: p["probability"], reverse=True)
    note = (None if slate_date == date.today().isoformat()
            else f"No games today — showing the {slate_date} slate.")
    return {"slate_date": slate_date, "picks": _cap_board(picks), "note": note}


# ---------------------------------------------------------------------------
# Soccer
# ---------------------------------------------------------------------------
def _soccer_candidates(soccer, team):
    """Likely contributors for one team, from its match logs: recent regular
    minutes, ranked by goals+assists per 90 (which also buries keepers)."""
    sc = soccer.sc
    columns = "player_id,player_name,team,match_date,minutes_played,goals,assists"
    rows = sc.fetch_all(sc.LOGS_TABLE, columns, filters=[("eq", "team", team)],
                        order_col="match_date")
    if not rows and sc.normalize_team(team) != team:
        rows = sc.fetch_all(sc.LOGS_TABLE, columns,
                            filters=[("eq", "team", sc.normalize_team(team))],
                            order_col="match_date")
    if not rows:
        return []

    recent_team_dates = sorted({r["match_date"] for r in rows})[-3:]
    by_player = {}
    for r in rows:
        by_player.setdefault(r["player_id"], []).append(r)

    cands = []
    for pid, plist in by_player.items():
        played = [r for r in plist if r.get("minutes_played")]
        if not played:
            continue
        # Drop players who've fallen out of the squad.
        if played[-1]["match_date"] not in recent_team_dates:
            continue
        recent = played[-3:]
        recent_minutes = sum(r["minutes_played"] for r in recent) / len(recent)
        if recent_minutes < SOCCER_MIN_RECENT_MINUTES:
            continue
        total_minutes = sum(r["minutes_played"] for r in played)
        goals = sum(r.get("goals") or 0 for r in played)
        assists = sum(r.get("assists") or 0 for r in played)
        cands.append({
            "player_id": pid,
            "player_name": played[-1]["player_name"],
            "team": team,
            "g_per90": goals / total_minutes * 90 if total_minutes else 0.0,
            "ga_per90": (goals + assists) / total_minutes * 90 if total_minutes else 0.0,
        })

    cands.sort(key=lambda c: c["ga_per90"], reverse=True)
    return cands[:SOCCER_PLAYERS_PER_TEAM]


def _soccer_player_pick(soccer, cand):
    """One player -> his strongest claim among To Score / No Goal / To Assist."""
    results = {}
    for stat in ("goals", "assists"):
        r = _project_with_retry(soccer.project_soccer_player,
                                cand["player_name"], stat, line=0.5)
        if r:
            results[stat] = r

    claims = []  # (label, direction, p, result)
    rg, ra = results.get("goals"), results.get("assists")
    if rg and rg.get("p_over") is not None:
        if rg["p_over"] >= TO_SCORE_MIN_P:
            claims.append(("To Score", "OVER", rg["p_over"], rg))
        elif rg["p_under"] >= NO_GOAL_MIN_P and cand["g_per90"] >= NO_GOAL_MIN_PER90:
            claims.append(("No Goal", "UNDER", rg["p_under"], rg))
    if ra and ra.get("p_over") is not None and ra["p_over"] >= TO_ASSIST_MIN_P:
        claims.append(("To Assist", "OVER", ra["p_over"], ra))
    if not claims:
        return None

    claims.sort(key=lambda c: c[2], reverse=True)
    label, direction, p, r0 = claims[0]
    predictions = [r0] + [r for r in (rg, ra) if r is not None and r is not r0]
    return {
        "sport": "soccer",
        "player_id": r0.get("player_id"),
        "player_name": r0["player_name"],
        "team": r0["team"],
        "position": r0.get("position"),
        "opponent": r0.get("opponent"),
        "home_away": r0.get("home_away"),
        "game_date": r0.get("match_date"),
        "competition": r0.get("competition"),
        "stat": r0["stat"],
        "headline": label,
        "direction": direction,
        "line": 0.5,
        "probability": round(p, 4),
        "predictions": predictions,
    }


def _build_soccer():
    soccer, soccer_game = _engines["soccer"], _engines["soccer_game"]
    sc = soccer.sc
    games = soccer_game.upcoming_games(days=7)
    if not games:
        return {"slate_date": None, "picks": [],
                "note": "No matches in the next week — run 20_soccer_schedule.py."}
    slate_date = games[0]["match_date"]
    slate = [g for g in games if g["match_date"] == slate_date]

    teams = []
    for g in slate:
        for t in (g["home_team"], g["away_team"]):
            if t not in teams:
                teams.append(t)

    candidates = []
    for team in teams:
        candidates.extend(_soccer_candidates(soccer, team))

    picks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for res in pool.map(lambda c: _swallow(_soccer_player_pick, soccer, c),
                            candidates):
            if res:
                picks.append(res)

    picks.sort(key=lambda p: p["probability"], reverse=True)
    board = _cap_board(picks, is_no_goal=lambda p: p["headline"] == "No Goal")

    # Round out the detail view for the players who made the board.
    for p in board:
        for stat, line in (("shots", 1.5), ("shots_on_target", 0.5)):
            r = _project_with_retry(soccer.project_soccer_player,
                                    p["player_name"], stat, line=line)
            if r:
                p["predictions"].append(r)

    note = (None if slate_date == date.today().isoformat()
            else f"No matches today — showing the {slate_date} slate.")
    return {"slate_date": slate_date, "picks": board, "note": note}


def _swallow(fn, *args):
    """One bad player must never sink the whole board."""
    try:
        return fn(*args)
    except Exception:
        traceback.print_exc()
        return None


def _project_with_retry(fn, *args, **kwargs):
    """Engine calls run 4-wide during a build and very occasionally trip on a
    transient (network/concurrency) error; one retry recovers those instead of
    silently dropping a stat from a player's card."""
    for attempt in (1, 2):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == 2:
                traceback.print_exc()
                return None
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Cache + public API
# ---------------------------------------------------------------------------
_cache = {}   # (sport, iso_date) -> {"status", "ts", "picks", ...}
_lock = threading.Lock()
ERROR_RETRY_SECONDS = 60


def get_picks(sport: str) -> dict:
    """Today's board for a sport. Kicks off a background build on the first
    call of the day and returns {"status": "building"} until it lands."""
    today = date.today().isoformat()
    key = (sport, today)
    with _lock:
        for stale in [k for k in _cache if k[1] != today]:
            _cache.pop(stale)
        entry = _cache.get(key)
        retry = (entry and entry["status"] == "error"
                 and time.time() - entry["ts"] > ERROR_RETRY_SECONDS)
        if entry is None or retry:
            entry = {"status": "building", "ts": time.time(), "picks": []}
            _cache[key] = entry
            threading.Thread(target=_run_build, args=(sport, today),
                             daemon=True).start()
    return _public(sport, today, entry)


def warm():
    """Start both builds in the background (called at API startup)."""
    for sport in ("nba", "soccer"):
        try:
            get_picks(sport)
        except Exception:
            traceback.print_exc()


def _run_build(sport, today):
    try:
        _ensure_engines(sport)
        body = _build_nba() if sport == "nba" else _build_soccer()
        entry = {"status": "ready", "ts": time.time(), **body}
    except Exception as exc:  # noqa: BLE001 - report, don't crash the API
        traceback.print_exc()
        entry = {"status": "error", "ts": time.time(), "picks": [],
                 "error": str(exc)}
    with _lock:
        _cache[(sport, today)] = entry


def _public(sport, today, entry):
    return {
        "sport": sport,
        "date": today,
        "status": entry["status"],
        "slate_date": entry.get("slate_date"),
        "picks": entry.get("picks", []),
        "note": entry.get("note"),
        "error": entry.get("error"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    sport = (sys.argv[1] if len(sys.argv) > 1 else "nba").lower()
    if sport not in ("nba", "soccer"):
        raise SystemExit("usage: python daily_picks.py [nba|soccer]")
    _ensure_engines(sport)
    t0 = time.time()
    body = _build_nba() if sport == "nba" else _build_soccer()
    print(f"\n=== {sport.upper()} board for {body.get('slate_date')} "
          f"({time.time() - t0:.0f}s) ===")
    if body.get("note"):
        print(f"note: {body['note']}")
    for p in body["picks"]:
        vs = f"{'@' if p.get('home_away') == 'AWAY' else 'vs'} {p.get('opponent')}"
        print(f"  {p['probability'] * 100:5.1f}%  {p['player_name']:<24} "
              f"({p['team']} {vs})  {p['headline']}")
    if not body["picks"]:
        print("  (no picks)")


if __name__ == "__main__":
    main()

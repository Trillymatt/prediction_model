"""
Roster + multi-prop helpers for the API.

Three things the single-prop projection engine (09_projections.py) doesn't do on
its own, all built on top of it so the math stays identical to the rest of the
app:

  * nba_roster()        -- every player on both teams of a game, rotation
                           players first, so the frontend can show a whole game
                           in one place instead of searching player by player.
  * player_projections()-- one player's projection across several stats at once
                           ("what the model thinks he'll hit"), so you see the
                           numbers before deciding which line to bet.
  * grade_batch()       -- grade a hand-built list of props in one call and
                           score them as a parlay, mirroring slip_analysis so a
                           typed slip reads the same as a scanned one.

Everything funnels through engine.project_player(); this module only arranges
the inputs and combines the outputs.
"""

AVERAGES_TABLE = "nba_player_averages"

# Stats shown by default for "what the model thinks he'll hit". Scoring +
# boards + playmaking + threes + the headline combo, which covers the props
# people actually shop for without running every stat the engine supports.
DEFAULT_STATS = ["points", "rebounds", "assists", "threes", "pra"]

# Soccer equivalents: the markets people actually shop, in confidence order.
SOCCER_DEFAULT_STATS = ["goals", "assists", "goals_assists", "shots",
                        "shots_on_target"]
# Goalkeepers don't shoot -- lead with the keeper markets (saves, passes from
# the build-out, goals conceded / clean sheet from the goal model) and keep
# goals/assists at the back so the rare keeper goal is still there to bet.
# Stats whose data isn't loaded (e.g. saves before the column exists) are
# filtered out downstream, so this degrades to whatever is available.
SOCCER_GK_STATS = ["saves", "passes", "goals_conceded", "goals", "assists"]

# Recent-minutes floor for who shows on a soccer roster. Lower than the picks
# board's 30 so squad players (not just nailed-on starters) appear, while still
# burying players who've dropped out of the matchday squad.
SOCCER_ROSTER_MIN_RECENT_MINUTES = 20.0

# NFL default markets, by position -- the props people actually shop. A QB
# leads with passing, a back with rushing + receiving, a pass-catcher with
# receiving, and everyone gets the anytime-TD market. Stats without data are
# filtered downstream, so this degrades to whatever the engine can project.
NFL_DEFAULT_STATS = ["rush_rec_yds", "rec_yds", "receptions", "rush_yds", "any_td"]
NFL_QB_STATS = ["pass_yds", "pass_td", "completions", "interceptions", "rush_yds"]
NFL_RB_STATS = ["rush_yds", "rush_att", "rec_yds", "receptions", "any_td"]
NFL_REC_STATS = ["rec_yds", "receptions", "targets", "rec_td", "any_td"]
# Recent scrimmage/passing yards a player needs to show high on an NFL roster.
NFL_ROSTER_STAT_COLUMNS = ["pass_yds", "rush_yds", "rec_yds"]


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _season_type(game_type: str) -> str:
    return {
        "regular": "Regular Season",
        "playoffs": "Playoffs",
    }.get((game_type or "auto").lower(), "auto")


def _home_away(location: str):
    return {"home": "HOME", "away": "AWAY"}.get((location or "").lower())


def _suggest_line(projection: float) -> float:
    """A clean over/under line near the projection (always an X.5), so the UI can
    pre-fill a sensible line the user can tweak instead of leaving it blank.
    Floored at 0.5 so low-count soccer stats never suggest a 0 or negative line."""
    line = round(projection * 2) / 2          # nearest half-point
    if line == int(line):                     # whole number -> drop to the .5 below
        line -= 0.5
    return max(0.5, round(line, 1))


def _soccer_default_stats(engine, player_name):
    """Pick the right default markets for a soccer player by position: keeper
    markets for goalkeepers, the attacking set for everyone else. Falls back to
    the attacking set if the player or his position can't be resolved (the
    optional soccer_players directory may not carry a position)."""
    try:
        sc = engine.sc
        player = engine.find_player(player_name)
        if sc.is_goalkeeper(player.get("position")):
            return SOCCER_GK_STATS
    except Exception:  # noqa: BLE001 - unknown position => attacking defaults
        pass
    return SOCCER_DEFAULT_STATS


def _nfl_season_type(game_type: str) -> str:
    return {"regular": "regular", "playoffs": "playoffs"}.get(
        (game_type or "auto").lower(), "auto")


def _nfl_default_stats(engine, player_name):
    """Position-aware default markets for an NFL player (QB / RB / receiver),
    falling back to the general set if the position can't be resolved."""
    try:
        pos = (engine.find_player(player_name).get("position") or "").upper()
    except Exception:  # noqa: BLE001 - unknown position => general defaults
        return NFL_DEFAULT_STATS
    if pos == "QB":
        return NFL_QB_STATS
    if pos in ("RB", "FB"):
        return NFL_RB_STATS
    if pos in ("WR", "TE"):
        return NFL_REC_STATS
    return NFL_DEFAULT_STATS


def _project(engine, sport, player, stat, line, opponent, location, game_type):
    """Dispatch to the right engine. All return the same output shape
    (projection, sigma, p_over/p_under, recommendation, factors, ...) so
    everything downstream is sport-agnostic."""
    if sport == "soccer":
        return engine.project_soccer_player(
            player_name=player, stat=stat, line=line, opponent=opponent or None)
    if sport == "nfl":
        return engine.project_player(
            player_name=player, stat=stat, line=line, opponent=opponent or None,
            home_away=_home_away(location), season_type=_nfl_season_type(game_type))
    return engine.project_player(
        player_name=player, stat=stat, line=line, opponent=opponent or None,
        home_away=_home_away(location), season_type=_season_type(game_type))


# ---------------------------------------------------------------------------
# Roster: every player on both teams of a game
# ---------------------------------------------------------------------------
def _team_name_variants(abbrs):
    """abbrev -> the team-name strings nba_players.team might store (nickname or
    full name), so we can query the roster table by name and map back to abbrev."""
    from nba_api.stats.static import teams as static_teams

    wanted = {}  # team-name string -> abbreviation
    for t in static_teams.get_teams():
        if t["abbreviation"] in abbrs:
            wanted[t["nickname"]] = t["abbreviation"]
            wanted[t["full_name"]] = t["abbreviation"]
    return wanted


def nba_roster(engine, home: str, away: str) -> dict:
    """Both teams' rosters for a matchup, players ordered by recent minutes so
    the rotation sits at the top and deep bench at the bottom.

    Each player carries enough to render a row and, when tapped, to project the
    right matchup (the player's opponent is the other team; location follows
    which side he's on)."""
    home, away = home.upper(), away.upper()
    wanted = _team_name_variants({home, away})
    if not wanted:
        raise ValueError(f"Unknown team abbreviation in '{home}'/'{away}'.")

    res = (
        engine.supabase.table(engine.PLAYERS_TABLE)
        .select("player_id,player_name,team,position")
        .in_("team", sorted(wanted))
        .execute()
    )
    roster = res.data or []

    # Recent-minutes + scoring averages, used only to rank the list.
    averages = {}
    ids = [r["player_id"] for r in roster]
    for chunk in _chunks(ids, 150):
        ares = (
            engine.supabase.table(AVERAGES_TABLE)
            .select("player_id,last_10_minutes,season_avg_points")
            .in_("player_id", chunk)
            .execute()
        )
        for a in ares.data or []:
            averages[a["player_id"]] = a

    sides = {home: "home", away: "away"}
    opponents = {home: away, away: home}
    by_abbr = {home: [], away: []}
    for r in roster:
        abbr = wanted.get(r["team"])
        if abbr not in by_abbr:
            continue
        a = averages.get(r["player_id"]) or {}
        by_abbr[abbr].append({
            "player_id": r["player_id"],
            "player_name": r["player_name"],
            "team": r["team"],
            "team_abbr": abbr,
            "position": r.get("position"),
            "l10_minutes": a.get("last_10_minutes"),
            "ppg": a.get("season_avg_points"),
            "opponent": opponents[abbr],
            "location": sides[abbr],
        })

    teams = []
    for abbr in (home, away):
        players = by_abbr[abbr]
        players.sort(key=lambda p: (p.get("l10_minutes") or 0), reverse=True)
        teams.append({
            "abbr": abbr,
            "side": sides[abbr],
            "opponent": opponents[abbr],
            "players": players,
        })

    return {"home_team": home, "away_team": away, "teams": teams}


# ---------------------------------------------------------------------------
# Roster: every player on both teams of a soccer match
# ---------------------------------------------------------------------------
def _soccer_team_players(sc, team: str) -> list:
    """Squad players for one team from its match logs: recent regular minutes,
    most-used first. Built from logs (not the optional soccer_players table) so
    it works wherever the picks board does."""
    columns = "player_id,player_name,team,match_date,minutes_played,goals,assists"
    rows = sc.fetch_all(sc.LOGS_TABLE, columns,
                        filters=[("eq", "team", team)], order_col="match_date")
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

    out = []
    for pid, plist in by_player.items():
        played = [r for r in plist if r.get("minutes_played")]
        if not played:
            continue
        # Drop players who've fallen out of the matchday squad.
        if played[-1]["match_date"] not in recent_team_dates:
            continue
        recent = played[-3:]
        recent_minutes = sum(r["minutes_played"] for r in recent) / len(recent)
        if recent_minutes < SOCCER_ROSTER_MIN_RECENT_MINUTES:
            continue
        total_minutes = sum(r["minutes_played"] for r in played)
        goals = sum(r.get("goals") or 0 for r in played)
        assists = sum(r.get("assists") or 0 for r in played)
        out.append({
            "player_id": pid,
            "player_name": played[-1]["player_name"],
            "team": played[-1].get("team") or team,
            "position": None,
            "recent_minutes": round(recent_minutes, 1),
            "ga_per90": round((goals + assists) / total_minutes * 90, 2)
                        if total_minutes else 0.0,
        })

    out.sort(key=lambda p: p["recent_minutes"], reverse=True)
    return out


def _soccer_positions(sc, player_ids) -> dict:
    """player_id -> position from the optional soccer_players directory, so the
    roster can flag goalkeepers (and the frontend lead with keeper markets).
    Returns {} if the table/column isn't there -- positions just stay unknown."""
    if not player_ids:
        return {}
    out = {}
    try:
        for chunk in _chunks(list(player_ids), 150):
            res = (
                sc.supabase.table(sc.PLAYERS_TABLE)
                .select("player_id,position")
                .in_("player_id", chunk)
                .execute()
            )
            for row in res.data or []:
                out[row["player_id"]] = row.get("position")
    except Exception:  # noqa: BLE001 - directory missing => positions unknown
        return {}
    return out


def soccer_roster(soccer, home: str, away: str) -> dict:
    """Both squads for a match, most-used players first -- the soccer twin of
    nba_roster(). Player opponent is carried at the team level (soccer
    projections take only an opponent, no home/away split). Each player carries
    his position + an is_goalkeeper flag so the UI can lead keepers with saves
    instead of goals."""
    sc = soccer.sc
    home_n = sc.normalize_team(home)
    away_n = sc.normalize_team(away)
    teams = []
    for team, side, opp in ((home_n, "home", away_n), (away_n, "away", home_n)):
        players = _soccer_team_players(sc, team)
        positions = _soccer_positions(sc, [p["player_id"] for p in players])
        for p in players:
            pos = positions.get(p["player_id"])
            if pos:
                p["position"] = pos
            p["is_goalkeeper"] = sc.is_goalkeeper(pos)
        teams.append({
            "abbr": team,
            "side": side,
            "opponent": opp,
            "players": players,
        })
    return {"home_team": home_n, "away_team": away_n, "teams": teams}


# ---------------------------------------------------------------------------
# Roster: every player on both teams of an NFL game
# ---------------------------------------------------------------------------
def _nfl_recent_usage(engine, team_names):
    """{player_id: recent scrimmage/passing yards} this season, for ranking a
    roster so featured players sit at the top. Empty when logs aren't loaded."""
    nc = engine.nc
    try:
        res = (
            nc.supabase.table(nc.LOGS_TABLE)
            .select("player_id,game_date," + ",".join(NFL_ROSTER_STAT_COLUMNS))
            .in_("team", list(team_names))
            .eq("season", nc.current_season())
            .order("game_date", desc=True)
            .execute()
        )
    except Exception:  # noqa: BLE001 - logs table missing => no ranking
        return {}
    recent = {}
    counts = {}
    for r in res.data or []:
        pid = r.get("player_id")
        if pid is None or counts.get(pid, 0) >= 3:   # last 3 games per player
            continue
        counts[pid] = counts.get(pid, 0) + 1
        yards = sum((r.get(c) or 0) for c in NFL_ROSTER_STAT_COLUMNS)
        recent[pid] = recent.get(pid, 0.0) + yards
    return {pid: v / counts[pid] for pid, v in recent.items()}


def nfl_roster(engine, home: str, away: str) -> dict:
    """Both teams' rosters for a matchup -- the NFL twin of nba_roster().
    Ordered by recent usage (featured players first) where game logs exist,
    otherwise offense -> defense -> specialists from the directory. Each player
    carries the opponent + which side he's on so tapping projects the right
    matchup."""
    base = engine.nc.roster_for_game(home, away)
    usage = _nfl_recent_usage(engine, {base["home_team"], base["away_team"]})
    for t in base["teams"]:
        for p in t["players"]:
            p["opponent"] = t["opponent"]
            p["location"] = t["side"]
            p["recent_yds"] = round(usage.get(p["player_id"], 0.0), 1)
        if usage:
            # Featured players (by recent yards) first; keep directory order as
            # the tiebreaker so position grouping still reads sensibly.
            t["players"].sort(key=lambda p: p.get("recent_yds") or 0, reverse=True)
    return base


# ---------------------------------------------------------------------------
# Multi-stat projection for one player ("what he's projected for")
# ---------------------------------------------------------------------------
def player_projections(engine, player: str, stats=None, opponent: str = None,
                       location: str = None, game_type: str = "auto",
                       sport: str = "nba") -> dict:
    """Project several stats for one player in a single response.

    No line is graded here -- we return the projection and its spread (sigma)
    for each stat so the frontend can show the numbers and compute an instant
    over/under read for any line the user types (same normal-CDF math the
    engine uses), and pre-fill a suggested line per stat."""
    if not stats and sport == "soccer":
        defaults = _soccer_default_stats(engine, player)
    elif not stats and sport == "nfl":
        defaults = _nfl_default_stats(engine, player)
    elif sport == "soccer":
        defaults = SOCCER_DEFAULT_STATS
    elif sport == "nfl":
        defaults = NFL_DEFAULT_STATS
    else:
        defaults = DEFAULT_STATS
    stats = [s for s in (stats or defaults) if s in engine.STAT_DEFS]

    meta = {}
    projections = []
    for stat in stats:
        try:
            r = _project(engine, sport, player, stat, None,
                         opponent, location, game_type)
        except Exception:  # noqa: BLE001 - skip a stat we can't project, keep the rest
            continue
        if not meta:
            meta = {k: r.get(k) for k in
                    ("player_name", "team", "position", "opponent", "home_away")}
        entry = {
            "stat": stat,
            "projection": r["projection"],
            "sigma": r["sigma"],
            "l5": r.get("l5"),
            "l10": r.get("l10"),
            "season_avg": r.get("season_avg") or r.get("avg"),
            "suggested_line": _suggest_line(r["projection"]),
        }
        # Clean-sheet read rides along with goals conceded so the UI can show
        # the shutout market without a second call.
        if r.get("p_clean_sheet") is not None:
            entry["p_clean_sheet"] = r["p_clean_sheet"]
        projections.append(entry)

    if not projections:
        raise LookupError(f"Couldn't project any stats for '{player}'.")
    return {**meta, "projections": projections}


# ---------------------------------------------------------------------------
# Batch grading: a hand-built slip of props -> per-leg grade + parlay read
# ---------------------------------------------------------------------------
def _grade_one(engine, prop: dict, sport: str) -> dict:
    """Grade a single typed prop, mirroring the fields a scanned slip leg
    carries so the frontend can reuse the same card."""
    player = (prop.get("player") or "").strip()
    stat = (prop.get("stat") or "").strip().lower()
    line = prop.get("line")
    side = (prop.get("side") or "").upper() or None

    label = f"{player} — {side or ''} {line if line is not None else ''} {stat}".strip()
    leg = {"player_name": player, "stat": stat, "line": line, "side": side,
           "label": label, "hit_probability": None}

    if stat not in engine.STAT_DEFS:
        leg["error"] = f"Unknown stat '{stat}'."
        return leg
    try:
        r = _project(engine, sport, player, stat, line,
                     prop.get("opponent"), prop.get("location"),
                     prop.get("game_type", "auto"))
    except (LookupError, ValueError) as exc:
        leg["error"] = str(exc)
        return leg
    except Exception as exc:  # noqa: BLE001 - degrade this leg, keep the slip
        leg["error"] = str(exc)
        return leg

    # If the caller didn't pick a side, follow the model's recommendation so the
    # parlay number answers "what if I bet the model on all of these?".
    rec = r.get("recommendation")
    if side is None:
        side = rec
    if side == "OVER":
        hit = r.get("p_over")
    elif side == "UNDER":
        hit = r.get("p_under")
    else:
        hit = r.get("confidence")

    leg.update({
        "side": side,
        "label": f"{r.get('player_name', player)} — {side or ''} "
                 f"{line if line is not None else ''} {stat}".strip(),
        "player_name": r.get("player_name", player),
        "team": r.get("team"),
        "position": r.get("position"),
        "opponent": r.get("opponent"),
        "home_away": r.get("home_away"),
        "projection": r.get("projection"),
        "sigma": r.get("sigma"),
        "p_over": r.get("p_over"),
        "p_under": r.get("p_under"),
        "recommendation": rec,
        "confidence_label": r.get("confidence_label"),
        "hit_probability": round(hit, 4) if hit is not None else None,
        "factors": r.get("factors", []),
        "injury": r.get("injury"),
        "note": r.get("note"),
    })
    return leg


def _combine(legs: list) -> dict:
    """Parlay-level read over the graded legs, same shape/assumptions as
    slip_analysis._summarize so the typed and scanned slips agree."""
    graded = [l for l in legs if l.get("hit_probability") is not None]
    summary = {
        "leg_count": len(legs),
        "graded_count": len(graded),
        "combined_probability": None,
        "fair_decimal_odds": None,
        "weakest_leg": None,
        "worry_legs": [],
        "note": None,
    }
    if not graded:
        summary["note"] = "Couldn't grade any of these props."
        return summary

    combined = 1.0
    for l in graded:
        combined *= l["hit_probability"]
    summary["combined_probability"] = round(combined, 4)
    if combined > 0:
        summary["fair_decimal_odds"] = round(1.0 / combined, 2)

    weakest = min(graded, key=lambda l: l["hit_probability"])
    summary["weakest_leg"] = weakest["label"]
    worry = sorted(
        [l for l in graded if l["hit_probability"] < 0.55],
        key=lambda l: l["hit_probability"],
    )
    summary["worry_legs"] = [
        {"label": l["label"], "hit_probability": l["hit_probability"]}
        for l in worry
    ]

    notes = []
    if len(legs) > len(graded):
        notes.append(f"{len(legs) - len(graded)} prop(s) couldn't be graded and "
                     f"are excluded from the combined number.")
    if len(graded) > 1:
        notes.append("Combined assumes the legs are independent; same-game legs "
                     "are correlated, so treat it as a rough estimate.")
    summary["note"] = " ".join(notes) or None
    return summary


def grade_batch(engine, props: list, sport: str = "nba") -> dict:
    """Grade every prop in the list and score them together as a parlay."""
    legs = [_grade_one(engine, p, sport) for p in props]
    return {"legs": legs, "combined": _combine(legs)}

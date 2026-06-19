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
    pre-fill a sensible line the user can tweak instead of leaving it blank."""
    line = round(projection * 2) / 2          # nearest half-point
    if line == int(line):                     # whole number -> drop to the .5 below
        line -= 0.5
    return round(line, 1)


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
# Multi-stat projection for one player ("what he's projected for")
# ---------------------------------------------------------------------------
def player_projections(engine, player: str, stats=None, opponent: str = None,
                       location: str = None, game_type: str = "auto") -> dict:
    """Project several stats for one player in a single response.

    No line is graded here -- we return the projection and its spread (sigma)
    for each stat so the frontend can show the numbers and compute an instant
    over/under read for any line the user types (same normal-CDF math the
    engine uses), and pre-fill a suggested line per stat."""
    stats = [s for s in (stats or DEFAULT_STATS) if s in engine.STAT_DEFS]
    home_away = _home_away(location)
    season_type = _season_type(game_type)

    meta = {}
    projections = []
    for stat in stats:
        try:
            r = engine.project_player(
                player_name=player, stat=stat, line=None,
                opponent=opponent or None, home_away=home_away,
                season_type=season_type,
            )
        except Exception:  # noqa: BLE001 - skip a stat we can't project, keep the rest
            continue
        if not meta:
            meta = {k: r.get(k) for k in
                    ("player_name", "team", "position", "opponent", "home_away")}
        projections.append({
            "stat": stat,
            "projection": r["projection"],
            "sigma": r["sigma"],
            "l5": r.get("l5"),
            "l10": r.get("l10"),
            "season_avg": r.get("season_avg"),
            "suggested_line": _suggest_line(r["projection"]),
        })

    if not projections:
        raise LookupError(f"Couldn't project any stats for '{player}'.")
    return {**meta, "projections": projections}


# ---------------------------------------------------------------------------
# Batch grading: a hand-built slip of props -> per-leg grade + parlay read
# ---------------------------------------------------------------------------
def _grade_one(engine, prop: dict) -> dict:
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
        r = engine.project_player(
            player_name=player, stat=stat, line=line,
            opponent=(prop.get("opponent") or None),
            home_away=_home_away(prop.get("location")),
            season_type=_season_type(prop.get("game_type", "auto")),
        )
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


def grade_batch(engine, props: list) -> dict:
    """Grade every prop in the list and score them together as a parlay."""
    legs = [_grade_one(engine, p) for p in props]
    return {"legs": legs, "combined": _combine(legs)}

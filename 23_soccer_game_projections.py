"""
Project a SOCCER MATCH OUTCOME: win / draw / win probabilities + score.

    # Auto-detect from the schedule (order doesn't matter):
    python 23_soccer_game_projections.py --home "Mexico" --away "South Africa"

    # A team's next match:
    python 23_soccer_game_projections.py --team "United States"

The soccer sibling of 14_game_projections.py, built for the 2026 World Cup.
Soccer needs a three-way market (the draw is a bet), so instead of a margin
model this uses the classic Poisson goal model used by football betting
models everywhere:

    1. Each side gets an expected-goals rate (lambda) from soccer_common's
       Elo + recent-form blend (see expected_goals()).
    2. A Poisson scoreline grid turns the two lambdas into P(home win),
       P(draw), P(away win), the most likely scorelines, over/under totals
       and both-teams-to-score.

The Elo ratings start from the researched snapshot in
soccer_team_priors.json and update with every completed result the nightly
refresh ingests -- so the model adapts as the tournament goes on. The same
priors file supplies the scouting notes (coach, style, attack/defense,
key players) that the factor cards quote, matching the NBA UX.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import argparse

import soccer_common as sc


MAX_GOALS = 10           # Poisson grid bound (P(>10 goals each) ~ 0)
DEFAULT_TOTAL_LINE = 2.5


# ---------------------------------------------------------------------------
# Poisson scoreline grid
# ---------------------------------------------------------------------------
def score_grid(lambda_home: float, lambda_away: float):
    """Joint P(home=i, away=j) for i,j in [0, MAX_GOALS]."""
    ph = [sc.poisson_pmf(i, lambda_home) for i in range(MAX_GOALS + 1)]
    pa = [sc.poisson_pmf(j, lambda_away) for j in range(MAX_GOALS + 1)]
    return [[ph[i] * pa[j] for j in range(MAX_GOALS + 1)]
            for i in range(MAX_GOALS + 1)]


def grid_markets(grid, total_line=DEFAULT_TOTAL_LINE):
    """Outcome, totals and BTTS probabilities + most likely scorelines."""
    p_home = p_draw = p_away = p_over = p_btts = 0.0
    scores = []
    for i, row in enumerate(grid):
        for j, p in enumerate(row):
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
            if i + j > total_line:
                p_over += p
            if i > 0 and j > 0:
                p_btts += p
            scores.append((p, i, j))
    scores.sort(reverse=True)
    top_scores = [{"score": f"{i}-{j}", "p": round(p, 4)}
                  for p, i, j in scores[:5]]
    # Normalize the tiny truncation residue so the three-way sums to 1.
    total = p_home + p_draw + p_away
    return {
        "p_home": p_home / total,
        "p_draw": p_draw / total,
        "p_away": p_away / total,
        "p_over": p_over,
        "p_under": 1.0 - p_over,
        "p_btts": p_btts,
        "top_scores": top_scores,
    }


def confidence_label(confidence: float) -> str:
    """Three-way market: a coin starts at ~33%, so thresholds sit lower than
    the NBA's two-way labels."""
    if confidence >= 0.55:
        return "STRONG"
    if confidence >= 0.45:
        return "LEAN"
    return "PASS (too close to call)"


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------
def upcoming_games(days: int = 10) -> list:
    """Upcoming matches in the next `days` days, soonest first."""
    from datetime import date, timedelta
    today = date.today().isoformat()
    horizon = (date.today() + timedelta(days=days)).isoformat()
    rows = sc.fetch_all(
        sc.SCHEDULE_TABLE,
        "match_id,match_date,match_time,competition,home_team,away_team",
        filters=[("eq", "status", "upcoming"),
                 ("gte", "match_date", today),
                 ("lte", "match_date", horizon)],
        order_col="match_date",
    )
    # World Cup matches first within each day -- that's what this is for.
    rows.sort(key=lambda g: (g.get("match_date") or "",
                             0 if sc.is_world_cup(g.get("competition")) else 1,
                             g.get("match_time") or ""))
    return rows


def find_fixture(schedule_rows, home, away):
    """The next upcoming meeting between two teams (orientation corrected)."""
    for g in schedule_rows:
        if g.get("status") != "upcoming":
            continue
        teams = {sc.normalize_team(g.get("home_team")),
                 sc.normalize_team(g.get("away_team"))}
        if teams == {sc.normalize_team(home), sc.normalize_team(away)}:
            return g
    return None


# ---------------------------------------------------------------------------
# Factor cards
# ---------------------------------------------------------------------------
def _form_string(form):
    """Last-5 W/D/L string, most recent last, e.g. 'W W D L W'."""
    out = []
    for m in form[-5:]:
        out.append("W" if m["won"] else ("D" if m["drawn"] else "L"))
    return " ".join(out) if out else None


def build_factors(result, xg, home_prof, away_prof):
    home, away = result["home_team"], result["away_team"]
    rnd = lambda v, n=2: round(v, n) if isinstance(v, (int, float)) else v
    factors = []

    # 1) Team strength (Elo) -- the anchor.
    gap = xg["elo_home"] + xg["elo_home_bonus"] - xg["elo_away"]
    tier_note = ""
    if home_prof.get("outlook") or away_prof.get("outlook"):
        tier_note = (f" Scouting: {home_prof.get('outlook') or ''} "
                     f"{away_prof.get('outlook') or ''}").rstrip()
    factors.append({
        "title": "Team strength (Elo)",
        "value": f"{home} {xg['elo_home']} vs {away} {xg['elo_away']}",
        "detail": (
            f"Elo ratings from researched pre-tournament ratings, updated by "
            f"every result since. A {abs(gap):.0f}-point edge "
            f"{'to ' + (home if gap >= 0 else away)} implies "
            f"{xg['elo_win_expectancy'] * 100:.0f}% win expectancy for {home} "
            f"before the draw is split out.{tier_note}"
        ),
    })

    # 2) Attack: recent scoring + scouting.
    if xg["home_gf"] is not None or xg["away_gf"] is not None:
        parts = []
        if xg["home_gf"] is not None:
            parts.append(f"{home} scores {rnd(xg['home_gf'])}/match "
                         f"(last {xg['home_form_n']})")
        if xg["away_gf"] is not None:
            parts.append(f"{away} scores {rnd(xg['away_gf'])}/match "
                         f"(last {xg['away_form_n']})")
        detail = " · ".join(filter(None, [home_prof.get("attack"),
                                          away_prof.get("attack")]))
        factors.append({
            "title": "Attack",
            "value": "; ".join(parts),
            "detail": detail or "Goals scored per match in recent internationals.",
        })

    # 3) Defense: recent concessions + scouting.
    if xg["home_ga"] is not None or xg["away_ga"] is not None:
        parts = []
        if xg["home_ga"] is not None:
            parts.append(f"{home} concedes {rnd(xg['home_ga'])}/match")
        if xg["away_ga"] is not None:
            parts.append(f"{away} concedes {rnd(xg['away_ga'])}/match")
        detail = " · ".join(filter(None, [home_prof.get("defense"),
                                          away_prof.get("defense")]))
        factors.append({
            "title": "Defense",
            "value": "; ".join(parts),
            "detail": detail or "Goals conceded per match in recent internationals.",
        })

    # 4) Recent form (W/D/L strings).
    hf, af = _form_string(xg["home_form"]), _form_string(xg["away_form"])
    if hf or af:
        form_detail = " · ".join(filter(None, [home_prof.get("form"),
                                               away_prof.get("form")]))
        factors.append({
            "title": "Recent form",
            "value": f"{home}: {hf or 'no data'}  |  {away}: {af or 'no data'}",
            "detail": form_detail or "Last five internationals, oldest first.",
        })

    # 5) Coaches & style (pure scouting -- soccer squads only gather a few
    #    times a year, so the manager's system matters more than club form).
    coach_bits = []
    for prof, name in ((home_prof, home), (away_prof, away)):
        if prof.get("coach"):
            style = f" Style: {prof['style']}" if prof.get("style") else ""
            coach_bits.append(f"{name}: {prof['coach']}{style}")
    if coach_bits:
        factors.append({
            "title": "Coaches & style",
            "value": "How the two benches set up",
            "detail": " — ".join(coach_bits),
        })

    # 6) Key players to watch.
    kp_bits = []
    for prof, name in ((home_prof, home), (away_prof, away)):
        kps = prof.get("key_players") or []
        if kps:
            kp_bits.append(f"{name}: {', '.join(kps[:4])}")
    if kp_bits:
        factors.append({
            "title": "Key players",
            "value": "Names that decide this match",
            "detail": " — ".join(kp_bits),
        })

    # 7) Host edge / venue.
    if xg["elo_home_bonus"]:
        why = ("World Cup host playing at home" if xg["world_cup"]
               else "home advantage")
        factors.append({
            "title": "Home edge",
            "value": f"{home} +{xg['elo_home_bonus']} Elo ({why})",
            "detail": "Hosts historically over-perform at World Cups (crowds, "
                      "travel, familiarity). Neutral-venue matches get no bonus.",
        })

    # 8) Group context.
    if xg["world_cup"] and (home_prof.get("group") or away_prof.get("group")):
        factors.append({
            "title": "Group context",
            "value": f"Group {home_prof.get('group') or away_prof.get('group')}",
            "detail": "Group-stage matches can end level -- the draw is a real "
                      "outcome (no extra time until the knockouts), which is why "
                      "three probabilities are shown.",
        })

    # 9) Method.
    factors.append({
        "title": "Projection method",
        "value": (f"Poisson goal model → {home} {result['projected_home_goals']}"
                  f" - {result['projected_away_goals']} {away}"),
        "detail": (
            f"Recent scoring/concession rates (vs the international average "
            f"of {rnd(xg['league_avg_goals'])}/team) set how open the match "
            f"should be; the Elo gap splits those goals between the sides. "
            f"A Poisson scoreline grid converts the two expected-goals "
            f"numbers into the win/draw/win, totals and scoreline "
            f"probabilities, so every number shown agrees with every other."
        ),
    })
    return factors


# ---------------------------------------------------------------------------
# The match projection engine
# ---------------------------------------------------------------------------
def project_soccer_game(home: str, away: str, match_date=None, match_id=None,
                        total_line: float = DEFAULT_TOTAL_LINE) -> dict:
    """Project `home` vs `away`: 1X2 probabilities, score, totals, factors.

    The single entry point for the API/CLI. If the fixture exists in
    soccer_schedule the stored orientation/date/competition are trusted.
    """
    home, away = sc.normalize_team(home), sc.normalize_team(away)
    if not home or not away or home == away:
        raise ValueError("Give two different team names.")

    schedule = sc.fetch_schedule_rows()

    competition = "FIFA World Cup"
    fixture = find_fixture(schedule, home, away)
    if fixture:
        match_date = match_date or fixture.get("match_date")
        match_id = match_id or fixture.get("match_id")
        competition = fixture.get("competition") or competition
        home = sc.normalize_team(fixture.get("home_team"))
        away = sc.normalize_team(fixture.get("away_team"))

    xg = sc.expected_goals(home, away, schedule_rows=schedule,
                           competition=competition)
    grid = score_grid(xg["lambda_home"], xg["lambda_away"])
    markets = grid_markets(grid, total_line=total_line)

    p_home, p_draw, p_away = markets["p_home"], markets["p_draw"], markets["p_away"]
    outcomes = [(p_home, home), (p_draw, "Draw"), (p_away, away)]
    confidence, predicted = max(outcomes)

    home_prof, away_prof = sc.team_profile(home), sc.team_profile(away)

    result = {
        "method": "poisson_elo",
        "home_team": home,
        "away_team": away,
        "match_date": match_date,
        "match_id": match_id,
        "competition": competition,
        "group": home_prof.get("group") if sc.is_world_cup(competition) else None,
        "p_home_win": round(p_home, 4),
        "p_draw": round(p_draw, 4),
        "p_away_win": round(p_away, 4),
        "predicted_outcome": predicted,
        "confidence": round(confidence, 4),
        "confidence_label": confidence_label(confidence),
        "projected_home_goals": round(xg["lambda_home"], 2),
        "projected_away_goals": round(xg["lambda_away"], 2),
        "projected_total": round(xg["lambda_home"] + xg["lambda_away"], 2),
        "total_line": total_line,
        "p_over": round(markets["p_over"], 4),
        "p_under": round(markets["p_under"], 4),
        "p_btts": round(markets["p_btts"], 4),
        "top_scores": markets["top_scores"],
        "elo_home": xg["elo_home"],
        "elo_away": xg["elo_away"],
        # Double-chance markets, derived from the same grid:
        "p_home_or_draw": round(p_home + p_draw, 4),
        "p_away_or_draw": round(p_away + p_draw, 4),
    }
    result["factors"] = build_factors(result, xg, home_prof, away_prof)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def format_report(r: dict) -> str:
    lines = []
    lines.append("=" * 64)
    head = f"{r['home_team']} vs {r['away_team']}"
    if r.get("match_date"):
        head += f"   {r['match_date']}"
    head += f"  ({r['competition']}"
    head += f", Group {r['group']})" if r.get("group") else ")"
    lines.append(head)
    lines.append("-" * 64)
    lines.append(
        f"  {r['home_team']}: {r['p_home_win'] * 100:.1f}%   "
        f"DRAW: {r['p_draw'] * 100:.1f}%   "
        f"{r['away_team']}: {r['p_away_win'] * 100:.1f}%"
    )
    lines.append(
        f"  >>> {r['predicted_outcome']}"
        + (" wins" if r["predicted_outcome"] != "Draw" else "")
        + f"  ({r['confidence'] * 100:.1f}% — {r['confidence_label']})"
    )
    lines.append(
        f"  GOALS: {r['home_team']} {r['projected_home_goals']} - "
        f"{r['projected_away_goals']} {r['away_team']}   "
        f"(total {r['projected_total']})"
    )
    lines.append(
        f"  O/U {r['total_line']}: over {r['p_over'] * 100:.1f}% / "
        f"under {r['p_under'] * 100:.1f}%   BTTS: {r['p_btts'] * 100:.1f}%"
    )
    likely = ", ".join(f"{s['score']} ({s['p'] * 100:.0f}%)"
                       for s in r.get("top_scores", [])[:3])
    lines.append(f"  LIKELY SCORES: {likely}")
    lines.append("-" * 64)
    lines.append("  Why:")
    for f in r.get("factors", []):
        lines.append(f"   • {f['title']}: {f['value']}")
        lines.append(f"       {f['detail']}")
    lines.append("=" * 64)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Project a soccer match outcome.")
    parser.add_argument("--team", help="Team name; auto-detect its next match.")
    parser.add_argument("--home", help="Home team (with --away).")
    parser.add_argument("--away", help="Away team (with --home).")
    parser.add_argument("--date", default=None, help="Match date YYYY-MM-DD.")
    parser.add_argument("--total-line", type=float, default=DEFAULT_TOTAL_LINE,
                        help="Goals total line to grade (default 2.5).")
    args = parser.parse_args()

    if args.team:
        team = sc.normalize_team(args.team)
        nexts = [g for g in upcoming_games(days=40)
                 if team in (sc.normalize_team(g.get("home_team")),
                             sc.normalize_team(g.get("away_team")))]
        if not nexts:
            raise SystemExit(f"No upcoming match found for {team}.")
        g = nexts[0]
        home, away, gdate = g["home_team"], g["away_team"], g["match_date"]
    elif args.home and args.away:
        home, away, gdate = args.home, args.away, args.date
    else:
        raise SystemExit("Give either --team, or --home and --away.")

    try:
        result = project_soccer_game(home, away, match_date=gdate,
                                     total_line=args.total_line)
    except (LookupError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")
    print(format_report(result))


if __name__ == "__main__":
    main()

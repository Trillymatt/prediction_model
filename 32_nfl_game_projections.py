"""
Project an NFL GAME OUTCOME: winner probability + projected final score.

    # Auto-detect the next scheduled game for a team:
    python 32_nfl_game_projections.py --team "Kansas City Chiefs"

    # Or name the matchup explicitly (home team first):
    python 32_nfl_game_projections.py --home KC --away BUF --date 2026-09-10

The NFL sibling of 14_game_projections.py (NBA, trained model) and
23_soccer_game_projections.py (soccer, Poisson). NFL doesn't have years of
ingested player box scores to train on yet, so this stage uses the same
Elo-plus-stats approach as the soccer engine instead: nfl_common.py's
project_matchup() blends

    1. Elo ratings (whole-season team strength, built from every completed
       nfl_schedule result -- see nfl_common.elo_ratings)
    2. a recency-weighted points model (current scoring form -- each side's
       attack rate vs. the other's defense, relative to the league average)
    3. a small rest adjustment (bye week / short week)

into a single projected margin + total. Everything else here (win
probability, score, confidence) is DERIVED from that margin/total, so
nothing can contradict:

    P(home win) = Phi(projected_margin / sigma_margin)
    home score  = (total + margin) / 2,   away score = (total - margin) / 2

Output matches 14_game_projections.py's shape field-for-field so the
frontend's existing GameResultCard renders it with no changes.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import argparse
from datetime import date, timedelta

import nfl_common as nc


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------
def find_fixture(schedule_rows, home, away):
    """The next upcoming meeting between two teams (orientation corrected)."""
    home_n, away_n = nc.normalize_team(home), nc.normalize_team(away)
    upcoming = [g for g in schedule_rows if g.get("status") == "upcoming"]
    upcoming.sort(key=lambda g: (g.get("game_date") or "", g.get("game_time") or ""))
    for g in upcoming:
        teams = {nc.normalize_team(g.get("home_team")),
                 nc.normalize_team(g.get("away_team"))}
        if teams == {home_n, away_n}:
            return g
    return None


def season_series(schedule_rows: list, home: str, away: str, season=None) -> dict:
    """This season's completed meetings between the two teams (rare in the
    NFL outside a division rematch or the playoffs, but worth showing when
    it exists). Without a known season (e.g. a fixture not on the schedule
    yet), falls back to all-time head-to-head."""
    home_n, away_n = nc.normalize_team(home), nc.normalize_team(away)
    home_wins = away_wins = 0
    margins = []
    for g in schedule_rows:
        if g.get("status") != "completed":
            continue
        teams = {nc.normalize_team(g.get("home_team")),
                 nc.normalize_team(g.get("away_team"))}
        if teams != {home_n, away_n}:
            continue
        if season and g.get("season") != season:
            continue
        hs, as_ = g.get("home_score"), g.get("away_score")
        if hs is None or as_ is None:
            continue
        margin = hs - as_ if nc.normalize_team(g.get("home_team")) == home_n else as_ - hs
        margins.append(margin)
        if margin > 0:
            home_wins += 1
        else:
            away_wins += 1
    return {
        "games": len(margins),
        "home_wins": home_wins,
        "away_wins": away_wins,
        "avg_margin": round(sum(margins) / len(margins), 1) if margins else None,
    }


def confidence_label(confidence: float) -> str:
    """Same buckets as the NBA game engine, so the UI language matches."""
    if confidence >= 0.65:
        return "STRONG"
    if confidence >= 0.57:
        return "LEAN"
    return "PASS (too close to call)"


# ---------------------------------------------------------------------------
# Factor cards
# ---------------------------------------------------------------------------
def build_factors(result: dict, xg: dict, series: dict) -> list:
    home, away = result["home_team"], result["away_team"]
    rnd = lambda v, n=1: round(v, n) if isinstance(v, (int, float)) else v
    factors = []

    # 1) Team strength (Elo) -- the anchor.
    gap = xg["elo_home"] + nc.ELO_HOME_BONUS - xg["elo_away"]
    factors.append({
        "title": "Team strength (Elo)",
        "value": f"{home} {xg['elo_home']} vs {away} {xg['elo_away']}",
        "detail": (
            f"Elo ratings built from every completed result on file (no "
            f"preseason priors yet -- see NFL_SETUP.md on backfilling prior "
            f"seasons). A {abs(gap):.0f}-point edge "
            f"{'to ' + (home if gap >= 0 else away)} implies "
            f"{xg['elo_win_expectancy'] * 100:.0f}% win expectancy for {home} "
            f"before this season's form is factored in."
        ),
    })

    # 2) Scoring form.
    if xg["home_pf"] is not None or xg["away_pf"] is not None:
        parts = []
        if xg["home_pf"] is not None:
            parts.append(f"{home} scores {rnd(xg['home_pf'])}, allows {rnd(xg['home_pa'])} "
                         f"(last {xg['home_games_n']})")
        if xg["away_pf"] is not None:
            parts.append(f"{away} scores {rnd(xg['away_pf'])}, allows {rnd(xg['away_pa'])} "
                         f"(last {xg['away_games_n']})")
        factors.append({
            "title": "Scoring form",
            "value": "; ".join(parts) if parts else "No games logged yet",
            "detail": (
                f"Points per game, rolling window (carries over across a "
                f"season boundary so Week 1 isn't blind), vs. the league "
                f"average of {rnd(xg['league_avg_points'])}/team."
            ),
        })

    # 3) Home field.
    factors.append({
        "title": "Home field",
        "value": f"{home} at home",
        "detail": (
            f"Worth +{nc.ELO_HOME_BONUS} Elo (~{nc.ELO_HOME_BONUS / nc.ELO_POINTS_PER_ELO:.1f} "
            f"pts) plus a direct +{nc.HOME_FIELD_POINTS} pt bump on the scoring "
            f"side -- standard NFL home-field edge."
        ),
    })

    # 4) Rest.
    hr, ar = xg.get("home_rest"), xg.get("away_rest")
    if hr is not None or ar is not None:
        bits = []
        if hr is not None:
            note = " (bye)" if hr >= 13 else " (short week)" if hr <= 5 else ""
            bits.append(f"{home} {hr}d rest{note}")
        if ar is not None:
            note = " (bye)" if ar >= 13 else " (short week)" if ar <= 5 else ""
            bits.append(f"{away} {ar}d rest{note}")
        factors.append({
            "title": "Rest",
            "value": ", ".join(bits),
            "detail": "Days since each team's last game. A bye week is a "
                      "small edge; a short week (e.g. Thursday off a Sunday "
                      "game) is a small cost.",
        })

    # 5) Season series.
    if series and series["games"]:
        lead = (f"{home} leads {series['home_wins']}-{series['away_wins']}"
                if series["home_wins"] > series["away_wins"]
                else f"{away} leads {series['away_wins']}-{series['home_wins']}"
                if series["away_wins"] > series["home_wins"]
                else f"split {series['home_wins']}-{series['away_wins']}")
        factors.append({
            "title": "Season series",
            "value": f"{lead} this season",
            "detail": f"Across {series['games']} meeting(s), average margin "
                      f"{series['avg_margin']:+} for {home}. Shown for "
                      f"context -- the model leans on full form, not one "
                      f"head-to-head result.",
        })

    # 6) Data confidence.
    if xg.get("thin_data"):
        factors.append({
            "title": "Limited data",
            "value": f"{home}: {xg['home_games_n']} games logged, "
                      f"{away}: {xg['away_games_n']}",
            "detail": "One or both teams have few completed games on file, "
                      "so this leans more on Elo than current form and the "
                      "spread is widened to reflect the extra uncertainty. "
                      "Backfilling prior seasons (see NFL_SETUP.md) fixes this.",
        })

    # 7) Method.
    factors.append({
        "title": "Projection method",
        "value": (f"Elo + points model → {home} {result['projected_home_score']} - "
                  f"{result['projected_away_score']} {away}"),
        "detail": (
            f"Elo's whole-season strength gap and a recency-weighted points "
            f"model (each side's scoring rate vs. the other's) are blended "
            f"{nc.STATS_ELO_BLEND * 100:.0f}/{(1 - nc.STATS_ELO_BLEND) * 100:.0f} "
            f"into the margin ({result['projected_margin']:+} ± "
            f"{result['sigma_margin']}) and total ({result['projected_total']} "
            f"± {result['sigma_total']}); the win probability and score are "
            f"derived from those, so they always agree. This is a first-pass "
            f"statistical model (no trained ML model yet -- see NFL_SETUP.md)."
        ),
    })
    return factors


# ---------------------------------------------------------------------------
# The game projection engine
# ---------------------------------------------------------------------------
def project_nfl_game(home: str, away: str, game_date=None, game_id=None,
                     season_type: str = None) -> dict:
    """Project `away` @ `home`: win probability, projected score, factors.

    The single entry point for the API/CLI. If the fixture exists in
    nfl_schedule, the stored orientation/date/season_type are trusted.
    """
    home, away = nc.normalize_team(home), nc.normalize_team(away)
    if not home or not away or home == away:
        raise ValueError("Give two different NFL teams.")

    schedule = nc.fetch_schedule_rows()

    fixture = find_fixture(schedule, home, away)
    season = None
    if fixture:
        game_date = game_date or fixture.get("game_date")
        game_id = game_id or fixture.get("game_id")
        season_type = season_type or fixture.get("season_type")
        season = fixture.get("season")
        home = nc.normalize_team(fixture.get("home_team"))
        away = nc.normalize_team(fixture.get("away_team"))

    gdate = nc.parse_date(game_date) or (date.today() + timedelta(days=1))
    xg = nc.project_matchup(home, away, schedule_rows=schedule,
                            game_date=gdate.isoformat())

    margin, total = xg["projected_margin"], xg["projected_total"]
    sigma_margin, sigma_total = xg["sigma_margin"], xg["sigma_total"]
    p_home = nc.normal_cdf(margin / sigma_margin)
    home_score = (total + margin) / 2.0
    away_score = (total - margin) / 2.0

    series = season_series(schedule, home, away, season=season)
    confidence = max(p_home, 1 - p_home)

    result = {
        "method": "elo_stats",
        "home_team": home,
        "away_team": away,
        "game_date": gdate.isoformat(),
        "game_id": game_id,
        "season_type": (season_type or "regular").capitalize(),
        "p_home_win": round(p_home, 4),
        "p_away_win": round(1 - p_home, 4),
        "predicted_winner": home if p_home >= 0.5 else away,
        "confidence": round(confidence, 4),
        "confidence_label": confidence_label(confidence),
        "projected_margin": round(margin, 1),
        "sigma_margin": round(sigma_margin, 1),
        "projected_total": round(total, 1),
        "sigma_total": round(sigma_total, 1),
        "projected_home_score": round(home_score, 1),
        "projected_away_score": round(away_score, 1),
        "season_series": series,
    }
    result["factors"] = build_factors(result, xg, series)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def format_report(r: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append(f"{r['away_team']} @ {r['home_team']}   {r['game_date']}  ({r['season_type']})")
    lines.append("-" * 60)
    lines.append(
        f"  WINNER:      {r['predicted_winner']}  "
        f"({max(r['p_home_win'], r['p_away_win']) * 100:.1f}% — {r['confidence_label']})"
    )
    lines.append(
        f"  SCORE:       {r['home_team']} {r['projected_home_score']} - "
        f"{r['projected_away_score']} {r['away_team']}"
    )
    lines.append(
        f"  MARGIN:      {r['projected_margin']:+} (± {r['sigma_margin']})   "
        f"TOTAL: {r['projected_total']} (± {r['sigma_total']})"
    )
    lines.append("-" * 60)
    lines.append("  Why:")
    for f in r.get("factors", []):
        lines.append(f"   • {f['title']}: {f['value']}")
        lines.append(f"       {f['detail']}")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Project an NFL game outcome.")
    parser.add_argument("--team", help="Team name/abbrev; auto-detect its next game.")
    parser.add_argument("--home", help="Home team (with --away).")
    parser.add_argument("--away", help="Away team (with --home).")
    parser.add_argument("--date", default=None, help="Game date YYYY-MM-DD.")
    args = parser.parse_args()

    if args.team:
        team = nc.normalize_team(args.team)
        nexts = [g for g in nc.upcoming_games(days=60)
                 if team in (nc.normalize_team(g.get("home_team")),
                             nc.normalize_team(g.get("away_team")))]
        if not nexts:
            raise SystemExit(f"No upcoming game found for {team}.")
        g = nexts[0]
        home, away, gdate = g["home_team"], g["away_team"], g["game_date"]
    elif args.home and args.away:
        home, away, gdate = args.home, args.away, args.date
    else:
        raise SystemExit("Give either --team, or --home and --away.")

    try:
        result = project_nfl_game(home, away, game_date=gdate)
    except (LookupError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")
    print(format_report(result))


if __name__ == "__main__":
    main()

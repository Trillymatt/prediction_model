"""
Project a soccer player's stat for their next match and grade a betting line.

    # Auto-detect the next fixture from soccer_schedule, grade a shots line:
    python 22_soccer_projections.py --player "Lionel Messi" --stat shots --line 2.5

    # Force the opponent (e.g. a hypothetical knockout matchup):
    python 22_soccer_projections.py --player "Kylian Mbappe" --stat goals \
        --line 0.5 --opponent "Norway"

The soccer sibling of 09_projections.py, built for the 2026 World Cup. Same
philosophy: you bring the line from your book, the engine projects the stat
from data and tells you honestly how confident the data is.

How it differs from the NBA engine (because soccer is different):
  * Per-90 rates, not per-game averages. National-team players don't play
    together often, so the form blend covers their international appearances
    over the last couple of years (L5 50% / L10 30% / all 20%), normalized
    to 90 minutes and scaled by the minutes they're actually expected to play.
  * Poisson probabilities, not normal. Goals/assists/shots are small counts;
    P(over 0.5 goals) comes from the Poisson tail, which handles 0-and-1
    outcomes far better than a bell curve.
  * Matchup via the team goal model. The player's attacking output is scaled
    by how many goals their TEAM is expected to score against THIS opponent
    (Elo + defensive form, from soccer_common.expected_goals) relative to the
    team's norm -- so a striker facing a bunker defense gets marked down even
    if his own logs look hot.

The factor cards quote the researched scouting notes (opponent's defense,
team's attack, coach/style) from soccer_team_priors.json, same UX as the NBA.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import math
import argparse
import statistics

import soccer_common as sc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEIGHT_L5 = 0.50
WEIGHT_L10 = 0.30
WEIGHT_ALL = 0.20

# How hard the team-vs-opponent goal expectation moves attacking stats.
MATCHUP_DAMPENING = 0.60
MATCHUP_MIN, MATCHUP_MAX = 0.60, 1.50

MIN_MINUTES_FLOOR = 10        # expected minutes never projected below this
POISSON_MAX_MEAN = 10.0       # above this, a normal spread reads better

# Friendly stat name -> log columns summed per match.
STAT_DEFS = {
    "goals": ["goals"],
    "assists": ["assists"],
    "goals_assists": ["goals", "assists"],
    "shots": ["shots"],
    "shots_on_target": ["shots_on_target"],
    "key_passes": ["key_passes"],
    "cards": ["yellow_cards", "red_cards"],
}

# Columns that may not exist yet (see SOCCER_SETUP.md). Probed once at
# import; the stats appear automatically once the columns are added.
OPTIONAL_STAT_DEFS = {
    "passes": ["passes"],
    "tackles": ["tackles"],
    "saves": ["saves"],
    "fouls_committed": ["fouls_committed"],
    "fouls_suffered": ["fouls_suffered"],
}

STAT_NOUNS = {
    "goals": "goals", "assists": "assists", "goals_assists": "goals + assists",
    "shots": "shots", "shots_on_target": "shots on target",
    "key_passes": "key passes", "cards": "cards", "passes": "passes",
    "tackles": "tackles", "saves": "saves",
    "fouls_committed": "fouls committed", "fouls_suffered": "fouls drawn",
}

# Stats that scale with how much the player's team attacks. Cards/saves/fouls
# don't follow team goal expectation, so they stay unscaled.
ATTACKING_STATS = {"goals", "assists", "goals_assists", "shots",
                   "shots_on_target", "key_passes", "passes"}

LOG_COLUMNS = [
    "player_id", "player_name", "match_date", "competition", "season", "team",
    "opponent", "home_away", "minutes_played", "goals", "assists", "shots",
    "shots_on_target", "xg", "xa", "key_passes", "yellow_cards", "red_cards",
]


def _enable_optional_stats():
    """Activate passes/tackles/... if their columns exist in Supabase."""
    optional_cols = sorted({c for cols in OPTIONAL_STAT_DEFS.values() for c in cols})
    active = []
    for col in optional_cols:
        try:
            sc.supabase.table(sc.LOGS_TABLE).select(col).limit(1).execute()
            active.append(col)
        except Exception:  # noqa: BLE001 - column absent => stat stays off
            continue
    for stat, cols in OPTIONAL_STAT_DEFS.items():
        if all(c in active for c in cols):
            STAT_DEFS[stat] = cols
            LOG_COLUMNS.extend(c for c in cols if c not in LOG_COLUMNS)


_enable_optional_stats()


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def search_players(query: str, limit: int = 10) -> list:
    """Autocomplete: soccer_players when present, else distinct from the logs."""
    query = (query or "").strip()
    if len(query) < 2:
        return []
    try:
        res = (
            sc.supabase.table(sc.PLAYERS_TABLE)
            .select("player_id,player_name,team,position")
            .ilike("player_name", f"%{query}%")
            .order("player_name")
            .limit(limit)
            .execute()
        )
        if res.data:
            return res.data
    except Exception:  # noqa: BLE001 - table missing => fall back to logs
        pass

    res = (
        sc.supabase.table(sc.LOGS_TABLE)
        .select("player_id,player_name,team,match_date")
        .ilike("player_name", f"%{query}%")
        .order("match_date", desc=True)
        .limit(200)
        .execute()
    )
    seen, out = set(), []
    for r in res.data or []:
        pid = r.get("player_id")
        if pid in seen:
            continue
        seen.add(pid)
        out.append({"player_id": pid, "player_name": r.get("player_name"),
                    "team": r.get("team"), "position": None})
        if len(out) >= limit:
            break
    return out


def find_player(name: str) -> dict:
    """Resolve a (partial) name to one player; raise with candidates if not."""
    rows = search_players(name, limit=15)
    if not rows:
        raise LookupError(f"No soccer player matching '{name}'.")
    exact = [r for r in rows if (r.get("player_name") or "").lower() == name.lower()]
    if exact:
        return exact[0]
    if len(rows) > 1:
        names = ", ".join(r.get("player_name", "?") for r in rows[:10])
        raise LookupError(f"'{name}' is ambiguous. Did you mean one of: {names}")
    return rows[0]


def fetch_player_logs(player_id) -> list:
    """All match-log rows for a player, oldest first (paged)."""
    return sc.fetch_all(
        sc.LOGS_TABLE, ",".join(LOG_COLUMNS),
        filters=[("eq", "player_id", player_id)],
        order_col="match_date",
    )


def next_match_for_team(team: str, schedule_rows):
    """(opponent, home_away, match_date, competition) for a team's next match."""
    team = sc.normalize_team(team)
    for g in schedule_rows:
        if g.get("status") != "upcoming":
            continue
        home = sc.normalize_team(g.get("home_team"))
        away = sc.normalize_team(g.get("away_team"))
        if team == home:
            return away, "HOME", g.get("match_date"), g.get("competition")
        if team == away:
            return home, "AWAY", g.get("match_date"), g.get("competition")
    return None, None, None, None


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def match_value(row: dict, columns) -> float:
    return float(sum((row.get(c) or 0) for c in columns))


def per90(rows, columns):
    """Stat per 90 minutes across a window of logs (None if no minutes)."""
    minutes = sum((r.get("minutes_played") or 0) for r in rows)
    if not minutes:
        return None
    total = sum(match_value(r, columns) for r in rows)
    return total * 90.0 / minutes


def blend(parts):
    """Weighted average of the (value, weight) pairs that have a value."""
    parts = [(v, w) for v, w in parts if v is not None]
    total_w = sum(w for _, w in parts)
    return sum(v * w for v, w in parts) / total_w if total_w else None


# ---------------------------------------------------------------------------
# Factor cards
# ---------------------------------------------------------------------------
def build_factors(result, stat, opponent, team, exp_minutes, matchup, xg,
                  opp_profile, team_profile):
    noun = STAT_NOUNS.get(stat, stat)
    rnd = lambda v, n=2: round(v, n) if isinstance(v, (int, float)) else v
    factors = []

    # 1) Recent form.
    factors.append({
        "title": "Recent form",
        "value": f"{result['per90_blend']} {noun} per 90",
        "detail": (
            f"International appearances: last 5 avg {result['l5']}, last 10 avg "
            f"{result['l10']}, overall {result['avg']} per match. Rates are "
            f"normalized to 90 minutes so substitute appearances don't "
            f"understate him."
        ),
    })

    # 2) Expected minutes.
    factors.append({
        "title": "Expected minutes",
        "value": f"~{round(exp_minutes)} minutes",
        "detail": "Based on his recent match minutes for the national team. "
                  "The per-90 rate is scaled to this workload.",
    })

    # 3) Matchup (attacking stats only -- driven by the team goal model).
    if opponent and matchup is not None and stat in ATTACKING_STATS:
        direction = ("lifts" if matchup > 1.001
                     else "trims" if matchup < 0.999 else "doesn't move")
        opp_def = (opp_profile or {}).get("defense")
        detail = (
            f"{team} is expected to score {rnd(xg['team_lambda'])} vs "
            f"{opponent} against a normal output of {rnd(xg['team_norm'])} "
            f"-- that {direction} attacking stats by "
            f"{abs(matchup - 1) * 100:.0f}%. "
        )
        if opp_def:
            detail += f"Scouting on {opponent}'s defense: {opp_def}"
        factors.append({
            "title": "Opponent defense",
            "value": f"{opponent} (Elo {xg['opp_elo']}) — matchup x{matchup:.2f}",
            "detail": detail,
        })
    elif opponent and stat not in ATTACKING_STATS:
        factors.append({
            "title": "Matchup",
            "value": f"vs {opponent}",
            "detail": f"{noun.capitalize()} don't track team goal expectation, "
                      f"so no matchup scaling is applied to this stat.",
        })

    # 4) Team attack context (scouting).
    tp = team_profile or {}
    if tp.get("attack") or tp.get("style"):
        factors.append({
            "title": "Team context",
            "value": f"{team} — how they create chances",
            "detail": " ".join(filter(None, [tp.get("style"), tp.get("attack")])),
        })

    # 5) Tournament context: coach + WC squad caveat.
    if tp.get("coach") or result.get("competition"):
        bits = []
        if tp.get("coach"):
            bits.append(f"Coach: {tp['coach']}")
        bits.append(
            "World Cup caveat: national teammates play together rarely, so "
            "team-level scouting and the coach's system carry extra weight "
            "next to the player's own logs."
        )
        factors.append({
            "title": "Tournament context",
            "value": result.get("competition") or "International",
            "detail": " ".join(bits),
        })

    # 6) Method.
    if result["distribution"] == "poisson":
        factors.append({
            "title": "Projection method",
            "value": f"Poisson → {result['projection']} expected {noun}",
            "detail": (
                "Low-count stats follow a Poisson distribution, so the "
                "over/under probability comes from the Poisson tail at your "
                "line -- the right tool for 0.5/1.5-type soccer props."
            ),
        })
    else:
        factors.append({
            "title": "Projection method",
            "value": f"Normal → {result['projection']} (± {result['sigma']})",
            "detail": "High-volume stat, so a normal spread around the "
                      "projection grades the line.",
        })
    return factors


# ---------------------------------------------------------------------------
# The projection engine
# ---------------------------------------------------------------------------
def project_soccer_player(player_name: str, stat: str, line: float = None,
                          opponent: str = None) -> dict:
    """Project `stat` for a player's next match; grade `line` if given.

    The single entry point for the API/CLI. Mirrors project_player()'s
    output shape (projection, sigma, factors, p_over/p_under/recommendation)
    so the frontend renders both sports identically.
    """
    stat = stat.lower()
    if stat not in STAT_DEFS:
        raise ValueError(
            f"Unknown stat '{stat}'. Choose from: {', '.join(sorted(STAT_DEFS))}"
        )
    columns = STAT_DEFS[stat]

    player = find_player(player_name)
    logs = fetch_player_logs(player["player_id"])
    played = [g for g in logs if g.get("minutes_played")]
    if not played:
        raise LookupError(
            f"No matches with minutes found for {player['player_name']}."
        )

    team = sc.normalize_team(played[-1].get("team") or player.get("team"))

    # --- Form blend (per-90, then scaled to expected minutes) ---------------
    l5_rows, l10_rows = played[-5:], played[-10:]
    per90_l5 = per90(l5_rows, columns)
    per90_l10 = per90(l10_rows, columns)
    per90_all = per90(played, columns)
    per90_blend = blend([(per90_l5, WEIGHT_L5), (per90_l10, WEIGHT_L10),
                         (per90_all, WEIGHT_ALL)])

    minutes_recent = [g.get("minutes_played") or 0 for g in l5_rows]
    exp_minutes = max(MIN_MINUTES_FLOOR,
                      min(90.0, sum(minutes_recent) / len(minutes_recent)))

    projection = (per90_blend or 0.0) * exp_minutes / 90.0

    # --- Matchup: scale attacking output by the team goal model -------------
    schedule = sc.fetch_schedule_rows()
    home_away = next_date = competition = None
    if opponent is None:
        opponent, home_away, next_date, competition = next_match_for_team(
            team, schedule)
    opponent = sc.normalize_team(opponent) if opponent else None

    matchup = None
    xg_info = {"team_lambda": None, "team_norm": None, "opp_elo": None}
    if opponent and team:
        # Orient the fixture: who's nominal home doesn't matter much at a
        # mostly-neutral World Cup, but host edge is handled inside.
        if home_away == "AWAY":
            xg = sc.expected_goals(opponent, team, schedule_rows=schedule,
                                   competition=competition or "FIFA World Cup")
            team_lambda, opp_elo = xg["lambda_away"], xg["elo_home"]
        else:
            xg = sc.expected_goals(team, opponent, schedule_rows=schedule,
                                   competition=competition or "FIFA World Cup")
            team_lambda, opp_elo = xg["lambda_home"], xg["elo_away"]
        team_form = sc.team_recent_form(schedule, team)
        team_norm = (sum(m["scored"] for m in team_form) / len(team_form)
                     if team_form else xg["league_avg_goals"])
        if team_norm:
            raw = team_lambda / team_norm
            matchup = min(MATCHUP_MAX,
                          max(MATCHUP_MIN,
                              1.0 + MATCHUP_DAMPENING * (raw - 1.0)))
            if stat in ATTACKING_STATS:
                projection *= matchup
        xg_info = {"team_lambda": team_lambda, "team_norm": team_norm,
                   "opp_elo": opp_elo}

    # --- Spread + probabilities ----------------------------------------------
    per_match = [match_value(g, columns) for g in played]
    distribution = "poisson" if projection < POISSON_MAX_MEAN else "normal"
    if distribution == "poisson":
        sigma = math.sqrt(projection) if projection > 0 else None
    else:
        recent = per_match[-15:]
        sigma = statistics.pstdev(recent) if len(recent) >= 2 else None

    result = {
        "method": "poisson_per90",
        "distribution": distribution,
        "player_name": player["player_name"],
        "player_id": player["player_id"],
        "team": team,
        "position": player.get("position"),
        "stat": stat,
        "opponent": opponent,
        "home_away": home_away,
        "match_date": next_date,
        "competition": competition,
        "games_used": len(played),
        "l5": round(sum(per_match[-5:]) / max(len(per_match[-5:]), 1), 2),
        "l10": round(sum(per_match[-10:]) / max(len(per_match[-10:]), 1), 2),
        "avg": round(sum(per_match) / len(per_match), 2),
        "per90_blend": round(per90_blend, 2) if per90_blend is not None else None,
        "expected_minutes": round(exp_minutes, 1),
        "matchup_factor": round(matchup, 3) if matchup is not None else None,
        "projection": round(projection, 2),
        "sigma": round(sigma, 2) if sigma is not None else None,
        "line": line,
    }

    opp_profile = sc.team_profile(opponent) if opponent else {}
    team_prof = sc.team_profile(team) if team else {}
    result["factors"] = build_factors(
        result, stat, opponent, team, exp_minutes, matchup, xg_info,
        opp_profile, team_prof,
    )

    # --- Grade the user's line -----------------------------------------------
    if line is not None:
        if distribution == "poisson":
            # Smoothing floor: a player with zero career events still has SOME
            # chance of one tomorrow -- never grade a line at 100%.
            p_over = sc.poisson_p_over(line, max(projection, 0.05))
        elif sigma:
            p_over = sc.normal_cdf((projection - line) / sigma)
        else:
            result["note"] = ("Not enough match history to estimate spread; "
                              "showing projection only.")
            return result
        p_under = 1.0 - p_over
        pick = "OVER" if p_over >= 0.5 else "UNDER"
        confidence = max(p_over, p_under)
        result.update({
            "p_over": round(p_over, 4),
            "p_under": round(p_under, 4),
            "recommendation": pick,
            "confidence": round(confidence, 4),
            "confidence_label": confidence_label(confidence),
        })
    return result


def confidence_label(confidence: float) -> str:
    """Same buckets as the NBA player engine, so the UI language matches."""
    if confidence >= 0.65:
        return "STRONG"
    if confidence >= 0.57:
        return "LEAN"
    return "PASS (too close to call)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def format_report(r: dict) -> str:
    lines = []
    lines.append("=" * 64)
    head = f"{r['player_name']}  ({r.get('team') or '?'})"
    lines.append(head)
    matchup = ""
    if r.get("opponent"):
        loc = {"HOME": "vs", "AWAY": "@"}.get(r.get("home_away"), "vs")
        matchup = f"  {loc} {r['opponent']}"
    lines.append(f"Stat: {r['stat'].upper()}{matchup}"
                 f"  ({r.get('competition') or 'International'})")
    lines.append("-" * 64)
    lines.append(f"  PROJECTION:   {r['projection']}"
                 + (f"   (sigma +/- {r['sigma']})" if r.get("sigma") else ""))
    lines.append("-" * 64)
    lines.append("  Why:")
    for f in r.get("factors", []):
        lines.append(f"   • {f['title']}: {f['value']}")
        lines.append(f"       {f['detail']}")
    if r.get("line") is not None and "recommendation" in r:
        lines.append("-" * 64)
        lines.append(f"  Your line:    {r['line']}")
        lines.append(f"  OVER {r['p_over'] * 100:.1f}%   |   "
                     f"UNDER {r['p_under'] * 100:.1f}%")
        lines.append(f"  >>> {r['recommendation']} {r['line']}  "
                     f"({r['confidence'] * 100:.1f}% - {r['confidence_label']})")
    elif r.get("note"):
        lines.append(f"  Note: {r['note']}")
    lines.append("=" * 64)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Project a soccer player's stat and grade your betting line."
    )
    parser.add_argument("--player", required=True, help="Player name (partial ok).")
    parser.add_argument("--stat", required=True,
                        help=f"One of: {', '.join(sorted(STAT_DEFS))}")
    parser.add_argument("--line", type=float, default=None,
                        help="The over/under line from YOUR book (optional).")
    parser.add_argument("--opponent", default=None,
                        help="Opponent country. Default: auto from soccer_schedule.")
    args = parser.parse_args()

    try:
        result = project_soccer_player(
            args.player, args.stat, line=args.line, opponent=args.opponent,
        )
    except (LookupError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}")
    print(format_report(result))


if __name__ == "__main__":
    main()

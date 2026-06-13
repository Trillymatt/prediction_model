"""
Bet-slip analyzer: screenshot -> per-leg hit probabilities + the worry.

This is the glue between the Gemini vision/prediction helpers (llm_analysis.py)
and our own projection engines. The routing rule the user asked for:

  * If a leg is an NBA or soccer PLAYER PROP and we can find that player in our
    data, grade it with our model (project_player / project_soccer_player) --
    real, data-backed numbers and the same factor cards the rest of the app uses.
  * Anything else -- other sports, team/moneyline/spread/total markets, or a
    player we don't track -- falls back to Gemini Flash with web grounding.

Each leg comes back in one shape regardless of source, so the frontend renders
them identically:

    {
      "label", "player", "sport", "market", "stat", "line", "side",
      "source": "model" | "gemini" | "unavailable",
      "hit_probability": float | None,   # probability THIS bet (its side) hits
      "projection": ...,
      "confidence_label", "worry_level": "low|medium|high",
      "concern": str, "factors": [{title,value,detail}], "error": str | None
    }

The engines are passed in (api.py loads them via importlib because their
filenames start with digits), so this module imports none of them directly.
"""

import llm_analysis


# Slip wording -> our STAT_DEFS keys, as a safety net on top of the LLM's own
# normalization (it is told the keys, but books phrase props many ways).
_NBA_STAT_ALIASES = {
    "pts": "points", "point": "points", "points": "points",
    "reb": "rebounds", "rebound": "rebounds", "rebounds": "rebounds", "trb": "rebounds",
    "ast": "assists", "assist": "assists", "assists": "assists",
    "3pm": "threes", "threes": "threes", "3-pointers made": "threes",
    "three pointers made": "threes", "made threes": "threes", "3pt made": "threes",
    "3pa": "threes_attempted", "steal": "steals", "steals": "steals",
    "block": "blocks", "blocks": "blocks", "to": "turnovers", "turnover": "turnovers",
    "turnovers": "turnovers", "pra": "pra", "pts+reb+ast": "pra",
    "points+rebounds+assists": "pra", "pr": "pr", "pts+reb": "pr",
    "pa": "pa", "pts+ast": "pa", "ra": "ra", "reb+ast": "ra",
    "stocks": "stocks", "stl+blk": "stocks", "steals+blocks": "stocks",
}
_SOCCER_STAT_ALIASES = {
    "goal": "goals", "goals": "goals", "anytime goalscorer": "goals",
    "to score": "goals", "assist": "assists", "assists": "assists",
    "goals+assists": "goals_assists", "goal+assist": "goals_assists",
    "shot": "shots", "shots": "shots", "sot": "shots_on_target",
    "shots on target": "shots_on_target", "shots on goal": "shots_on_target",
    "sog": "shots_on_target", "card": "cards", "cards": "cards",
    "to be carded": "cards", "yellow card": "cards",
}

_NBA_SPORTS = {"nba", "basketball"}
_SOCCER_SPORTS = {"soccer", "football"}


def _norm_stat(sport: str, stat, stat_raw) -> str | None:
    """Map a leg's stat to one of our engine keys, or None if not mappable."""
    aliases = _NBA_STAT_ALIASES if sport in _NBA_SPORTS else _SOCCER_STAT_ALIASES
    for cand in (stat, stat_raw):
        if not cand:
            continue
        key = str(cand).strip().lower()
        if key in aliases:
            return aliases[key]
    # The LLM may already have produced a valid key.
    if stat and str(stat).strip().lower() in set(aliases.values()):
        return str(stat).strip().lower()
    return None


def _worry_from_prob(p: float) -> str:
    if p is None:
        return "unknown"
    if p < 0.45:
        return "high"
    if p < 0.55:
        return "medium"
    return "low"


def _side_hit_probability(r: dict, side: str | None):
    """Pick the probability for the bettor's side from an engine result."""
    p_over, p_under = r.get("p_over"), r.get("p_under")
    if p_over is None or p_under is None:
        return None, None
    s = (side or "").lower()
    if s in ("under", "no"):
        return p_under, "UNDER"
    if s in ("over", "yes", "score", ""):
        return p_over, "OVER"
    return p_over, "OVER"


def _leg_label(leg: dict) -> str:
    who = leg.get("player") or leg.get("team") or "Selection"
    side = (leg.get("side") or "").upper()
    stat = leg.get("stat_raw") or leg.get("stat") or leg.get("market") or ""
    line = leg.get("line")
    line_txt = f" {line}" if line is not None else ""
    tail = f"{side}{line_txt} {stat}".strip()
    return f"{who} — {tail}" if tail else who


def _concern_for_model(r: dict, hit: float, side_label: str) -> tuple[str, str]:
    """(worry_level, concern) for a model-graded leg, factoring injuries in."""
    worry = _worry_from_prob(hit)
    inj = r.get("injury") or {}
    status = (inj.get("status") or "").lower()
    if status and status not in ("", "active", "available"):
        reason = inj.get("reason")
        return "high", f"Injury flag: {inj.get('status')}" + (f" ({reason})" if reason else "")
    if hit is None:
        return "unknown", "Not enough history to grade a probability."
    rec = r.get("recommendation")
    if rec and side_label and rec != side_label:
        return worry, f"Model leans {rec}, against this {side_label} bet."
    if hit < 0.5:
        return "high", f"Model gives the {side_label} side under 50%."
    label = r.get("confidence_label", "")
    if label.startswith("PASS"):
        return "medium", "Too close to call — little edge either way."
    return worry, f"Model agrees with the {side_label} side."


def _grade_with_engine(engine, leg, sport, *, soccer) -> dict | None:
    """Try to grade an NBA/soccer player prop with our model. None = can't."""
    if leg.get("market") != "player_prop" or not leg.get("player"):
        return None
    stat = _norm_stat(sport, leg.get("stat"), leg.get("stat_raw"))
    if stat is None or stat not in engine.STAT_DEFS:
        return None
    line = leg.get("line")
    try:
        engine.find_player(leg["player"])  # routing probe: do we track them?
    except Exception:
        return None
    try:
        if soccer:
            r = engine.project_soccer_player(player_name=leg["player"], stat=stat, line=line)
        else:
            r = engine.project_player(player_name=leg["player"], stat=stat, line=line)
    except Exception as exc:  # noqa: BLE001 - degrade this leg, keep the slip
        return {"_error": str(exc)}

    hit, side_label = _side_hit_probability(r, leg.get("side"))
    worry, concern = _concern_for_model(r, hit, side_label)
    return {
        "source": "model",
        "sport": sport,
        "stat": stat,
        "line": line,
        "side": leg.get("side"),
        "hit_probability": round(hit, 4) if hit is not None else None,
        "projection": r.get("projection"),
        "sigma": r.get("sigma"),
        "confidence_label": r.get("confidence_label"),
        "recommendation": r.get("recommendation"),
        "worry_level": worry,
        "concern": concern,
        "factors": r.get("factors", []),
        "note": r.get("note"),
        "player_name": r.get("player_name"),
        "team": r.get("team"),
        "opponent": r.get("opponent"),
        "home_away": r.get("home_away"),
    }


def _grade_with_llm(leg: dict) -> dict:
    """Fallback prediction via Gemini for non-modeled legs."""
    if not llm_analysis.available():
        return {
            "source": "unavailable",
            "hit_probability": None,
            "worry_level": "unknown",
            "concern": "No model coverage and no Gemini key configured.",
            "factors": [],
            "error": "LLM unavailable (set GEMINI_API_KEY).",
        }
    try:
        out = llm_analysis.predict_leg(leg)
    except llm_analysis.LLMUnavailable as exc:
        return {
            "source": "unavailable",
            "hit_probability": None,
            "worry_level": "unknown",
            "concern": "Could not reach the prediction model for this leg.",
            "factors": [],
            "error": str(exc),
        }
    hit = out.get("hit_probability")
    worry = out.get("worry_level") or _worry_from_prob(hit)
    return {
        "source": "gemini",
        "sport": leg.get("sport"),
        "stat": leg.get("stat") or leg.get("stat_raw"),
        "line": leg.get("line"),
        "side": leg.get("side"),
        "hit_probability": round(hit, 4) if hit is not None else None,
        "projection": out.get("projection"),
        "confidence_label": None,
        "worry_level": worry,
        "concern": out.get("concern", ""),
        "factors": out.get("factors", []),
        "player_name": leg.get("player"),
        "team": leg.get("team"),
    }


def analyze(image_bytes: bytes, mime_type: str, *, nba_engine, soccer_engine) -> dict:
    """Full pipeline: parse the screenshot, grade every leg, score the parlay.

    Raises llm_analysis.LLMUnavailable if the screenshot can't be read at all
    (no key / API failure) -- the caller maps that to a clear HTTP error.
    """
    parsed = llm_analysis.parse_slip(image_bytes, mime_type)
    raw_legs = parsed.get("legs") or []

    results = []
    for leg in raw_legs:
        sport = (leg.get("sport") or "unknown").lower()
        graded = None
        if sport in _NBA_SPORTS and nba_engine is not None:
            graded = _grade_with_engine(nba_engine, leg, "nba", soccer=False)
        elif sport in _SOCCER_SPORTS and soccer_engine is not None:
            graded = _grade_with_engine(soccer_engine, leg, "soccer", soccer=True)

        if graded is not None and "_error" not in graded:
            out = graded
        else:
            # No model coverage (or the model errored on this player) -> Gemini.
            out = _grade_with_llm(leg)
            if graded is not None and "_error" in graded:
                out["note"] = f"Model could not grade this leg ({graded['_error']}); used Gemini."

        out["label"] = _leg_label(leg)
        out.setdefault("player_name", leg.get("player"))
        out.setdefault("team", leg.get("team"))
        out.setdefault("market", leg.get("market"))
        out.setdefault("odds", leg.get("odds"))
        results.append(out)

    return {
        "bet_type": parsed.get("bet_type", "single"),
        "book": parsed.get("book"),
        "legs": results,
        "parlay": _summarize(results),
    }


def _summarize(legs: list) -> dict:
    """Combine the graded legs into a parlay-level read and flag the worry."""
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
        summary["note"] = "Couldn't grade any legs (no model coverage and no LLM key)."
        return summary

    combined = 1.0
    for l in graded:
        combined *= l["hit_probability"]
    summary["combined_probability"] = round(combined, 4)
    if combined > 0:
        summary["fair_decimal_odds"] = round(1.0 / combined, 2)

    weakest = min(graded, key=lambda l: l["hit_probability"])
    summary["weakest_leg"] = weakest["label"]
    # Surface the legs most likely to bust the slip (weakest first).
    worry = sorted(
        [l for l in graded if l["hit_probability"] < 0.55 or l.get("worry_level") == "high"],
        key=lambda l: l["hit_probability"],
    )
    summary["worry_legs"] = [
        {"label": l["label"], "hit_probability": l["hit_probability"],
         "worry_level": l.get("worry_level"), "concern": l.get("concern")}
        for l in worry
    ]
    if len(legs) > len(graded):
        summary["note"] = (
            f"{len(legs) - len(graded)} leg(s) couldn't be graded and are "
            f"excluded from the combined number."
        )
    if len(graded) > 1:
        extra = ("Combined assumes the legs are independent; same-game legs are "
                 "correlated, so treat it as a rough estimate.")
        summary["note"] = f"{summary['note']} {extra}".strip() if summary["note"] else extra
    return summary

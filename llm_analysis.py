"""
Gemini Flash helpers for the bet-slip analyzer.

Two jobs, both done by Gemini Flash (free tier), talked to over the plain REST
API so we add no heavy SDK -- `requests` is already a dependency:

  1. parse_slip(image_bytes, mime)  -> read a screenshot of a bet slip / parlay
     into structured legs (player, sport, stat, line, side, odds). Vision, no
     web access; uses Gemini's JSON mode (responseSchema) for a clean shape.

  2. predict_leg(leg)               -> for a leg our own NBA/soccer models do
     NOT cover (other sports, team/moneyline markets), estimate the hit
     probability. Uses Google Search grounding so the model can pull recent
     form / injury / lineup news instead of guessing from stale memory.

Everything is config-driven via .env so the app still boots (and the NBA/soccer
tools keep working) even when no key is set -- the slip endpoint just reports
that the LLM is unavailable.

    GEMINI_API_KEY   required for the slip analyzer (free at aistudio.google.com)
    GEMINI_MODEL     optional, defaults to gemini-2.5-flash
"""

import os
import json
import base64
import datetime

import requests

from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_ENV_PATH)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_TIMEOUT = 45  # seconds; vision + grounding can be a touch slow


class LLMUnavailable(RuntimeError):
    """Raised when no Gemini key is configured or the API call fails hard."""


def available() -> bool:
    """True when a Gemini key is configured (the slip analyzer needs it)."""
    return bool(GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Low-level Gemini REST call
# ---------------------------------------------------------------------------
def _generate(parts, *, schema=None, grounding=False, temperature=0.2) -> str:
    """POST one turn to Gemini and return the concatenated text of the reply.

    `parts` is the list of request parts (text and/or inline_data). `schema`
    turns on JSON mode (incompatible with grounding, so callers pick one).
    `grounding` adds the Google Search tool. Raises LLMUnavailable on any
    transport/HTTP error so the caller can degrade gracefully.
    """
    if not GEMINI_API_KEY:
        raise LLMUnavailable("GEMINI_API_KEY is not set.")

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": temperature},
    }
    if schema is not None:
        body["generationConfig"]["responseMimeType"] = "application/json"
        body["generationConfig"]["responseSchema"] = schema
    if grounding:
        body["tools"] = [{"google_search": {}}]

    url = f"{_BASE}/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    try:
        resp = requests.post(url, json=body, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise LLMUnavailable(f"Could not reach Gemini: {exc}") from exc

    if resp.status_code != 200:
        # Grounding can 400 on some models/keys; let the caller retry plain.
        raise LLMUnavailable(
            f"Gemini returned {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    try:
        cand = data["candidates"][0]
        chunks = [p.get("text", "") for p in cand["content"]["parts"]]
    except (KeyError, IndexError):
        # Safety blocks / empty completions arrive with no parts.
        raise LLMUnavailable(f"Gemini returned no usable content: {str(data)[:300]}")
    return "".join(chunks).strip()


def _loads_lenient(text: str):
    """Parse JSON that may be wrapped in ``` fences or have leading prose."""
    if not text:
        raise ValueError("empty response")
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
        if t.lstrip().lower().startswith("json"):
            t = t.lstrip()[4:]
    # Trim to the outermost JSON object/array if the model added commentary.
    start = min([i for i in (t.find("{"), t.find("[")) if i != -1], default=-1)
    if start > 0:
        t = t[start:]
    end = max(t.rfind("}"), t.rfind("]"))
    if end != -1:
        t = t[: end + 1]
    return json.loads(t)


# ---------------------------------------------------------------------------
# Job 1: read the screenshot into structured legs
# ---------------------------------------------------------------------------
# Stat keys our own engines understand, handed to the model so it normalizes
# slip wording ("3PM", "Pts+Reb+Ast", "Anytime Goalscorer") to what we expect.
_NBA_STATS = ("points", "rebounds", "assists", "threes", "threes_attempted",
              "fgm", "fga", "ftm", "fta", "steals", "blocks", "turnovers",
              "pra", "pr", "pa", "ra", "stocks")
_SOCCER_STATS = ("goals", "assists", "goals_assists", "shots",
                 "shots_on_target", "cards")

_PARSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "bet_type": {"type": "STRING"},   # single | parlay
        "book": {"type": "STRING", "nullable": True},
        "legs": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "player": {"type": "STRING", "nullable": True},
                    "team": {"type": "STRING", "nullable": True},
                    "sport": {"type": "STRING"},
                    "market": {"type": "STRING"},
                    "stat": {"type": "STRING", "nullable": True},
                    "stat_raw": {"type": "STRING", "nullable": True},
                    "line": {"type": "NUMBER", "nullable": True},
                    "side": {"type": "STRING", "nullable": True},
                    "odds": {"type": "STRING", "nullable": True},
                },
                "required": ["sport", "market"],
            },
        },
    },
    "required": ["bet_type", "legs"],
}

_PARSE_PROMPT = f"""You are reading a screenshot of a sports betting slip (a single bet or a parlay).
Extract EVERY leg. Return JSON only, matching the provided schema.

For each leg:
- player: the player's full name for a player prop; null for team bets.
- team: the team/country if shown (e.g. "Lakers", "France"); else null.
- sport: one of nba, soccer, nfl, mlb, nhl, ncaab, ncaaf, tennis, golf, mma, other, unknown.
  Use basketball context (NBA teams/players) -> nba; international/club football -> soccer.
- market: one of player_prop, moneyline, spread, total, other.
- stat: NORMALIZE the prop to one of these keys when the sport matches.
    nba: {", ".join(_NBA_STATS)}
    soccer: {", ".join(_SOCCER_STATS)}
  Map wording: "3PM"/"Made Threes"->threes, "Pts+Reb+Ast"->pra, "Stl+Blk"->stocks,
  "Shots On Goal"/"SOG"->shots_on_target, "To Record A Card"/"Yellow Card"->cards.
  For "Anytime Goalscorer"/"To Score": sport=soccer, stat=goals, line=0.5, side=over.
  If the sport is not nba or soccer, leave stat as the raw label.
- stat_raw: the prop label exactly as printed.
- line: the over/under number (e.g. 25.5). For "X+" markets use X-0.5 (e.g. "2+ Goals"->1.5).
- side: over, under, yes, or null. "X+", "Anytime", "To Record" -> over.
- odds: the American odds for the leg if shown (e.g. "-110", "+140"), else null.

bet_type: "parlay" if multiple legs, else "single". book: sportsbook name if visible, else null.
If a value is not visible, use null rather than guessing."""


def parse_slip(image_bytes: bytes, mime_type: str = "image/png") -> dict:
    """Vision-parse a bet-slip screenshot into {bet_type, book, legs:[...]}."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    parts = [
        {"text": _PARSE_PROMPT},
        {"inline_data": {"mime_type": mime_type or "image/png", "data": b64}},
    ]
    text = _generate(parts, schema=_PARSE_SCHEMA, temperature=0.1)
    data = _loads_lenient(text)
    if not isinstance(data, dict) or "legs" not in data:
        raise LLMUnavailable("Could not read any legs from the screenshot.")
    data.setdefault("bet_type", "single")
    data.setdefault("book", None)
    return data


# ---------------------------------------------------------------------------
# Job 2: predict a leg our own models don't cover
# ---------------------------------------------------------------------------
def predict_leg(leg: dict) -> dict:
    """Estimate hit probability + the worry for one non-modeled leg.

    Returns {hit_probability, projection, worry_level, concern, factors:[...]}
    with factors in the same {title,value,detail} shape the engines use, so the
    frontend renders LLM legs and model legs identically. Falls back to a plain
    (ungrounded) call if Google Search grounding is rejected for this model/key.
    """
    today = datetime.date.today().isoformat()
    who = leg.get("player") or leg.get("team") or "the selection"
    desc = _describe_leg(leg)
    prompt = f"""You are a sharp sports betting analyst. Today is {today}.
Estimate the probability that this specific bet HITS (the bettor's side wins).

Bet: {desc}

Use Google Search for the most recent information available: the player's/team's
recent form, role and usage, injury or lineup news, the opponent, and the
matchup/venue. Be calibrated and honest -- most props land between 0.30 and 0.70;
never output exactly 0 or 1.

Return ONLY a JSON object (no markdown) with:
  "hit_probability": number 0-1 that the bettor's side hits,
  "projection": short string of your expected stat line (e.g. "~23 pts"), or null,
  "worry_level": "low" | "medium" | "high"  (high = most likely to bust the bet),
  "concern": one sentence on the single biggest risk for this leg about {who},
  "factors": array of 2-4 objects {{"title": str, "value": str, "detail": str}}
             summarizing form, matchup, injuries, and role."""

    try:
        text = _generate([{"text": prompt}], grounding=True, temperature=0.2)
    except LLMUnavailable:
        # Some models/keys reject the search tool -> retry without grounding.
        text = _generate([{"text": prompt}], grounding=False, temperature=0.2)

    data = _loads_lenient(text)
    prob = data.get("hit_probability")
    try:
        prob = float(prob)
    except (TypeError, ValueError):
        prob = None
    if prob is not None:
        prob = max(0.01, min(0.99, prob))

    factors = data.get("factors")
    if not isinstance(factors, list):
        factors = []
    clean = []
    for f in factors:
        if isinstance(f, dict):
            clean.append({
                "title": str(f.get("title", "")),
                "value": str(f.get("value", "")),
                "detail": str(f.get("detail", "")),
            })
    return {
        "hit_probability": prob,
        "projection": data.get("projection"),
        "worry_level": (data.get("worry_level") or "medium").lower(),
        "concern": data.get("concern") or "",
        "factors": clean,
    }


def _describe_leg(leg: dict) -> str:
    """Human-readable one-liner of a leg for the prediction prompt."""
    sport = leg.get("sport") or "unknown sport"
    who = leg.get("player") or leg.get("team") or "selection"
    market = leg.get("market")
    if market == "player_prop":
        side = (leg.get("side") or "over").upper()
        stat = leg.get("stat_raw") or leg.get("stat") or "stat"
        line = leg.get("line")
        line_txt = f" {line}" if line is not None else ""
        return f"[{sport}] {who} — {side}{line_txt} {stat}"
    bits = [f"[{sport}] {who}", market or "bet"]
    if leg.get("line") is not None:
        bits.append(str(leg["line"]))
    if leg.get("odds"):
        bits.append(f"({leg['odds']})")
    return " ".join(bits)

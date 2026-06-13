# Bet-Slip Analyzer

Upload a screenshot of a single bet or a parlay and get, for every leg:

- the **probability it hits** (the bettor's actual side),
- a **worry level** (low / medium / high) and a one-line concern,
- the **factors** behind the number,

plus a **parlay summary**: combined probability, fair decimal odds, and the
legs most likely to bust the slip.

## How it decides who grades each leg

| Leg | Graded by |
| --- | --- |
| NBA player prop, player we track | **our model** (`09_projections.py`) |
| World Cup player prop, player we track | **our model** (`22_soccer_projections.py`) |
| Any other sport (NFL, MLB, NHL, tennis…) | **Gemini Flash** (web-grounded) |
| Team markets (moneyline / spread / total) | **Gemini Flash** (web-grounded) |
| NBA/soccer player we don't track | **Gemini Flash** (web-grounded) |

The screenshot itself is always read by **Gemini Flash vision** (no OCR
library) into structured legs, then each leg is routed by the table above.

## Setup

The analyzer needs a Gemini API key (free at
[aistudio.google.com](https://aistudio.google.com/app/apikey)). Add it to the
same `.env` the rest of the app uses, or to your Railway service variables:

```
GEMINI_API_KEY=your_key_here
# optional, defaults to gemini-2.5-flash
GEMINI_MODEL=gemini-2.5-flash
```

Then install the new dependency (already in `requirements.txt`):

```
pip install -r requirements.txt   # adds python-multipart for the upload
```

No key? The NBA and soccer tools keep working; the **Scan Slip** tab just
reports that the LLM is unavailable. Check status any time:

```
GET /api/slip/health   ->  {"llm_available": true, "model": "gemini-2.5-flash"}
```

## API

```
POST /api/analyze-slip      (multipart form, field "image": the screenshot)
```

Response shape:

```json
{
  "bet_type": "parlay",
  "book": "DraftKings",
  "legs": [
    {
      "label": "LeBron James — OVER 25.5 Points",
      "source": "model",
      "sport": "nba",
      "hit_probability": 0.62,
      "projection": 27.1,
      "confidence_label": "LEAN",
      "worry_level": "low",
      "concern": "Model agrees with the OVER side.",
      "factors": [{ "title": "...", "value": "...", "detail": "..." }]
    },
    {
      "label": "Patrick Mahomes — OVER 275.5 Pass Yds",
      "source": "gemini",
      "sport": "nfl",
      "hit_probability": 0.41,
      "worry_level": "high",
      "concern": "Faces the league's #2 pass defense on the road."
    }
  ],
  "parlay": {
    "leg_count": 2,
    "graded_count": 2,
    "combined_probability": 0.254,
    "fair_decimal_odds": 3.94,
    "weakest_leg": "Patrick Mahomes — OVER 275.5 Pass Yds",
    "worry_legs": [ ... ],
    "note": "Combined assumes the legs are independent..."
  }
}
```

## Notes & limits

- **Independence caveat.** The combined parlay probability multiplies the legs,
  which assumes independence. Same-game legs are correlated, so treat the number
  as a rough estimate (the response says so too).
- **Gemini legs use web search** for recent form / injuries / lineups, so they
  reflect news up to the moment of the request — but they're model estimates,
  not data-backed like the NBA/soccer engines.
- Supported uploads: PNG, JPEG, WEBP, HEIC, up to 10 MB.
- Stat wording is normalized for you ("3PM" → threes, "Anytime Goalscorer" →
  goals @ 0.5, "Shots On Goal" → shots_on_target, etc.).

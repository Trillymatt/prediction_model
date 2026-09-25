# Fantasy advisor — setup

The **🏆 Fantasy** tab connects your Sleeper or ESPN league and gives you:

| Section | What it does |
| --- | --- |
| **Start/Sit** | This week's optimal lineup for your league's exact slots and scoring, with every projection adjusted for Vegas implied team totals, spread game script (RBs on big favorites, pass-catchers on big underdogs), and injury status. Shows the swaps vs. your current lineup, close calls, and floor/ceiling ranges. |
| **Trades** | Scans every team for 1-for-1, 2-for-1, 1-for-2 and 2-for-2 offers that raise **your** starting lineup *and* fill a need on **their** side, with an acceptance estimate and a pitch to send ("they rank 9/12 at RB and are deep at WR…"). |
| **Grade a trade** | Build any offer (or paste in a trade request you received) and see the lineup impact for both sides, value on paper, acceptance odds, and how your positional ranks change. |
| **Buy/Sell** | Buy-low targets (volume without points yet, projections above results, rising usage) and sell-high candidates (TD-fueled or efficiency spikes, falling usage, aging RBs). Yours are listed first. |
| **Needs** | A league-wide map of every team's starter strength rank at QB/RB/WR/TE and its tradeable depth, so you can trade from your surplus into their holes. |
| **Waivers** | Free agents who'd improve your lineup, a suggested drop, and Sleeper's trending adds that are still available in your league. |
| **Vegas & Parlays** | This week's spreads, totals, moneylines and implied team totals. Build a parlay from game lines and player props: each leg gets a hit probability, and the parlay gets fair odds, EV vs. the book's payout, and same-game correlation warnings. |

## How it works

- **One player universe.** Projections and weekly stats come from Sleeper's
  public data (`api.sleeper.com`), scored with *your league's* settings (PPR,
  half, 6-pt pass TDs, TE premium…). ESPN rosters are matched onto that
  database by name, position and team, so both platforms get identical math.
- **Rest-of-season value** = projected points per game (blended with actual
  results as the season goes on) above a replacement-level starter for your
  league size and lineup (flex slots included), times remaining games.
- **Team needs** compare each team's top starters at a position against the
  rest of the league; "depth" is a startable player beyond what that team
  needs to fill its lineup.
- **Buy-low/sell-high** compares points scored with *expected* points from
  opportunity (pass attempts, carries, targets at the position's league-wide
  rate). Usage is stickier than touchdowns.
- **Vegas**: implied team total = total / 2 ∓ spread / 2. Game-line parlay legs
  are priced off the market (no model edge claimed); prop legs use the model's
  weekly projection adjusted by the game's implied total.

## Connecting a league

- **Sleeper**: enter your username → pick the league. Or paste the league ID
  from `sleeper.com/leagues/<ID>`. No password or token needed.
- **ESPN public league**: paste the `leagueId` from the league URL.
- **ESPN private league**: also paste your `espn_s2` and `SWID` cookies
  (espn.com, logged in → DevTools → Application → Cookies). They're sent only
  with that request and never stored; the app only remembers the league ID
  and your team.

## Optional: live prop lines

Set `ODDS_API_KEY` (from [the-odds-api.com](https://the-odds-api.com)) in the
server's environment / `.env`. With it:

- Game lines use a median consensus across US books instead of ESPN's single
  provider. ESPN still decides which matchups are "this week" (The Odds API
  lists several weeks ahead), and games that have kicked off keep ESPN's line
  and close for betting in the UI.
- Each game on the Vegas board gets a **Props ›** button that loads every
  player-prop line (pass/rush/rec yards, receptions, pass TDs, anytime TD) and
  ranks them by model edge. Each game's props cost API credits, so they're
  fetched only when you tap, and cached for 15 minutes.

## API

| Method | Path | Body / query |
| --- | --- | --- |
| GET | `/api/fantasy/sleeper/leagues` | `username`, `season?` |
| POST | `/api/fantasy/league` | `{platform, league_id, season?, espn_s2?, swid?}` |
| POST | `/api/fantasy/analyze` | league fields + `team_id`, `week?` |
| POST | `/api/fantasy/trade` | league fields + `team_id`, `partner_id`, `give[]`, `get[]` |
| GET | `/api/fantasy/odds` | `week?` |
| GET | `/api/fantasy/players` | `q` |
| POST | `/api/fantasy/parlay` | `{legs: [...]}` (see `fantasy_engine.grade_parlay`) |
| GET | `/api/fantasy/props` | `event_id` (needs `ODDS_API_KEY`) |
| GET | `/api/fantasy/player` | `name`, `team?`, `position?`, `week?` — one player's projection (NFL tab) |

Errors come back as `{"detail": "..."}` with 400 (bad input), 403 (private
ESPN league without cookies), 404 (unknown user/league), 503 (props without
an `ODDS_API_KEY`) or 502 (Sleeper/ESPN/odds provider trouble).

No Supabase tables are needed: the fantasy side runs off the platforms'
public APIs. The Sleeper player database (~5 MB) is cached in `.cache/` for
24 hours; projections, stats and lines are cached in memory for minutes to
hours.

## Tests

```bash
pip install pytest httpx
python -m pytest tests/test_fantasy.py -q
```

The tests fake the HTTP layer, so they run offline. They cover league
normalization for both platforms, scoring, lineups, needs, trades,
buy/sell, start/sit, waivers, parlays, and every API route.

## Caveats

- Sleeper's projection/stat endpoints are public but unofficial; if they
  change shape, `fantasy_sources.py` is the one place to update.
- IDP slots aren't supported yet (they're ignored when building lineups).
- Parlay legs are treated as independent for the hit chance. The
  correlation notes tell you when that's optimistic or pessimistic, and legs
  that can't both win (both moneylines, over and under of the same total)
  are flagged and price the parlay at 0%.

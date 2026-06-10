# Soccer (World Cup 2026) — setup

The soccer side mirrors the NBA side: pull scripts load Supabase, engines
read Supabase + the researched `soccer_team_priors.json`, the same API/
frontend serve it. Three pieces of one-time setup:

## 1. Supabase changes (run in the SQL editor)

Your existing `soccer_player_match_logs` and `soccer_schedule` tables work
as-is. These additions make ingestion idempotent and autocomplete fast:

```sql
-- a) Player directory (powers the frontend's player autocomplete).
create table if not exists soccer_players (
  player_id   int4 primary key,
  player_name text,
  team        text,
  position    text,
  created_at  timestamptz default now()
);

-- b) Tie each log row to its match so re-runs upsert instead of duplicating.
alter table soccer_player_match_logs add column if not exists match_id int4;
create unique index if not exists soccer_logs_player_match_uq
  on soccer_player_match_logs (player_id, match_id);
```

Optional — extra prop markets. Add any of these and the pipeline + API
pick them up automatically on the next run (no code change, same pattern as
the NBA's optional fouls/oreb/dreb):

```sql
alter table soccer_player_match_logs
  add column if not exists passes int4,
  add column if not exists tackles int4,
  add column if not exists saves int4,
  add column if not exists fouls_committed int4,
  add column if not exists fouls_suffered int4;
```

Notes on existing columns: `xg`, `xa` and `key_passes` stay NULL when data
comes from the ESPN feed (it doesn't publish them). If you load logs from a
source that has them (FBref/Opta-style), the engine uses them where present;
nothing breaks while they're NULL.

## 2. Seed the data (one-time, ~10 minutes)

```bash
# Two years of international results + the WC schedule -> soccer_schedule.
# History matters: it's what the Elo/form ratings are computed from.
python 20_soccer_schedule.py --backfill 2024-06-01

# Player stats for every completed match ingested above.
python 21_soccer_player_logs.py --backfill 2024-06-01
```

Both scripts use ESPN's public API (no key). `--check` on either does a
dry run that prints what it parsed without writing. Note: run these from
your Mac like the NBA pipeline — same cron/LaunchAgent, which already calls
`refresh.py` (steps 20/21 are appended to its pipeline and the gate now
also checks soccer match days).

## 3. Try it

```bash
# Match outcome (win / draw / win, projected goals, totals, BTTS):
python 23_soccer_game_projections.py --home "Mexico" --away "South Africa"

# Player prop graded against your book's line:
python 22_soccer_projections.py --player "Mbappe" --stat shots --line 2.5

# Or the app: uvicorn api:app --port 8000  ->  ⚽ World Cup tab
```

## How the model works (and adapts during the tournament)

- **Team ratings**: every team starts at the Elo in `soccer_team_priors.json`
  (researched June 2026: scouting, form, eloratings.net). Every completed
  result in `soccer_schedule` after the snapshot date moves the rating
  (weighted: WC > qualifiers > friendlies, scaled by margin). So a team that
  over-performs in the group stage carries a higher rating into the knockouts
  automatically — no retraining step.
- **Match outcomes**: Elo gap → expected margin (~125 pts/goal, hosts +80),
  blended with recent goals scored/conceded rates → per-team expected goals →
  Poisson scoreline grid → P(win/draw/win), totals, BTTS, likely scores.
  Draws are a first-class outcome (three-way market).
- **Player props**: per-90 rates over the player's international
  appearances (L5/L10/career-weighted) × expected minutes × a matchup
  scaler from the team goal model (a striker facing a top defense gets
  marked down even if his logs are hot). Probabilities come from the
  Poisson tail — the right distribution for 0.5/1.5-goal-type lines.
- **Explanations**: every pick ships factor cards quoting the researched
  scouting profiles (coach & system, attack/defense notes, key players,
  group context) plus the live numbers that produced it.

`soccer_team_priors.json` is a knowledge file you can edit by hand — fix a
coach, update an injury note, nudge an Elo — and the engines pick it up on
restart.

# NFL — setup

The NFL side follows the same shape as the NBA and soccer sides: pull
scripts load Supabase, engines read Supabase, the same API/frontend serve
it. It's being built in stages:

| Stage | What | Status |
| --- | --- | --- |
| 1 | Schedule ingestion (`30_nfl_schedule.py`) + `/api/nfl/games` + NFL tab | ✅ done |
| 2 | Roster directory (`31_nfl_rosters.py`) + `/api/nfl/players` + `/api/nfl/roster` + player search/roster UI | ✅ done |
| 3 | Game-outcome model: Elo + a points model (`32_nfl_game_projections.py`) + `/api/nfl/game` | ✅ this release |
| 4 | Player-prop projections (pass/rush/rec yards, TDs, receptions) | 🔜 next |

The frontend NFL tab now has real game predictions: tap a game to see the
win probability, projected score, and both rosters. Tapping a player still
shows a "coming soon" note — that needs player game logs, which is stage 4.

### How the game model works (and how to make it smarter)

Unlike the soccer side, there's **no researched priors file** for NFL —
every team starts at a neutral Elo rating (1500) and the model learns
entirely from completed games in `nfl_schedule`. That means:

- **Before any backfill**, predictions are a coin flip with a small
  home-field nudge (still returns a real, honest projection — it just says
  so via a "Limited data" factor card and widens its uncertainty).
- **Backfilling 1-2 prior seasons** (see the load command below) gives Elo
  real signal before Week 1 even kicks off, and lets the "Scoring form"
  factor use actual recent games instead of nothing.
- **It keeps learning all season**: every completed game the nightly
  refresh ingests updates both teams' Elo and recent-scoring averages
  automatically — no retraining step, unlike the NBA's trained model.

This is a first-pass statistical model (Elo + a recency-weighted points
model, blended), not a trained ML model — there's no historical box-score
data ingested yet to train one on. See `32_nfl_game_projections.py`'s
docstring for the exact math. A backtest harness (like the soccer side's
`25_soccer_backtest.py`) would be a good next addition once a season of
real results exists to validate against.

## 1. Supabase table (run in the SQL editor)

```sql
create table if not exists nfl_schedule (
  id          bigint generated always as identity primary key,
  game_id     bigint unique not null,   -- ESPN event id
  game_date   date,                     -- US/Eastern game day
  game_time   text,                     -- HH:MM Eastern kickoff
  season      text,                     -- e.g. '2026'
  season_type text,                     -- preseason | regular | playoffs
  week        int4,
  home_team   text,
  away_team   text,
  status      text,                     -- upcoming | live | completed
  home_score  int4,
  away_score  int4,
  created_at  timestamptz default now()
);
```

```sql
-- Player directory (powers autocomplete + the per-game roster view).
create table if not exists nfl_players (
  player_id   int4 primary key,   -- ESPN athlete id
  player_name text,
  team        text,               -- canonical team name, e.g. 'Kansas City Chiefs'
  position    text,               -- e.g. 'QB', 'WR', 'CB'
  created_at  timestamptz default now()
);
```

If RLS is enabled on your project, allow the service key to read/write both
tables the same way your other tables do.

If `nfl_schedule` was created before `season_type` was added to this doc,
add it (the puller writes it on every upsert, so an older table will fail
with `PGRST204: Could not find the 'season_type' column`):

```sql
alter table nfl_schedule add column if not exists season_type text;
```

## 2. Load the schedule + rosters

```bash
# Sanity-check the ESPN feed first (writes nothing):
python 30_nfl_schedule.py --check

# Load the upcoming season (2026 schedule is already published):
python 30_nfl_schedule.py --backfill 2026-08-01 --days-ahead 220

# Recommended: also backfill 1-2 prior seasons so the Elo + points model
# has real signal from Week 1 instead of starting blind (see "How the game
# model works" above). ESPN's scoreboard API serves historical dates fine:
python 30_nfl_schedule.py --backfill 2024-09-01 --days-ahead 220

# Load rosters (all 32 teams; cheap, safe to re-run):
python 31_nfl_rosters.py --check      # sanity-check first, writes nothing
python 31_nfl_rosters.py

# Nightly refresh -- both scripts are already in refresh.py's pipeline,
# and it gates on nfl_schedule the same way it gates on soccer_schedule.
```

Dates/kickoffs are stored in US/Eastern, statuses are
`upcoming`/`live`/`completed` — identical conventions to `soccer_schedule`,
so the refresh gate and boards work the same way.

## 3. API

- `GET /api/nfl/games?days=30` — upcoming slate (soonest first,
  already-kicked-off games dropped).
- `GET /api/nfl/players?q=<text>` — player autocomplete against the roster
  directory.
- `GET /api/nfl/roster?home=&away=` — both teams' rosters for a matchup,
  ordered offense → defense → specialists (accepts full names or
  abbreviations, e.g. `KC` or `Kansas City Chiefs`).
- `GET /api/nfl/game?home=&away=` — win probability + projected score
  (Elo + points model; no table setup needed beyond `nfl_schedule` itself).

Like the soccer endpoints, all four degrade gracefully: if a table doesn't
exist yet you get a 503 with a setup hint and the NBA/soccer sides keep
working.

## Data sources (for the next stage)

- **Schedule / scores**: ESPN scoreboard API (in use, no key needed).
- **Rosters**: ESPN team + roster API (in use, no key needed).
- **Team & player stats**: ESPN summary/boxscore endpoints per event, or
  nflverse's public data releases (free CSVs, no key) for historical
  training data.
- **Injuries**: ESPN team injuries endpoint.

Stage 4 (player props) needs an `nfl_player_game_logs` table feeding a
props engine, the same shape as the soccer side's
`soccer_player_match_logs` → `22_soccer_projections.py`. The game model
(stage 3) needed no new table — it reads `nfl_schedule` directly.

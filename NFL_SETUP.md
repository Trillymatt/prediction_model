# NFL — setup

The NFL side follows the same shape as the NBA and soccer sides: pull
scripts load Supabase, engines read Supabase, the same API/frontend serve
it. It's being built in stages:

| Stage | What | Status |
| --- | --- | --- |
| 1 | Schedule ingestion (`30_nfl_schedule.py`) + `/api/nfl/games` + NFL tab | ✅ done |
| 2a | Roster directory (`31_nfl_rosters.py`) + `/api/nfl/players` + `/api/nfl/roster` + player search/roster UI | ✅ this release |
| 2b | Team stats + player game logs ingestion (the data props need) | 🔜 next |
| 3 | Game-outcome model (win prob, spread, total) | planned |
| 4 | Player-prop projections (pass/rush/rec yards, TDs, receptions) | planned |

The frontend NFL tab shows the upcoming schedule and lets you tap a game to
see both rosters, or search any player by name — but there are no stats or
projections behind them yet (that's stage 2b onward). Tapping a player shows
a "coming soon" note instead of a projection.

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

Like the soccer endpoints, all three degrade gracefully: if a table doesn't
exist yet you get a 503 with a setup hint and the NBA/soccer sides keep
working.

## Data sources (for the next stages)

- **Schedule / scores**: ESPN scoreboard API (in use, no key needed).
- **Rosters**: ESPN team + roster API (in use, no key needed).
- **Team & player stats**: ESPN summary/boxscore endpoints per event, or
  nflverse's public data releases (free CSVs, no key) for historical
  training data.
- **Injuries**: ESPN team injuries endpoint.

The plan is to mirror the soccer architecture: an `nfl_player_game_logs`
table feeding a props engine, priors/ratings feeding a game model, and the
existing multi-prop/slip tooling picking the sport up automatically.

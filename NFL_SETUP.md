# NFL — setup

The NFL side follows the same shape as the NBA and soccer sides: pull
scripts load Supabase, engines read Supabase, the same API/frontend serve
it. It's being built in stages:

| Stage | What | Status |
| --- | --- | --- |
| 1 | Schedule ingestion (`30_nfl_schedule.py`) + `/api/nfl/games` + NFL tab | ✅ this release |
| 2 | Team stats + player game logs ingestion | 🔜 next |
| 3 | Game-outcome model (win prob, spread, total) | planned |
| 4 | Player-prop projections (pass/rush/rec yards, TDs, receptions) | planned |

The frontend NFL tab already shows the upcoming schedule; projections light
up as the later stages land.

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

If RLS is enabled on your project, allow the service key to read/write it
the same way your other tables do.

## 2. Load the schedule

```bash
# Sanity-check the ESPN feed first (writes nothing):
python 30_nfl_schedule.py --check

# Load the upcoming season (2026 schedule is already published):
python 30_nfl_schedule.py --backfill 2026-08-01 --days-ahead 220

# Nightly refresh (add to the same pipeline as the other pullers):
python 30_nfl_schedule.py
```

Dates/kickoffs are stored in US/Eastern, statuses are
`upcoming`/`live`/`completed` — identical conventions to `soccer_schedule`,
so the refresh gate and boards work the same way.

## 3. API

`GET /api/nfl/games?days=30` returns the upcoming slate (soonest first,
already-kicked-off games dropped). Like the soccer endpoints it degrades
gracefully: if the table doesn't exist yet you get a 503 with a setup hint
and the NBA/soccer sides keep working.

## Data sources (for the next stages)

- **Schedule / scores**: ESPN scoreboard API (in use, no key needed).
- **Team & player stats**: ESPN summary/boxscore endpoints per event, or
  nflverse's public data releases (free CSVs, no key) for historical
  training data.
- **Injuries**: ESPN team injuries endpoint.

The plan is to mirror the soccer architecture: an `nfl_player_game_logs`
table feeding a props engine, priors/ratings feeding a game model, and the
existing multi-prop/slip tooling picking the sport up automatically.

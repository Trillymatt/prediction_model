# NFL — setup

The NFL side mirrors the NBA side: pull scripts load Supabase, engines read
Supabase + trained models, the same API/frontend serve it. It's now a full
sport — schedule, rosters, box-score logs, injuries, a game-outcome model and a
player-prop model — and it's the app's default tab.

| Stage | What | Status |
| --- | --- | --- |
| 1 | Schedule ingestion (`30_nfl_schedule.py`) + `/api/nfl/games` + NFL tab | ✅ |
| 2a | Roster directory (`31_nfl_rosters.py`) + player search/roster UI | ✅ |
| 2b | Box-score logs (`32`) + injuries (`33`) ingestion | ✅ |
| 3 | Game-outcome model — win/tie/loss, spread, total, team TDs (`36`/`37`, `39`) | ✅ |
| 4 | Player-prop model — pass/rush/rec yards, TDs, receptions, anytime-TD (`34`/`35`, `38`) | ✅ |

Everything degrades gracefully: before the logs are backfilled the game model
falls back to opponent-adjusted team ratings (so Week 1 still returns a call),
and the whole NFL side can be missing without touching the NBA app.

## 1. Supabase tables (run in the SQL editor)

`nfl_schedule` and `nfl_players` are unchanged from stages 1/2a. Add the two
tables the stats pipeline needs:

```sql
-- Per-player box scores. player_id is the ESPN athlete id (same id space as
-- nfl_players), game_id is the ESPN event id (same as nfl_schedule), so logs,
-- the roster directory and autocomplete all line up.
create table if not exists nfl_player_game_logs (
  id            bigint generated always as identity primary key,
  player_id     int4 not null,
  player_name   text,
  team          text,
  opponent      text,
  game_id       bigint not null,
  game_date     date,
  season        text,
  season_type   text,        -- preseason | regular | playoffs
  week          int4,
  home_away     text,        -- HOME | AWAY
  pass_att      int4,
  completions   int4,
  pass_yds      int4,
  pass_td       int4,
  interceptions int4,
  rush_att      int4,
  rush_yds      int4,
  rush_td       int4,
  targets       int4,
  receptions    int4,
  rec_yds       int4,
  rec_td        int4,
  fumbles_lost  int4,
  created_at    timestamptz default now()
);

-- Re-runs upsert instead of duplicating.
create unique index if not exists nfl_logs_player_game_uq
  on nfl_player_game_logs (player_id, game_id);

-- ESPN injury report (one snapshot per run, tagged with game_date).
create table if not exists nfl_injuries (
  id           bigint generated always as identity primary key,
  player_id    int4,
  player_name  text,
  team         text,
  position     text,
  status       text,          -- Out | Doubtful | Questionable | ...
  reason       text,
  game_date    date,
  created_at   timestamptz default now()
);
```

If RLS is enabled, allow the service key to read/write both tables the way your
other tables do.

## 2. Seed the data + train (one-time, ~15–20 minutes)

```bash
# Schedule + rosters (schedules are published in the spring). A couple of past
# seasons give the prop models real history:
python 30_nfl_schedule.py --backfill 2023-08-01 --days-ahead 220
python 31_nfl_rosters.py

# Box-score logs for every completed game since the backfill date, then injuries:
python 32_nfl_player_logs.py --backfill 2023-09-01   # --check first to dry-run
python 33_nfl_injuries.py

# Build training data + train the models (props, then the game model):
python 34_nfl_build_props_training.py && python 35_nfl_train_props.py
python 36_nfl_build_game_training.py && python 37_nfl_train_game_model.py
```

All ingestion uses ESPN's public API (no key). Every script takes `--check`
(where relevant) to dry-run. Run these the same way as the NBA pipeline — the
nightly `refresh.py` already includes steps 30–37 and gates on `nfl_schedule`,
so once seeded the models retrain themselves each week.

## 3. Try it

```bash
# Game outcome (win/tie/loss, spread, total, team touchdowns):
python 39_nfl_game_projections.py --home KC --away BUF

# Player prop graded against your book's line:
python 38_nfl_projections.py --player "Patrick Mahomes" --stat pass_yds --line 275.5
python 38_nfl_projections.py --player "Bijan Robinson" --stat any_td

# Today's board:
python daily_picks.py nfl

# Or the app: uvicorn api:app --port 8000  ->  🏈 NFL tab
```

## 4. API

- `GET /api/nfl/games?days=` — upcoming slate (soonest first).
- `GET /api/nfl/stats` — the markets the tool projects.
- `GET /api/nfl/players?q=` — player autocomplete.
- `GET /api/nfl/roster?home=&away=` — both rosters, featured players first,
  each tappable for its projection.
- `GET /api/nfl/project?player=&stat=&line=&opponent=&location=&game_type=` —
  one market, graded against your line.
- `GET /api/nfl/player/projections?player=&…` — several markets at once.
- `POST /api/nfl/project-batch` — grade a hand-built slip as a parlay.
- `GET /api/nfl/game?home=&away=&…` — game outcome.
- `GET /api/picks?sport=nfl` — the daily "My Picks" + per-game "Best Bets" board.

Markets: `pass_yds`, `pass_att`, `completions`, `pass_td`, `interceptions`,
`rush_yds`, `rush_att`, `rush_td`, `rec_yds`, `receptions`, `targets`, `rec_td`,
plus combos `pass_rush_yds`, `rush_rec_yds`, `total_td`, and `any_td` (anytime
touchdown, graded from the Poisson tail).

## How the model works (and grows through the season)

- **Team ratings / strength of schedule** (`nfl_common.team_ratings`): an
  opponent-adjusted Simple Rating System computed from `nfl_schedule` results,
  blended with last season's rating as a Week-1 prior and shrinking toward the
  live number as games come in. Because a rating already accounts for who a team
  has played, this *is* the strength-of-schedule read — `strength_of_schedule()`
  surfaces each team's average opponent rating across the games it has played
  and the ones still to come. No retrain is needed for ratings to move; they
  update the moment a result lands.
- **Game outcomes** (`37`/`39`): gradient-boosted models predict the margin, the
  total and total touchdowns as an adjustment over the ratings gap (+ home
  field), recent form and rest. Win/tie/loss, projected score, spread and team
  TDs are all derived from those, so they can't contradict each other. Ties are
  a first-class (if rare) outcome.
- **Player props** (`34`/`35`/`38`): each market anchors to the player's recent
  per-game rate (season → L5 → L3) and the model predicts the adjustment from
  the opponent's defense for that stat, home/away and season stage, with a
  per-row spread from its quantile models. Anytime-TD uses the Poisson tail on
  expected rushing + receiving touchdowns.
- **Injuries** (`33`/`nfl_common.injury_status`): the most recent status feeds
  the projection cards and drops `Out` players from the daily board.

As the season goes on, the nightly `refresh.py` re-ingests results, recomputes
ratings, and retrains both models — so a team that over- or under-performs its
schedule, and a player whose role changes, are reflected automatically, the same
way the soccer Elo adapted during the World Cup.

## Data sources

- **Schedule / scores / rosters / box scores / injuries**: ESPN public API (no
  key). For deeper historical training data you can later enrich from nflverse's
  free CSVs; the current pipeline is ESPN-only so every id lines up.

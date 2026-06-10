# Deploying to Railway

One Railway service runs everything: the Dockerfile builds the React frontend,
installs the Python API, bakes in the trained models from `models/`, and
serves it all from a single process. Open the service URL on your phone and
"Add to Home Screen" for an app-like experience.

## One-time setup

1. Push this repo to GitHub (the `models/` directory must be committed —
   it's ~7 MB and is what the deployed API predicts with).
2. In [Railway](https://railway.app): **New Project → Deploy from GitHub repo**,
   pick this repo. The `railway.toml` + `Dockerfile` are picked up automatically.
3. In the service's **Variables** tab add:
   - `SUPABASE_URL` — same value as in your local `.env`
   - `SUPABASE_KEY` — same value as in your local `.env`
4. In **Settings → Networking** click **Generate Domain**. That URL is your app.

Health check: `https://<your-domain>/api/health` should return
`{"status": "ok", "model_loaded": true}`.

## How data + models stay fresh

The deployed API reads **live data from Supabase** on every request (player
logs, schedule, injuries), so projections update as soon as the nightly
refresh ingests new games — no redeploy needed for data.

The **trained models** are baked into the image. The nightly `refresh.py`
(local cron on your Mac) retrains them after every game day; to ship the
retrained models, commit and push `models/` — Railway redeploys automatically.
In practice the models drift slowly (they're trained on ~5,000 games / ~90k
player-games), so pushing once a week is plenty.

## Why the refresh job stays on your Mac (for now)

`stats.nba.com` aggressively blocks requests from cloud datacenter IPs, so the
data-pull pipeline (01–08, 12) is unreliable from Railway/AWS/GCP boxes. The
deployed web app is unaffected — it only talks to Supabase at runtime. Keep
the existing local LaunchAgent (runs refresh.py at 4am, catches up after
sleep):

    ~/Library/LaunchAgents/com.moneyfromababy.refresh.plist
    # manage with: launchctl kickstart|bootout gui/501/com.moneyfromababy.refresh

If you later want the refresh fully in the cloud, the usual options are a
residential/rotating proxy for the nba_api calls, or a small always-on box
(e.g. a home Raspberry Pi) that runs the cron and pushes models to the repo.

## Soccer (World Cup)

The soccer side (scripts 20–23, `soccer_common.py`, `soccer_team_priors.json`)
deploys with the same image — no extra service. It pulls from ESPN's public
API, which (unlike stats.nba.com) doesn't block datacenter IPs, but the
simplest setup is to let the existing Mac cron run everything: steps 20/21
are part of `refresh.py`'s pipeline. One-time setup (Supabase SQL + history
backfill) is in `SOCCER_SETUP.md`. There are no baked soccer models — the
Elo/Poisson engine reads Supabase + the priors file live, so it needs no
redeploys during the tournament (only edit/commit `soccer_team_priors.json`
if you want to tweak the scouting notes).

## Local production test

    cd frontend && npm run build && cd ..
    uvicorn api:app --port 8000
    # open http://localhost:8000 — the API now serves the built frontend too

Or with Docker:

    docker build -t mfab .
    docker run -p 8000:8000 --env-file .env mfab

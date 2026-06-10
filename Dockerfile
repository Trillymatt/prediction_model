# ---- Stage 1: build the React frontend ------------------------------------
FROM node:20-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python runtime (API + engines + trained models) -------------
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The app: API, projection engines, pipeline scripts, the trained models,
# and the soccer scouting/Elo knowledge base (load_priors() falls back to
# defaults silently if it's missing -- so it must ship in the image).
COPY *.py ./
COPY soccer_team_priors.json ./
COPY models/ models/
COPY --from=frontend /app/frontend/dist frontend/dist

# SUPABASE_URL and SUPABASE_KEY come from the platform's env vars (no .env in
# the image). Railway injects PORT; default to 8000 for local docker runs.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]

// Tiny fetch wrappers around the FastAPI backend. URLs are relative; the Vite
// dev server proxies /api to http://localhost:8000.

async function getJSON(url) {
  const res = await fetch(url);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return body;
}

export function fetchStats() {
  return getJSON("/api/stats").then((d) => d.stats);
}

export function searchPlayers(q) {
  return getJSON(`/api/players?q=${encodeURIComponent(q)}`).then((d) => d.players);
}

export function projectStat({ player, stat, line, opponent, location, gameType }) {
  const params = new URLSearchParams({ player, stat, location, game_type: gameType });
  if (line !== "" && line != null) params.set("line", line);
  if (opponent) params.set("opponent", opponent);
  return getJSON(`/api/project?${params.toString()}`);
}

// Today's "My Picks" board. While the server is still computing the day's
// board this returns { status: "building" } and the caller should poll.
export function fetchDailyPicks(sport) {
  return getJSON(`/api/picks?sport=${encodeURIComponent(sport)}`);
}

export function fetchUpcomingGames(days = 10) {
  return getJSON(`/api/games?days=${days}`).then((d) => d.games);
}

export function projectGame({ home, away, date, gameId }) {
  const params = new URLSearchParams({ home, away });
  if (date) params.set("date", date);
  if (gameId) params.set("game_id", gameId);
  return getJSON(`/api/game?${params.toString()}`);
}

// ---- Roster + multi-prop (NBA & soccer) ------------------------------------
// Each call routes to the NBA or soccer endpoint by `sport`; the response
// shapes match, so the same UI renders both.

// Every player on both teams of a game/match, most-used players first.
export function fetchRoster({ home, away, sport = "nba" }) {
  const params = new URLSearchParams({ home, away });
  const base = sport === "soccer" ? "/api/soccer/roster" : "/api/roster";
  return getJSON(`${base}?${params.toString()}`);
}

// One player's projection across several stats at once (no line graded).
export function fetchPlayerProjections({
  player,
  stats,
  opponent,
  location = "auto",
  gameType = "auto",
  sport = "nba",
}) {
  const params = new URLSearchParams({ player });
  if (stats && stats.length) params.set("stats", stats.join(","));
  if (opponent) params.set("opponent", opponent);
  if (sport === "soccer") {
    return getJSON(`/api/soccer/player/projections?${params.toString()}`);
  }
  params.set("location", location);
  params.set("game_type", gameType);
  return getJSON(`/api/player/projections?${params.toString()}`);
}

// Grade a hand-built list of props and score them as a parlay.
export function projectBatch(props, sport = "nba") {
  const url = sport === "soccer" ? "/api/soccer/project-batch" : "/api/project-batch";
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ props }),
  }).then(async (res) => {
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
    return body;
  });
}

// ---- NFL (schedule only for now; projections land with the NFL pipeline) ---

export function fetchUpcomingNflGames(days = 30) {
  return getJSON(`/api/nfl/games?days=${days}`).then((d) => d.games);
}

// ---- Bet-slip analyzer -----------------------------------------------------

// Upload a screenshot of a line/parlay; get each leg graded (our model for
// NBA/soccer props, Gemini for everything else) plus a parlay summary.
export function analyzeSlip(file) {
  const form = new FormData();
  form.append("image", file);
  return fetch("/api/analyze-slip", { method: "POST", body: form }).then(
    async (res) => {
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
      return body;
    }
  );
}

// ---- Soccer (World Cup) ----------------------------------------------------

export function fetchSoccerStats() {
  return getJSON("/api/soccer/stats").then((d) => d.stats);
}

export function searchSoccerPlayers(q) {
  return getJSON(`/api/soccer/players?q=${encodeURIComponent(q)}`).then(
    (d) => d.players
  );
}

export function projectSoccerStat({ player, stat, line, opponent }) {
  const params = new URLSearchParams({ player, stat });
  if (line !== "" && line != null) params.set("line", line);
  if (opponent) params.set("opponent", opponent);
  return getJSON(`/api/soccer/project?${params.toString()}`);
}

export function fetchUpcomingSoccerGames(days = 10) {
  return getJSON(`/api/soccer/games?days=${days}`).then((d) => d.games);
}

export function projectSoccerGame({ home, away, date, matchId }) {
  const params = new URLSearchParams({ home, away });
  if (date) params.set("date", date);
  if (matchId) params.set("match_id", matchId);
  return getJSON(`/api/soccer/game?${params.toString()}`);
}

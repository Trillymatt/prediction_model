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

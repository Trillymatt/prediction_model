import { useEffect, useRef, useState } from "react";
import {
  fetchStats,
  searchPlayers,
  projectStat,
  fetchUpcomingGames,
  projectGame,
  fetchSoccerStats,
  searchSoccerPlayers,
  projectSoccerStat,
  fetchUpcomingSoccerGames,
  projectSoccerGame,
} from "./api.js";

// Friendly labels for the stat dropdown.
const STAT_LABELS = {
  points: "Points",
  rebounds: "Rebounds",
  assists: "Assists",
  threes: "3-Pointers Made",
  threes_attempted: "3-Point Attempts",
  fgm: "Field Goals Made",
  fga: "Field Goal Attempts",
  ftm: "Free Throws Made",
  fta: "Free Throw Attempts",
  steals: "Steals",
  blocks: "Blocks",
  turnovers: "Turnovers",
  fouls: "Personal Fouls",
  oreb: "Offensive Rebounds",
  dreb: "Defensive Rebounds",
  pra: "Pts + Reb + Ast",
  pr: "Pts + Reb",
  pa: "Pts + Ast",
  ra: "Reb + Ast",
  stocks: "Steals + Blocks",
};

// Dropdown order: scoring first, then shooting volume, boards/playmaking,
// defense/misc, then combos. Anything the API adds later lands at the end.
const STAT_ORDER = [
  "points", "rebounds", "assists", "threes", "threes_attempted",
  "fgm", "fga", "ftm", "fta",
  "oreb", "dreb", "steals", "blocks", "turnovers", "fouls",
  "pra", "pr", "pa", "ra", "stocks",
];

function sortStats(stats) {
  return [...stats].sort((a, b) => {
    const ia = STAT_ORDER.indexOf(a);
    const ib = STAT_ORDER.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });
}

function PlayerSearch({ selected, onSelect, searchFn = searchPlayers }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const boxRef = useRef(null);

  // Debounced search as the user types.
  useEffect(() => {
    if (selected && query === selected.player_name) return;
    if (query.trim().length < 2) {
      setResults([]);
      return;
    }
    const t = setTimeout(() => {
      searchFn(query).then(setResults).catch(() => setResults([]));
    }, 220);
    return () => clearTimeout(t);
  }, [query, selected, searchFn]);

  // Close the dropdown on outside click.
  useEffect(() => {
    const handler = (e) => {
      if (boxRef.current && !boxRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <div className="field player-search" ref={boxRef}>
      <label>Player</label>
      <input
        type="text"
        placeholder="Search a player…"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
          if (selected) onSelect(null);
        }}
        onFocus={() => setOpen(true)}
      />
      {open && results.length > 0 && (
        <ul className="dropdown">
          {results.map((p) => (
            <li
              key={p.player_id}
              onClick={() => {
                onSelect(p);
                setQuery(p.player_name);
                setOpen(false);
              }}
            >
              <span>{p.player_name}</span>
              <span className="muted">
                {p.team}
                {p.position ? ` · ${p.position}` : ""}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ConfidenceBar({ pOver, pUnder }) {
  const overPct = Math.round(pOver * 100);
  return (
    <div className="conf-bar">
      <div className="conf-bar-fill over" style={{ width: `${overPct}%` }}>
        {overPct >= 18 && <span>OVER {overPct}%</span>}
      </div>
      <div className="conf-bar-fill under" style={{ width: `${100 - overPct}%` }}>
        {100 - overPct >= 18 && <span>UNDER {Math.round(pUnder * 100)}%</span>}
      </div>
    </div>
  );
}

function ResultCard({ r }) {
  const loc = r.home_away === "AWAY" ? "@" : r.home_away === "HOME" ? "vs" : "";
  const recClass =
    r.confidence_label === "STRONG"
      ? "strong"
      : r.confidence_label === "LEAN"
      ? "lean"
      : "pass";
  const inj =
    r.injury &&
    !["", "active", "available"].includes((r.injury.status || "").toLowerCase())
      ? r.injury
      : null;

  return (
    <div className="card result">
      <div className="result-head">
        <div>
          <h2>{r.player_name}</h2>
          <div className="muted">
            {r.team} · {r.position}
            {r.opponent ? `  ${loc} ${r.opponent}` : ""}
          </div>
        </div>
        <span className={`badge ${r.method}`}>
          {r.method === "model" ? "trained model" : "heuristic"}
        </span>
      </div>

      <div className="proj">
        <div className="proj-num">
          {r.projection}
          <span className="sigma"> ± {r.sigma}</span>
        </div>
        <div className="muted">projected {STAT_LABELS[r.stat] || r.stat}</div>
      </div>

      {r.line != null && r.recommendation && (
        <>
          <ConfidenceBar pOver={r.p_over} pUnder={r.p_under} />
          <div className={`recommendation ${recClass}`}>
            {r.recommendation} {r.line}
            <span className="conf-label">
              {Math.round(r.confidence * 100)}% · {r.confidence_label}
            </span>
          </div>
        </>
      )}
      {r.line != null && !r.recommendation && r.note && (
        <div className="note">{r.note}</div>
      )}

      {inj && (
        <div className="injury">
          ⚠ {inj.status}
          {inj.reason ? ` — ${inj.reason}` : ""}
        </div>
      )}

      <div className="splits">
        <div>
          <span className="muted">L5</span>
          <b>{r.l5 ?? "–"}</b>
        </div>
        <div>
          <span className="muted">L10</span>
          <b>{r.l10 ?? "–"}</b>
        </div>
        <div>
          <span className="muted">Season</span>
          <b>{r.season_avg ?? "–"}</b>
        </div>
        {r.model_q16 != null && (
          <div>
            <span className="muted">Model range</span>
            <b>
              {r.model_q16}–{r.model_q84}
            </b>
          </div>
        )}
      </div>

      {r.factors && r.factors.length > 0 && (
        <div className="why">
          <h3>Why this projection</h3>
          {r.factors.map((f, i) => (
            <div className="factor" key={i}>
              <div className="factor-head">
                <span className="factor-title">{f.title}</span>
                <span className="factor-value">{f.value}</span>
              </div>
              <div className="factor-detail">{f.detail}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function WinProbBar({ r }) {
  const homePct = Math.round(r.p_home_win * 100);
  return (
    <div className="conf-bar">
      <div className="conf-bar-fill over" style={{ width: `${homePct}%` }}>
        {homePct >= 22 && (
          <span>
            {r.home_team} {homePct}%
          </span>
        )}
      </div>
      <div className="conf-bar-fill under" style={{ width: `${100 - homePct}%` }}>
        {100 - homePct >= 22 && (
          <span>
            {r.away_team} {100 - homePct}%
          </span>
        )}
      </div>
    </div>
  );
}

function GameResultCard({ r }) {
  const recClass =
    r.confidence_label === "STRONG"
      ? "strong"
      : r.confidence_label === "LEAN"
      ? "lean"
      : "pass";

  return (
    <div className="card result">
      <div className="result-head">
        <div>
          <h2>
            {r.away_team} @ {r.home_team}
          </h2>
          <div className="muted">
            {r.game_date} · {r.season_type}
          </div>
        </div>
        <span className="badge model">trained model</span>
      </div>

      <div className="proj">
        <div className="proj-num game-score">
          {r.projected_home_score}
          <span className="score-sep"> – </span>
          {r.projected_away_score}
        </div>
        <div className="muted">
          projected score ({r.home_team} home · {r.away_team} away)
        </div>
      </div>

      <WinProbBar r={r} />
      <div className={`recommendation ${recClass}`}>
        {r.predicted_winner} wins
        <span className="conf-label">
          {Math.round(Math.max(r.p_home_win, r.p_away_win) * 100)}% ·{" "}
          {r.confidence_label}
        </span>
      </div>

      <div className="splits">
        <div>
          <span className="muted">Margin ({r.home_team})</span>
          <b>
            {r.projected_margin > 0 ? "+" : ""}
            {r.projected_margin} ± {r.sigma_margin}
          </b>
        </div>
        <div>
          <span className="muted">Total</span>
          <b>
            {r.projected_total} ± {r.sigma_total}
          </b>
        </div>
        {r.season_series && r.season_series.games > 0 && (
          <div>
            <span className="muted">Series</span>
            <b>
              {r.season_series.home_wins}-{r.season_series.away_wins}
            </b>
          </div>
        )}
      </div>

      {r.factors && r.factors.length > 0 && (
        <div className="why">
          <h3>Why this call</h3>
          {r.factors.map((f, i) => (
            <div className="factor" key={i}>
              <div className="factor-head">
                <span className="factor-title">{f.title}</span>
                <span className="factor-value">{f.value}</span>
              </div>
              <div className="factor-detail">{f.detail}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function GameView() {
  const [games, setGames] = useState([]);
  const [selected, setSelected] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchUpcomingGames()
      .then(setGames)
      .catch(() => setGames([]));
  }, []);

  const run = async (g) => {
    setSelected(g.game_id);
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const r = await projectGame({
        home: g.home_team,
        away: g.away_team,
        date: g.game_date,
        gameId: g.game_id,
      });
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="card controls">
        <div className="field">
          <label>Upcoming games</label>
          {games.length === 0 && (
            <div className="muted">No upcoming games found in the schedule.</div>
          )}
          <div className="game-list">
            {games.map((g) => (
              <button
                key={g.game_id}
                className={`game-pick ${selected === g.game_id ? "active" : ""}`}
                onClick={() => run(g)}
                disabled={loading}
              >
                <span className="game-teams">
                  {g.away_team} @ {g.home_team}
                </span>
                <span className="muted">{g.game_date}</span>
              </button>
            ))}
          </div>
        </div>
        {loading && <div className="muted">Crunching…</div>}
        {error && <div className="error">{error}</div>}
      </div>

      {result && <GameResultCard r={result} />}
    </>
  );
}

// ===========================================================================
// Soccer (World Cup)
// ===========================================================================

const SOCCER_STAT_LABELS = {
  goals: "Goals",
  assists: "Assists",
  goals_assists: "Goals + Assists",
  shots: "Shots",
  shots_on_target: "Shots on Target",
  key_passes: "Key Passes",
  passes: "Passes",
  tackles: "Tackles",
  saves: "Saves (GK)",
  cards: "Cards (Yellow + Red)",
  fouls_committed: "Fouls Committed",
  fouls_suffered: "Fouls Drawn",
};

const SOCCER_STAT_ORDER = [
  "goals", "assists", "goals_assists", "shots", "shots_on_target",
  "key_passes", "passes", "tackles", "cards", "fouls_committed",
  "fouls_suffered", "saves",
];

function sortSoccerStats(stats) {
  return [...stats].sort((a, b) => {
    const ia = SOCCER_STAT_ORDER.indexOf(a);
    const ib = SOCCER_STAT_ORDER.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });
}

const recClassFor = (label) =>
  label === "STRONG" ? "strong" : label === "LEAN" ? "lean" : "pass";

function FactorList({ title, factors }) {
  if (!factors || factors.length === 0) return null;
  return (
    <div className="why">
      <h3>{title}</h3>
      {factors.map((f, i) => (
        <div className="factor" key={i}>
          <div className="factor-head">
            <span className="factor-title">{f.title}</span>
            <span className="factor-value">{f.value}</span>
          </div>
          <div className="factor-detail">{f.detail}</div>
        </div>
      ))}
    </div>
  );
}

function SoccerResultCard({ r }) {
  const loc = r.home_away === "AWAY" ? "@" : "vs";
  const recClass = recClassFor(r.confidence_label);

  return (
    <div className="card result">
      <div className="result-head">
        <div>
          <h2>{r.player_name}</h2>
          <div className="muted">
            {r.team}
            {r.position ? ` · ${r.position}` : ""}
            {r.opponent ? `  ${loc} ${r.opponent}` : ""}
            {r.match_date ? ` · ${r.match_date}` : ""}
          </div>
        </div>
        <span className="badge model">poisson model</span>
      </div>

      <div className="proj">
        <div className="proj-num">
          {r.projection}
          {r.sigma != null && <span className="sigma"> ± {r.sigma}</span>}
        </div>
        <div className="muted">
          projected {SOCCER_STAT_LABELS[r.stat] || r.stat}
        </div>
      </div>

      {r.line != null && r.recommendation && (
        <>
          <ConfidenceBar pOver={r.p_over} pUnder={r.p_under} />
          <div className={`recommendation ${recClass}`}>
            {r.recommendation} {r.line}
            <span className="conf-label">
              {Math.round(r.confidence * 100)}% · {r.confidence_label}
            </span>
          </div>
        </>
      )}
      {r.note && <div className="note">{r.note}</div>}

      <div className="splits">
        <div>
          <span className="muted">L5</span>
          <b>{r.l5 ?? "–"}</b>
        </div>
        <div>
          <span className="muted">L10</span>
          <b>{r.l10 ?? "–"}</b>
        </div>
        <div>
          <span className="muted">Career avg</span>
          <b>{r.avg ?? "–"}</b>
        </div>
        <div>
          <span className="muted">Minutes</span>
          <b>{r.expected_minutes ?? "–"}</b>
        </div>
      </div>

      <FactorList title="Why this projection" factors={r.factors} />
    </div>
  );
}

function ThreeWayBar({ r }) {
  const homePct = Math.round(r.p_home_win * 100);
  const drawPct = Math.round(r.p_draw * 100);
  const awayPct = 100 - homePct - drawPct;
  return (
    <div className="conf-bar">
      <div className="conf-bar-fill over" style={{ width: `${homePct}%` }}>
        {homePct >= 16 && (
          <span>
            {r.home_team} {homePct}%
          </span>
        )}
      </div>
      <div className="conf-bar-fill draw" style={{ width: `${drawPct}%` }}>
        {drawPct >= 14 && <span>Draw {drawPct}%</span>}
      </div>
      <div className="conf-bar-fill under" style={{ width: `${awayPct}%` }}>
        {awayPct >= 16 && (
          <span>
            {r.away_team} {awayPct}%
          </span>
        )}
      </div>
    </div>
  );
}

function SoccerGameCard({ r }) {
  const recClass = recClassFor(r.confidence_label);
  const pickText =
    r.predicted_outcome === "Draw" ? "Draw" : `${r.predicted_outcome} wins`;

  return (
    <div className="card result">
      <div className="result-head">
        <div>
          <h2>
            {r.home_team} vs {r.away_team}
          </h2>
          <div className="muted">
            {r.match_date}
            {r.group ? ` · Group ${r.group}` : ""} · {r.competition}
          </div>
        </div>
        <span className="badge model">poisson · elo</span>
      </div>

      <div className="proj">
        <div className="proj-num game-score">
          {r.projected_home_goals}
          <span className="score-sep"> – </span>
          {r.projected_away_goals}
        </div>
        <div className="muted">projected goals (90 minutes)</div>
      </div>

      <ThreeWayBar r={r} />
      <div className={`recommendation ${recClass}`}>
        {pickText}
        <span className="conf-label">
          {Math.round(r.confidence * 100)}% · {r.confidence_label}
        </span>
      </div>

      <div className="splits">
        <div>
          <span className="muted">Over {r.total_line}</span>
          <b>{Math.round(r.p_over * 100)}%</b>
        </div>
        <div>
          <span className="muted">BTTS</span>
          <b>{Math.round(r.p_btts * 100)}%</b>
        </div>
        <div>
          <span className="muted">{r.home_team} or draw</span>
          <b>{Math.round(r.p_home_or_draw * 100)}%</b>
        </div>
        <div>
          <span className="muted">{r.away_team} or draw</span>
          <b>{Math.round(r.p_away_or_draw * 100)}%</b>
        </div>
      </div>

      {r.top_scores && r.top_scores.length > 0 && (
        <div className="note">
          Likely scores:{" "}
          {r.top_scores
            .slice(0, 3)
            .map((s) => `${s.score} (${Math.round(s.p * 100)}%)`)
            .join(" · ")}
        </div>
      )}

      <FactorList title="Why this call" factors={r.factors} />
    </div>
  );
}

function SoccerGameView() {
  const [games, setGames] = useState([]);
  const [selected, setSelected] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchUpcomingSoccerGames()
      .then(setGames)
      .catch(() => setGames([]));
  }, []);

  const run = async (g) => {
    setSelected(g.match_id);
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const r = await projectSoccerGame({
        home: g.home_team,
        away: g.away_team,
        date: g.match_date,
        matchId: g.match_id,
      });
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="card controls">
        <div className="field">
          <label>Upcoming matches</label>
          {games.length === 0 && (
            <div className="muted">
              No upcoming matches found — run 20_soccer_schedule.py to load
              the schedule.
            </div>
          )}
          <div className="game-list">
            {games.map((g) => (
              <button
                key={g.match_id}
                className={`game-pick ${selected === g.match_id ? "active" : ""}`}
                onClick={() => run(g)}
                disabled={loading}
              >
                <span className="game-teams">
                  {g.home_team} vs {g.away_team}
                </span>
                <span className="muted">
                  {g.match_date}
                  {g.competition === "FIFA World Cup" ? " · World Cup" : ""}
                </span>
              </button>
            ))}
          </div>
        </div>
        {loading && <div className="muted">Crunching…</div>}
        {error && <div className="error">{error}</div>}
      </div>

      {result && <SoccerGameCard r={result} />}
    </>
  );
}

function SoccerPropsView() {
  const [stats, setStats] = useState([]);
  const [player, setPlayer] = useState(null);
  const [stat, setStat] = useState("goals");
  const [line, setLine] = useState("");
  const [opponent, setOpponent] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchSoccerStats()
      .then((s) => setStats(sortSoccerStats(s)))
      .catch(() => {});
  }, []);

  const run = async () => {
    if (!player) {
      setError("Pick a player first.");
      return;
    }
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const r = await projectSoccerStat({
        player: player.player_name,
        stat,
        line,
        opponent: opponent.trim(),
      });
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="card controls">
        <PlayerSearch
          selected={player}
          onSelect={setPlayer}
          searchFn={searchSoccerPlayers}
        />

        <div className="row">
          <div className="field">
            <label>Stat</label>
            <select value={stat} onChange={(e) => setStat(e.target.value)}>
              {stats.map((s) => (
                <option key={s} value={s}>
                  {SOCCER_STAT_LABELS[s] || s}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Your line</label>
            <input
              type="number"
              step="0.5"
              placeholder="e.g. 0.5"
              value={line}
              onChange={(e) => setLine(e.target.value)}
            />
          </div>
        </div>

        <div className="row">
          <div className="field">
            <label>Opponent (optional)</label>
            <input
              type="text"
              placeholder="auto-detect from schedule"
              value={opponent}
              onChange={(e) => setOpponent(e.target.value)}
            />
          </div>
        </div>

        <button className="go" onClick={run} disabled={loading}>
          {loading ? "Crunching…" : "Get Confidence"}
        </button>
        {error && <div className="error">{error}</div>}
      </div>

      {result && <SoccerResultCard r={result} />}
    </>
  );
}

export default function App() {
  const [sport, setSport] = useState("nba");
  const [mode, setMode] = useState("props");
  const [stats, setStats] = useState([]);
  const [player, setPlayer] = useState(null);
  const [stat, setStat] = useState("points");
  const [line, setLine] = useState("");
  const [opponent, setOpponent] = useState("");
  const [location, setLocation] = useState("auto");
  const [gameType, setGameType] = useState("auto");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchStats().then((s) => setStats(sortStats(s))).catch(() => {});
  }, []);

  const run = async () => {
    if (!player) {
      setError("Pick a player first.");
      return;
    }
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const r = await projectStat({
        player: player.player_name,
        stat,
        line,
        opponent: opponent.trim().toUpperCase(),
        location,
        gameType,
      });
      setResult(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>🍼 Money From a Baby</h1>
        <p className="muted">
          Paste any book's line. Get a data-backed confidence on the over/under.
        </p>
      </header>

      <div className="tabs sport-tabs">
        <button
          className={sport === "nba" ? "tab active" : "tab"}
          onClick={() => setSport("nba")}
        >
          🏀 NBA
        </button>
        <button
          className={sport === "soccer" ? "tab active" : "tab"}
          onClick={() => setSport("soccer")}
        >
          ⚽ World Cup
        </button>
      </div>

      <div className="tabs">
        <button
          className={mode === "props" ? "tab active" : "tab"}
          onClick={() => setMode("props")}
        >
          Player Props
        </button>
        <button
          className={mode === "game" ? "tab active" : "tab"}
          onClick={() => setMode("game")}
        >
          {sport === "soccer" ? "Match Outcome" : "Game Outcome"}
        </button>
      </div>

      {sport === "soccer" && mode === "game" && <SoccerGameView />}
      {sport === "soccer" && mode === "props" && <SoccerPropsView />}

      {sport === "nba" && mode === "game" && <GameView />}

      {sport === "nba" && mode === "props" && (
      <div className="card controls">
        <PlayerSearch selected={player} onSelect={setPlayer} />

        <div className="row">
          <div className="field">
            <label>Stat</label>
            <select value={stat} onChange={(e) => setStat(e.target.value)}>
              {stats.map((s) => (
                <option key={s} value={s}>
                  {STAT_LABELS[s] || s}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Your line</label>
            <input
              type="number"
              step="0.5"
              placeholder="e.g. 27.5"
              value={line}
              onChange={(e) => setLine(e.target.value)}
            />
          </div>
        </div>

        <div className="row">
          <div className="field">
            <label>Opponent (optional)</label>
            <input
              type="text"
              placeholder="auto-detect"
              value={opponent}
              onChange={(e) => setOpponent(e.target.value)}
            />
          </div>
          <div className="field">
            <label>Location</label>
            <select value={location} onChange={(e) => setLocation(e.target.value)}>
              <option value="auto">Auto</option>
              <option value="home">Home</option>
              <option value="away">Away</option>
            </select>
          </div>
          <div className="field">
            <label>Game type</label>
            <select value={gameType} onChange={(e) => setGameType(e.target.value)}>
              <option value="auto">Auto (by date)</option>
              <option value="regular">Regular Season</option>
              <option value="playoffs">Playoffs</option>
            </select>
          </div>
        </div>

        <button className="go" onClick={run} disabled={loading}>
          {loading ? "Crunching…" : "Get Confidence"}
        </button>
        {error && <div className="error">{error}</div>}
      </div>
      )}

      {sport === "nba" && mode === "props" && result && <ResultCard r={result} />}
    </div>
  );
}

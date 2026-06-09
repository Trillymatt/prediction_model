import { useEffect, useRef, useState } from "react";
import {
  fetchStats,
  searchPlayers,
  projectStat,
  fetchUpcomingGames,
  projectGame,
} from "./api.js";

// Friendly labels for the stat dropdown.
const STAT_LABELS = {
  points: "Points",
  rebounds: "Rebounds",
  assists: "Assists",
  threes: "3-Pointers Made",
  steals: "Steals",
  blocks: "Blocks",
  turnovers: "Turnovers",
  pra: "Pts + Reb + Ast",
  pr: "Pts + Reb",
  pa: "Pts + Ast",
  ra: "Reb + Ast",
  stocks: "Steals + Blocks",
};

function PlayerSearch({ selected, onSelect }) {
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
      searchPlayers(query).then(setResults).catch(() => setResults([]));
    }, 220);
    return () => clearTimeout(t);
  }, [query, selected]);

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
                {p.team} · {p.position}
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

export default function App() {
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
    fetchStats().then(setStats).catch(() => {});
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
          Game Outcome
        </button>
      </div>

      {mode === "game" && <GameView />}

      {mode === "props" && (
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

      {mode === "props" && result && <ResultCard r={result} />}
    </div>
  );
}

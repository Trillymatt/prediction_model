import { useEffect, useRef, useState } from "react";
import {
  fetchStats,
  searchPlayers,
  projectStat,
  fetchUpcomingGames,
  projectGame,
  fetchDailyPicks,
  fetchSoccerStats,
  searchSoccerPlayers,
  projectSoccerStat,
  fetchUpcomingSoccerGames,
  projectSoccerGame,
  analyzeSlip,
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
  const inputRef = useRef(null);

  const clear = () => {
    setQuery("");
    setResults([]);
    setOpen(false);
    if (selected) onSelect(null);
    inputRef.current?.focus();
  };

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
      <div className="input-wrap">
        <input
          ref={inputRef}
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
        {query && (
          <button
            type="button"
            className="clear-input"
            aria-label="Clear player"
            onClick={clear}
          >
            ×
          </button>
        )}
      </div>
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

// Wraps a freshly-loaded result card and scrolls it into view: on the game/
// match outcome tabs the upcoming-games list sits above the card, so without
// this the prediction renders off-screen and the user has to scroll to it.
// Smooth scroll first; if the browser skipped the animation (reduced-motion
// settings, some embedded webviews), snap instantly so it always lands.
function ScrollIntoView({ children }) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    const fallback = setTimeout(() => {
      const top = el.getBoundingClientRect().top;
      if (top < -20 || top > window.innerHeight / 2) {
        el.scrollIntoView({ block: "start" });
      }
    }, 700);
    return () => clearTimeout(fallback);
  }, []);
  return <div ref={ref}>{children}</div>;
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

// `collapseWhy` hides the factor list behind an "Explanation" toggle — used on
// the per-game board where the card sits inside an already-expanded game.
function GameResultCard({ r, collapseWhy }) {
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

      {collapseWhy ? (
        <CollapsibleFactors factors={r.factors} />
      ) : (
        <FactorList title="Why this call" factors={r.factors} />
      )}
    </div>
  );
}

// ===========================================================================
// Daily "My Picks" board
// ===========================================================================
// The server builds one board per sport per day (the model's most confident
// calls for today's slate, plus a per-game breakdown) and caches it; while
// it's still computing we get status "building" and poll.
function usePicksBoard(sport) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let timer;
    setData(null);
    setError("");
    const load = () =>
      fetchDailyPicks(sport)
        .then((d) => {
          if (cancelled) return;
          setData(d);
          if (d.status === "building") timer = setTimeout(load, 4000);
        })
        .catch((e) => {
          if (!cancelled) setError(e.message);
        });
    load();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [sport]);

  return { data, error };
}

// Accordion list of player picks. Tapping a pick expands the full projection
// card — same component as a manual search — plus tabs for the player's
// other projected stats, so "why" and "what else" are one tap away.
function PicksList({ picks, sport }) {
  const [open, setOpen] = useState(-1);
  const [predIdx, setPredIdx] = useState(0);
  const labels = sport === "soccer" ? SOCCER_STAT_LABELS : STAT_LABELS;
  const Card = sport === "soccer" ? SoccerResultCard : ResultCard;

  return (
    <div className="picks-list">
      {picks.map((p, i) => (
        <div key={`${p.player_id}-${p.stat}`}>
          <button
            className={`pick-row ${open === i ? "active" : ""}`}
            onClick={() => {
              setOpen(open === i ? -1 : i);
              setPredIdx(0);
            }}
          >
            <span className="pick-left">
              <span className="pick-name">{p.player_name}</span>
              <span className="muted">
                {p.team}
                {p.opponent
                  ? ` ${p.home_away === "AWAY" ? "@" : "vs"} ${p.opponent}`
                  : ""}
              </span>
            </span>
            <span
              className={`pick-claim ${
                p.direction === "UNDER" ? "under" : "over"
              }`}
            >
              {p.headline}
              <span className="pick-prob">
                {Math.round(p.probability * 100)}%
              </span>
            </span>
          </button>

          {open === i && (
            <div className="pick-detail">
              {p.predictions.length > 1 && (
                <div className="pred-tabs">
                  {p.predictions.map((r, j) => (
                    <button
                      key={j}
                      className={`pred-tab ${predIdx === j ? "active" : ""}`}
                      onClick={() => setPredIdx(j)}
                    >
                      {labels[r.stat] || r.stat}
                    </button>
                  ))}
                </div>
              )}
              <Card r={p.predictions[predIdx] || p.predictions[0]} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function DailyPicks({ sport }) {
  const { data, error } = usePicksBoard(sport);

  // If picks aren't available (e.g. soccer tables not set up), stay out of
  // the way — the search tool above still works.
  if (error) return null;

  const picks = (data && data.picks) || [];

  return (
    <div className="card picks">
      <div className="picks-head">
        <label>🎯 My Picks{data?.slate_date ? ` · ${data.slate_date}` : ""}</label>
        <span className="muted">the model's most confident calls today</span>
      </div>

      {!data && <div className="muted">Loading…</div>}
      {data?.status === "building" && (
        <div className="muted">
          Building today's board — the first load of the day takes a minute…
        </div>
      )}
      {data?.note && <div className="muted picks-note">{data.note}</div>}
      {data?.status === "ready" && picks.length === 0 && (
        <div className="muted">No confident picks for this slate.</div>
      )}

      <PicksList
        key={`${sport}-${data?.slate_date || ""}`}
        picks={picks}
        sport={sport}
      />
    </div>
  );
}

// Factor list hidden behind an "Explanation" toggle. A drop-down (rather than
// a pop-up) keeps the reading flow inline and works better on mobile.
function CollapsibleFactors({ factors }) {
  const [show, setShow] = useState(false);
  if (!factors || factors.length === 0) return null;
  return (
    <div className="why">
      <button className="why-toggle" onClick={() => setShow(!show)}>
        Explanation
        <span className="why-caret">{show ? "▲" : "▼"}</span>
      </button>
      {show &&
        factors.map((f, i) => (
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

// ===========================================================================
// Per-game "Best Bets" board
// ===========================================================================
// Every game on the slate, each expandable into the game-outcome call
// (win prob for NBA, win/draw/win for soccer) with its explanation tucked
// behind a toggle, plus the model's strongest player picks for that game.
function GameBoard({ sport }) {
  const { data, error } = usePicksBoard(sport);
  const [open, setOpen] = useState(-1);

  useEffect(() => setOpen(-1), [sport]);

  const games = (data && data.games) || [];
  const GameCard = sport === "soccer" ? SoccerGameCard : GameResultCard;

  return (
    <div className="card picks">
      <div className="picks-head">
        <label>
          🔥 Best Bets by Game{data?.slate_date ? ` · ${data.slate_date}` : ""}
        </label>
        <span className="muted">the model's strongest calls, game by game</span>
      </div>

      {error && <div className="muted">Board unavailable: {error}</div>}
      {!data && !error && <div className="muted">Loading…</div>}
      {data?.status === "building" && (
        <div className="muted">
          Building today's board — the first load of the day takes a minute…
        </div>
      )}
      {data?.note && <div className="muted picks-note">{data.note}</div>}
      {data?.status === "ready" && games.length === 0 && (
        <div className="muted">No games found for this slate.</div>
      )}

      <div className="picks-list">
        {games.map((g, i) => {
          const o = g.outcome;
          const winner = o && (o.predicted_winner || o.predicted_outcome);
          const winnerPct =
            o &&
            Math.round(
              (o.confidence ?? Math.max(o.p_home_win, o.p_away_win)) * 100
            );
          return (
            <div key={g.game_id}>
              <button
                className={`pick-row ${open === i ? "active" : ""}`}
                onClick={() => setOpen(open === i ? -1 : i)}
              >
                <span className="pick-left">
                  <span className="pick-name">
                    {sport === "soccer"
                      ? `${g.home_team} vs ${g.away_team}`
                      : `${g.away_team} @ ${g.home_team}`}
                  </span>
                  <span className="muted">
                    {g.game_date}
                    {g.competition ? ` · ${g.competition}` : ""}
                  </span>
                </span>
                {o && (
                  <span className="pick-claim over">
                    {winner === "Draw" ? "Draw" : winner}
                    <span className="pick-prob">{winnerPct}%</span>
                  </span>
                )}
              </button>

              {open === i && (
                <div className="pick-detail">
                  {o ? (
                    <GameCard r={o} collapseWhy />
                  ) : (
                    <div className="muted picks-note">
                      No outcome projection available for this game.
                    </div>
                  )}
                  <h3 className="bets-title">Best player bets</h3>
                  {g.picks.length > 0 ? (
                    <PicksList picks={g.picks} sport={sport} />
                  ) : (
                    <div className="muted">
                      No confident player picks for this game.
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
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

      {result && (
        <ScrollIntoView>
          <GameResultCard r={result} />
        </ScrollIntoView>
      )}
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

function SoccerGameCard({ r, collapseWhy }) {
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

      {collapseWhy ? (
        <CollapsibleFactors factors={r.factors} />
      ) : (
        <FactorList title="Why this call" factors={r.factors} />
      )}
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

      {result && (
        <ScrollIntoView>
          <SoccerGameCard r={result} />
        </ScrollIntoView>
      )}
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

      {result && (
        <ScrollIntoView>
          <SoccerResultCard r={result} />
        </ScrollIntoView>
      )}

      <DailyPicks sport="soccer" />
    </>
  );
}

// ---- Bet-slip analyzer ------------------------------------------------------

function pct(p) {
  return p == null ? "–" : `${Math.round(p * 100)}%`;
}

function worryClass(level) {
  if (level === "high") return "worry high";
  if (level === "medium") return "worry medium";
  if (level === "low") return "worry low";
  return "worry unknown";
}

function HitBar({ p }) {
  const w = p == null ? 0 : Math.round(p * 100);
  const cls = p == null ? "" : p >= 0.55 ? "over" : p >= 0.45 ? "draw" : "under";
  return (
    <div className="conf-bar">
      <div className={`conf-bar-fill ${cls}`} style={{ width: `${w}%` }}>
        {w >= 16 && <span>{w}% to hit</span>}
      </div>
    </div>
  );
}

function SlipLegCard({ leg }) {
  const matchup =
    leg.opponent ? ` ${leg.home_away === "AWAY" ? "@" : "vs"} ${leg.opponent}` : "";
  return (
    <div className="card slip-leg">
      <div className="result-head">
        <div>
          <h3 className="leg-title">{leg.label}</h3>
          <div className="muted">
            {(leg.sport || "").toUpperCase()}
            {leg.player_name && leg.player_name !== leg.label ? ` · ${leg.player_name}` : ""}
            {matchup}
          </div>
        </div>
        <div className="leg-badges">
          <span className={`badge ${leg.source === "model" ? "model" : "heuristic"}`}>
            {leg.source === "model"
              ? "our model"
              : leg.source === "gemini"
              ? "Gemini"
              : "no grade"}
          </span>
          <span className={worryClass(leg.worry_level)}>{leg.worry_level}</span>
        </div>
      </div>

      <HitBar p={leg.hit_probability} />
      <div className="leg-stat-row">
        <div>
          <span className="muted">Hit chance</span>
          <b>{pct(leg.hit_probability)}</b>
        </div>
        {leg.projection != null && (
          <div>
            <span className="muted">Projection</span>
            <b>{typeof leg.projection === "number" ? leg.projection : leg.projection}</b>
          </div>
        )}
        {leg.confidence_label && (
          <div>
            <span className="muted">Grade</span>
            <b>{leg.confidence_label}</b>
          </div>
        )}
      </div>

      {leg.concern && (
        <div className={`leg-concern ${leg.worry_level === "high" ? "alarm" : ""}`}>
          ⚠ {leg.concern}
        </div>
      )}
      {leg.error && <div className="note">{leg.error}</div>}
      {leg.note && <div className="note">{leg.note}</div>}

      <FactorList title="Why" factors={leg.factors} />
    </div>
  );
}

function SlipParlaySummary({ parlay, betType }) {
  if (!parlay) return null;
  return (
    <div className="card slip-summary">
      <div className="result-head">
        <h2>{betType === "parlay" ? "Parlay read" : "Bet read"}</h2>
        {parlay.combined_probability != null && (
          <span className="badge model">{pct(parlay.combined_probability)} to cash</span>
        )}
      </div>
      <div className="splits">
        <div>
          <span className="muted">Legs</span>
          <b>{parlay.leg_count}</b>
        </div>
        <div>
          <span className="muted">Graded</span>
          <b>{parlay.graded_count}</b>
        </div>
        {parlay.fair_decimal_odds != null && (
          <div>
            <span className="muted">Fair odds</span>
            <b>{parlay.fair_decimal_odds}x</b>
          </div>
        )}
      </div>

      {parlay.worry_legs && parlay.worry_legs.length > 0 && (
        <div className="why">
          <h3>Legs to worry about</h3>
          {parlay.worry_legs.map((w, i) => (
            <div className="factor" key={i}>
              <div className="factor-head">
                <span className="factor-title">{w.label}</span>
                <span className={worryClass(w.worry_level)}>{pct(w.hit_probability)}</span>
              </div>
              {w.concern && <div className="factor-detail">{w.concern}</div>}
            </div>
          ))}
        </div>
      )}
      {parlay.note && <div className="note">{parlay.note}</div>}
    </div>
  );
}

function SlipAnalyzer() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const onPick = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError("");
    setPreview(URL.createObjectURL(f));
  };

  const run = async () => {
    if (!file) {
      setError("Pick a screenshot of your line or parlay first.");
      return;
    }
    setError("");
    setLoading(true);
    setResult(null);
    try {
      setResult(await analyzeSlip(file));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="card controls">
        <p className="muted">
          Upload a screenshot of your bet slip or parlay. We grade NBA &amp; World
          Cup player props with our own model and use Gemini for anything else,
          then flag the legs most likely to bust it.
        </p>
        <label
          className="slip-drop"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            onPick(e.dataTransfer.files?.[0]);
          }}
        >
          {preview ? (
            <img src={preview} alt="bet slip preview" className="slip-preview" />
          ) : (
            <span className="muted">Tap to choose an image, or drop it here</span>
          )}
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp,image/heic"
            onChange={(e) => onPick(e.target.files?.[0])}
            hidden
          />
        </label>
        <button className="go" onClick={run} disabled={loading}>
          {loading ? "Reading the slip…" : "Analyze slip"}
        </button>
        {error && <div className="error">{error}</div>}
      </div>

      {result && (
        <ScrollIntoView>
          <div className="slip-results">
            <SlipParlaySummary parlay={result.parlay} betType={result.bet_type} />
            {result.legs.map((leg, i) => (
              <SlipLegCard leg={leg} key={i} />
            ))}
          </div>
        </ScrollIntoView>
      )}
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
        <button
          className={sport === "slip" ? "tab active" : "tab"}
          onClick={() => setSport("slip")}
        >
          📸 Scan Slip
        </button>
      </div>

      {sport === "slip" && <SlipAnalyzer />}

      {sport !== "slip" && (
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
        <button
          className={mode === "bets" ? "tab active" : "tab"}
          onClick={() => setMode("bets")}
        >
          Best Bets
        </button>
      </div>
      )}

      {sport !== "slip" && mode === "bets" && <GameBoard sport={sport} />}

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

      {sport === "nba" && mode === "props" && result && (
        <ScrollIntoView>
          <ResultCard r={result} />
        </ScrollIntoView>
      )}

      {sport === "nba" && mode === "props" && <DailyPicks sport="nba" />}
    </div>
  );
}

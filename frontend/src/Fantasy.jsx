import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchSleeperLeagues,
  connectFantasyLeague,
  analyzeFantasyTeam,
  evaluateFantasyTrade,
  fetchFantasyOdds,
  searchFantasyPlayers,
  gradeFantasyParlay,
  fetchFantasyProps,
} from "./api.js";

// Remembers which league/team you picked (never ESPN cookies) so the report is
// one tap away next time. Storage can be unavailable (private mode) -- then
// it simply doesn't remember.
const SAVED_KEY = "mfab_fantasy_league";
function loadSaved() {
  try {
    return JSON.parse(localStorage.getItem(SAVED_KEY) || "null");
  } catch {
    return null;
  }
}
function saveLeague(v) {
  try {
    if (v) localStorage.setItem(SAVED_KEY, JSON.stringify(v));
    else localStorage.removeItem(SAVED_KEY);
  } catch {
    /* storage unavailable */
  }
}

const fmt = (x, d = 1) => (x == null ? "–" : Number(x).toFixed(d));
const pct = (p) => (p == null ? "–" : `${Math.round(p * 100)}%`);
const signed = (x, d = 1) => (x == null ? "–" : `${x > 0 ? "+" : ""}${Number(x).toFixed(d)}`);
const american = (o) => (o == null ? "–" : o > 0 ? `+${o}` : `${o}`);
const spreadStr = (s) => (s == null ? "–" : s === 0 ? "PK" : s > 0 ? `+${s}` : `${s}`);

function PlayerTag({ p }) {
  if (!p) return <span className="muted">empty</span>;
  return (
    <span className="ff-player">
      <b>{p.name}</b>{" "}
      <span className="muted">
        {p.pos}
        {p.team ? ` · ${p.team}` : ""}
      </span>
      {p.injury && <span className="ff-inj">{p.injury}</span>}
    </span>
  );
}

// ===========================================================================
// Connect a league
// ===========================================================================
function ConnectLeague({ onConnected }) {
  const [platform, setPlatform] = useState("sleeper");
  const [username, setUsername] = useState("");
  const [leagues, setLeagues] = useState(null);
  const [leagueId, setLeagueId] = useState("");
  const [season, setSeason] = useState("");
  const [s2, setS2] = useState("");
  const [swid, setSwid] = useState("");
  const [showPrivate, setShowPrivate] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const findLeagues = async () => {
    setError("");
    setBusy(true);
    try {
      const r = await fetchSleeperLeagues(username.trim(), season.trim());
      setLeagues(r.leagues);
      if (r.leagues.length === 1) setLeagueId(r.leagues[0].league_id);
      if (!r.leagues.length) setError("No NFL leagues found for that user this season.");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const connect = async () => {
    setError("");
    const conn = { platform, league_id: leagueId.trim() };
    if (season.trim()) conn.season = season.trim();
    if (platform === "espn" && s2.trim() && swid.trim()) {
      conn.espn_s2 = s2.trim();
      conn.swid = swid.trim();
    }
    if (!/^\d+$/.test(conn.league_id)) {
      setError("Enter the numeric league ID.");
      return;
    }
    setBusy(true);
    try {
      const league = await connectFantasyLeague(conn);
      onConnected(conn, league, username.trim());
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card controls">
      <div className="tabs ff-subtabs">
        {["sleeper", "espn"].map((p) => (
          <button
            key={p}
            className={platform === p ? "tab active" : "tab"}
            onClick={() => {
              setPlatform(p);
              setError("");
            }}
          >
            {p === "sleeper" ? "Sleeper" : "ESPN"}
          </button>
        ))}
      </div>

      {platform === "sleeper" && (
        <>
          <div className="row">
            <div className="field">
              <label>Sleeper username</label>
              <input
                value={username}
                placeholder="your username"
                onChange={(e) => setUsername(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && username.trim() && findLeagues()}
              />
            </div>
          </div>
          <button className="go ff-secondary" onClick={findLeagues} disabled={busy || !username.trim()}>
            {busy && !leagues ? "Looking…" : "Find my leagues"}
          </button>
          {leagues && leagues.length > 0 && (
            <div className="field" style={{ marginTop: 14 }}>
              <label>League</label>
              <select value={leagueId} onChange={(e) => setLeagueId(e.target.value)}>
                <option value="">Pick a league…</option>
                {leagues.map((l) => (
                  <option key={l.league_id} value={l.league_id}>
                    {l.name} ({l.teams} teams)
                  </option>
                ))}
              </select>
            </div>
          )}
          <p className="muted ff-hint">
            Or paste a league ID from the Sleeper URL (sleeper.com/leagues/<b>ID</b>).
          </p>
        </>
      )}

      <div className="row">
        <div className="field">
          <label>League ID</label>
          <input
            value={leagueId}
            inputMode="numeric"
            placeholder={platform === "espn" ? "from ?leagueId= in the ESPN URL" : "league ID"}
            onChange={(e) => setLeagueId(e.target.value)}
          />
        </div>
        <div className="field ff-narrow">
          <label>Season</label>
          <input
            value={season}
            inputMode="numeric"
            placeholder="current"
            onChange={(e) => setSeason(e.target.value)}
          />
        </div>
      </div>

      {platform === "espn" && (
        <>
          <button className="ff-link" onClick={() => setShowPrivate((v) => !v)}>
            {showPrivate ? "▾" : "▸"} Private league? Add your ESPN cookies
          </button>
          {showPrivate && (
            <>
              <p className="muted ff-hint">
                On espn.com (logged in): DevTools → Application → Cookies → copy{" "}
                <code>espn_s2</code> and <code>SWID</code>. They're sent only to load your
                league and are never saved.
              </p>
              <div className="field">
                <label>espn_s2</label>
                <input value={s2} onChange={(e) => setS2(e.target.value)} />
              </div>
              <div className="field">
                <label>SWID</label>
                <input value={swid} placeholder="{XXXXXXXX-…}" onChange={(e) => setSwid(e.target.value)} />
              </div>
            </>
          )}
        </>
      )}

      <button className="go" onClick={connect} disabled={busy || !leagueId.trim()}>
        {busy && (leagues || platform === "espn") ? "Connecting…" : "Connect league"}
      </button>
      {error && <div className="error">{error}</div>}
    </div>
  );
}

// ===========================================================================
// Report sections
// ===========================================================================
function StartSit({ ss }) {
  return (
    <div className="card">
      <h3 className="ff-h">
        Optimal lineup <span className="muted">· {fmt(ss.projected_total)} pts projected</span>
      </h3>
      {ss.current_total != null && ss.swaps.length > 0 && (
        <div className="ff-callout">
          <b>Change your lineup</b> ({signed(ss.projected_total - ss.current_total)} pts):
          <ul>
            {ss.swaps.map((s, i) => (
              <li key={i}>
                {s.start && <>Start <b>{s.start}</b></>}
                {s.start && s.bench && " over "}
                {s.bench && <b>{s.bench}</b>}
                {!s.start && " → bench"}
                {s.start && s.bench && <span className="muted"> ({signed(s.gain)})</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
      {ss.current_total != null && ss.swaps.length === 0 && (
        <div className="ff-callout ok">Your current lineup is already optimal. ✅</div>
      )}
      <div className="ff-list">
        {ss.lineup.map((row, i) => (
          <LineupRow key={i} slot={row.slot} r={row.player} />
        ))}
      </div>
      {ss.close_calls.length > 0 && (
        <>
          <h4 className="ff-h4">Close calls</h4>
          {ss.close_calls.map((c, i) => (
            <div key={i} className="ff-small">
              {c.slot}: <b>{c.starter}</b> over {c.alternative} by {fmt(c.margin)} — if you need a
              safer week take the higher {c.tiebreak === "floor" ? "floor (the starter)" : "ceiling (the alternative)"}.
            </div>
          ))}
        </>
      )}
      <h4 className="ff-h4">Bench</h4>
      <div className="ff-list">
        {ss.bench.map((r) => (
          <LineupRow key={r.id} slot="BN" r={r} />
        ))}
      </div>
      <p className="note">
        Projections are adjusted for Vegas implied team totals (league avg {fmt(ss.avg_implied)}),
        spread game script, and injury status.
      </p>
    </div>
  );
}

function LineupRow({ slot, r }) {
  return (
    <div className="ff-row">
      <span className="ff-slot">{slot.replace("SUPER_FLEX", "SFLX").replace("_FLEX", "")}</span>
      <div className="ff-grow">
        <PlayerTag p={r} />
        {r && (
          <div className="muted ff-small">
            {r.opponent ? `${r.home ? "vs" : "@"} ${r.opponent}` : "no game"}
            {r.implied != null && ` · implied ${fmt(r.implied)} · ${spreadStr(r.spread)}`}
            {r.notes && r.notes.length > 0 && ` · ${r.notes.filter((n) => !n.startsWith("team implied")).join(" · ")}`}
          </div>
        )}
      </div>
      {r && (
        <div className="ff-num">
          <b>{fmt(r.adj_proj)}</b>
          <div className="muted ff-small">
            {fmt(r.floor)}–{fmt(r.ceiling)}
          </div>
        </div>
      )}
    </div>
  );
}

function TradeCard({ t, teamName }) {
  return (
    <div className="ff-trade">
      <div className="ff-trade-head">
        <span>
          with <b>{t.partner || teamName}</b>
        </span>
        <span className={`ff-verdict ${t.verdict.startsWith("win") ? "good" : t.verdict.includes("hurts") ? "bad" : ""}`}>
          {t.verdict}
        </span>
      </div>
      <div className="ff-trade-sides">
        <div>
          <div className="muted ff-small">You give</div>
          {t.give.map((p) => (
            <div key={p.id}>
              <PlayerTag p={p} />
            </div>
          ))}
        </div>
        <div>
          <div className="muted ff-small">You get</div>
          {t.get.map((p) => (
            <div key={p.id}>
              <PlayerTag p={p} />
            </div>
          ))}
        </div>
      </div>
      <div className="ff-trade-stats">
        <span>Your lineup {signed(t.my_gain_ppg)}/wk</span>
        <span>Theirs {signed(t.their_gain_ppg)}/wk</span>
        <span>Accept ~{pct(t.accept_prob)}</span>
        <span>
          Value {fmt(t.value_give, 0)} → {fmt(t.value_get, 0)}
        </span>
      </div>
      {t.pitch && <div className="ff-pitch">💬 {t.pitch}</div>}
      {t.rank_change && (
        <div className="ff-small muted">
          Your position ranks:{" "}
          {Object.entries(t.rank_change)
            .map(([pos, v]) => `${pos} ${v.before}→${v.after}`)
            .join(" · ")}
        </div>
      )}
    </div>
  );
}

function TradeBuilder({ conn, report }) {
  const myId = report.my_team.team_id;
  const partners = report.power_rankings.filter((t) => t.team_id !== myId);
  const [partner, setPartner] = useState("");
  const [give, setGive] = useState([]);
  const [get, setGet] = useState([]);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setGet([]);
    setResult(null);
  }, [partner]);

  const toggle = (list, setList, id) =>
    setList(list.includes(id) ? list.filter((x) => x !== id) : [...list, id]);

  const run = async () => {
    setError("");
    setBusy(true);
    try {
      setResult(await evaluateFantasyTrade(conn, myId, partner, give, get));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const Pick = ({ players, chosen, onToggle }) => (
    <div className="ff-pick">
      {players
        .filter((p) => !["K", "DEF"].includes(p.pos))
        .map((p) => (
          <label key={p.id} className={chosen.includes(p.id) ? "on" : ""}>
            <input type="checkbox" checked={chosen.includes(p.id)} onChange={() => onToggle(p.id)} />
            <PlayerTag p={p} />
            <span className="muted ff-small">{fmt(p.ros_ppg)}</span>
          </label>
        ))}
    </div>
  );

  return (
    <div className="card">
      <h3 className="ff-h">Grade a trade</h3>
      <p className="muted ff-hint">Got a trade request, or want to test an offer? Build it here.</p>
      <div className="field">
        <label>Trade partner</label>
        <select value={partner} onChange={(e) => setPartner(e.target.value)}>
          <option value="">Pick a team…</option>
          {partners.map((t) => (
            <option key={t.team_id} value={t.team_id}>
              {t.name} ({t.owner})
            </option>
          ))}
        </select>
      </div>
      {partner && (
        <div className="ff-trade-sides">
          <div>
            <div className="muted ff-small">You give</div>
            <Pick players={report.rosters[myId] || []} chosen={give} onToggle={(id) => toggle(give, setGive, id)} />
          </div>
          <div>
            <div className="muted ff-small">You get</div>
            <Pick players={report.rosters[partner] || []} chosen={get} onToggle={(id) => toggle(get, setGet, id)} />
          </div>
        </div>
      )}
      <button className="go" onClick={run} disabled={busy || !partner || (!give.length && !get.length)}>
        {busy ? "Grading…" : "Grade trade"}
      </button>
      {error && <div className="error">{error}</div>}
      {result && <TradeCard t={result} />}
    </div>
  );
}

function Trades({ conn, report }) {
  const { trades } = report;
  return (
    <>
      <div className="card">
        <h3 className="ff-h">Trade ideas</h3>
        <p className="muted ff-hint">
          Offers that raise your starting lineup <i>and</i> fill a need on their side — the ones
          managers actually accept.
        </p>
        {trades.length === 0 && (
          <div className="note">No win-win trades found right now — check back after waivers.</div>
        )}
        {trades.map((t, i) => (
          <TradeCard key={i} t={t} />
        ))}
      </div>
      <TradeBuilder conn={conn} report={report} />
    </>
  );
}

function BuySell({ report }) {
  const names = useMemo(
    () => Object.fromEntries(report.power_rankings.map((t) => [t.team_id, t.name])),
    [report]
  );
  const List = ({ rows, kind }) =>
    rows.length === 0 ? (
      <div className="note">Nothing stands out yet.</div>
    ) : (
      rows.map((r) => (
        <div key={r.id} className="ff-bs">
          <div className="ff-row">
            <div className="ff-grow">
              <PlayerTag p={r} />
              <div className="muted ff-small">
                {r.mine ? "your team" : names[r.owner_team_id] || "free agent"}
                {r.snap_pct != null && ` · ${Math.round(r.snap_pct * 100)}% snaps`}
                {` · ${fmt(r.opp_pg)} opp/g`}
              </div>
            </div>
            <div className="ff-num">
              <b>{fmt(r.ppg)}</b>
              <div className="muted ff-small">
                exp {fmt(r.xfp_pg)} · proj {fmt(r.proj_pg)}
              </div>
            </div>
          </div>
          <ul className={`ff-reasons ${kind}`}>
            {r.reasons.map((x, i) => (
              <li key={i}>{x}</li>
            ))}
          </ul>
        </div>
      ))
    );
  return (
    <>
      <div className="card">
        <h3 className="ff-h">📉 Buy low</h3>
        <p className="muted ff-hint">
          Volume without the points (yet). Usage predicts future scoring better than past points.
        </p>
        <List rows={report.buy_sell.buy_low} kind="buy" />
      </div>
      <div className="card">
        <h3 className="ff-h">📈 Sell high</h3>
        <p className="muted ff-hint">Points outrunning their usage — cash in before it corrects. Yours first.</p>
        <List rows={report.buy_sell.sell_high} kind="sell" />
      </div>
    </>
  );
}

function LeagueNeeds({ report }) {
  const myId = report.my_team.team_id;
  const positions = ["QB", "RB", "WR", "TE"].filter((p) =>
    report.power_rankings.some((t) => t.positions[p])
  );
  return (
    <div className="card">
      <h3 className="ff-h">League needs map</h3>
      <p className="muted ff-hint">
        Each team's starter strength by position (rank of {report.power_rankings.length}). Red = need,
        green = strong or has tradeable depth. Trade <i>from</i> your green into their red.
      </p>
      <div className="ff-needs">
        <div className="ff-needs-row head">
          <span className="ff-grow">Team · proj pts/wk</span>
          {positions.map((p) => (
            <span key={p} className="ff-cell">
              {p}
            </span>
          ))}
        </div>
        {report.power_rankings.map((t) => (
          <div key={t.team_id} className={`ff-needs-row ${t.team_id === myId ? "mine" : ""}`}>
            <span className="ff-grow">
              <b>{t.name}</b>
              <div className="muted ff-small">
                {t.record} · {fmt(t.lineup_ppg)}
                {t.positions && Object.values(t.positions).some((v) => v.depth.length)
                  ? ` · depth: ${Object.entries(t.positions)
                      .filter(([, v]) => v.depth.length)
                      .map(([p, v]) => `${p} (${v.depth.map((d) => d.name.split(" ").slice(-1)[0]).join(", ")})`)
                      .join("; ")}`
                  : ""}
              </div>
            </span>
            {positions.map((p) => {
              const v = t.positions[p];
              return (
                <span key={p} className={`ff-cell grade-${v ? v.grade : "ok"}`}>
                  {v ? v.rank : "–"}
                </span>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

function Waivers({ report }) {
  const { targets, trending } = report.waivers;
  return (
    <div className="card">
      <h3 className="ff-h">Waiver targets</h3>
      {targets.length === 0 && <div className="note">No free agent beats your roster right now.</div>}
      {targets.map((t) => (
        <div key={t.id} className="ff-row">
          <div className="ff-grow">
            <PlayerTag p={t} />
            <div className="muted ff-small">
              {fmt(t.ros_ppg)} pts/g rest of season
              {t.trending_adds > 0 && ` · 🔥 ${t.trending_adds.toLocaleString()} adds`}
              {t.drop && ` · drop ${t.drop.name}`}
            </div>
          </div>
          <div className="ff-num">
            <b>{signed(t.lineup_gain_ppg)}</b>
            <div className="muted ff-small">lineup/wk</div>
          </div>
        </div>
      ))}
      {trending.length > 0 && (
        <>
          <h4 className="ff-h4">Trending on Sleeper (available in your league)</h4>
          {trending.map((t) => (
            <div key={t.id} className="ff-small">
              🔥 <PlayerTag p={t} /> — {t.trending_adds.toLocaleString()} adds
            </div>
          ))}
        </>
      )}
    </div>
  );
}

function MyRoster({ report }) {
  const { my_team } = report;
  return (
    <div className="card">
      <h3 className="ff-h">
        {my_team.name} <span className="muted">· {my_team.owner}</span>
      </h3>
      <div className="ff-small">
        {my_team.needs.needs.length > 0 ? (
          <>
            Needs: <b>{my_team.needs.needs.join(", ")}</b>
          </>
        ) : (
          "No glaring holes."
        )}
        {my_team.needs.surplus.length > 0 && (
          <>
            {" "}
            · Trade from: <b>{my_team.needs.surplus.join(", ")}</b>
          </>
        )}
      </div>
      <div className="ff-list">
        {my_team.roster.map((p) => (
          <div key={p.id} className="ff-row">
            <div className="ff-grow">
              <PlayerTag p={p} />
              <div className="muted ff-small">
                {p.games > 0 ? `${fmt(p.ppg)} pts/g over ${p.games}` : "no games yet"} · this week{" "}
                {fmt(p.proj_week)}
              </div>
            </div>
            <div className="ff-num">
              <b>{fmt(p.ros_ppg)}</b>
              <div className="muted ff-small">value {fmt(p.value, 0)}</div>
            </div>
          </div>
        ))}
      </div>
      <p className="note">
        Value = rest-of-season points above a replacement-level starter in your league's format.
      </p>
    </div>
  );
}

const SECTIONS = [
  ["startsit", "Start/Sit"],
  ["trades", "Trades"],
  ["buysell", "Buy/Sell"],
  ["needs", "Needs"],
  ["waivers", "Waivers"],
  ["roster", "Roster"],
];

function LeagueReport({ conn, league, teamId, onReset }) {
  const [report, setReport] = useState(null);
  const [section, setSection] = useState("startsit");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const load = () => {
    setLoading(true);
    setError("");
    analyzeFantasyTeam(conn, teamId)
      .then(setReport)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };
  useEffect(load, [conn.platform, conn.league_id, teamId]);

  return (
    <>
      <div className="card ff-league-head">
        <div className="ff-grow">
          <b>{league.name}</b>
          <div className="muted ff-small">
            {conn.platform === "espn" ? "ESPN" : "Sleeper"} · {league.teams.length} teams · {league.scoring}
            {report && ` · Week ${report.league.week}`}
          </div>
        </div>
        <button className="ff-link" onClick={load} disabled={loading}>
          ↻
        </button>
        <button className="ff-link" onClick={onReset}>
          Switch
        </button>
      </div>
      {league.unmatched_players && league.unmatched_players.length > 0 && (
        <div className="note">
          Couldn't match {league.unmatched_players.length} ESPN player(s):{" "}
          {league.unmatched_players.slice(0, 5).join(", ")}
        </div>
      )}
      {loading && !report && (
        <div className="card">
          <div className="note">Crunching projections, usage, and Vegas lines for every roster…</div>
        </div>
      )}
      {error && <div className="error">{error}</div>}
      {report && (
        <>
          <div className="tabs ff-subtabs ff-scroll">
            {SECTIONS.map(([k, label]) => (
              <button key={k} className={section === k ? "tab active" : "tab"} onClick={() => setSection(k)}>
                {label}
              </button>
            ))}
          </div>
          {section === "startsit" && <StartSit ss={report.start_sit} />}
          {section === "trades" && <Trades conn={conn} report={report} />}
          {section === "buysell" && <BuySell report={report} />}
          {section === "needs" && <LeagueNeeds report={report} />}
          {section === "waivers" && <Waivers report={report} />}
          {section === "roster" && <MyRoster report={report} />}
        </>
      )}
    </>
  );
}

function MyLeague() {
  const saved = loadSaved();
  const [conn, setConn] = useState(saved ? saved.conn : null);
  const [league, setLeague] = useState(null);
  const [teamId, setTeamId] = useState(saved ? saved.teamId : "");
  const [error, setError] = useState("");

  // Reconnect a remembered league (ESPN private leagues need cookies again).
  useEffect(() => {
    if (conn && !league) {
      connectFantasyLeague(conn)
        .then(setLeague)
        .catch((e) => {
          setError(e.message);
          setConn(null);
        });
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const reset = () => {
    saveLeague(null);
    setConn(null);
    setLeague(null);
    setTeamId("");
  };

  if (!conn || !league) {
    return (
      <>
        {conn && !league && !error && (
          <div className="card">
            <div className="note">Reconnecting your league…</div>
          </div>
        )}
        {error && <div className="error">{error}</div>}
        {!conn && (
          <ConnectLeague
            onConnected={(c, lg, username) => {
              setConn(c);
              setLeague(lg);
              const mine = username
                ? lg.teams.find((t) => (t.owner || "").toLowerCase() === username.toLowerCase())
                : null;
              if (mine) {
                setTeamId(mine.team_id);
                const { espn_s2, swid, ...safe } = c; // eslint-disable-line no-unused-vars
                saveLeague({ conn: safe, teamId: mine.team_id });
              }
            }}
          />
        )}
      </>
    );
  }

  if (!teamId) {
    return (
      <div className="card controls">
        <h3 className="ff-h">{league.name}</h3>
        <div className="field">
          <label>Which team is yours?</label>
          <select
            value=""
            onChange={(e) => {
              setTeamId(e.target.value);
              const { espn_s2, swid, ...safe } = conn; // eslint-disable-line no-unused-vars
              saveLeague({ conn: safe, teamId: e.target.value });
            }}
          >
            <option value="">Pick your team…</option>
            {league.teams.map((t) => (
              <option key={t.team_id} value={t.team_id}>
                {t.name} ({t.owner})
              </option>
            ))}
          </select>
        </div>
        <button className="ff-link" onClick={reset}>
          ← different league
        </button>
      </div>
    );
  }

  return <LeagueReport conn={conn} league={league} teamId={teamId} onReset={reset} />;
}

// ===========================================================================
// Vegas board + parlay builder (works without a league)
// ===========================================================================
const PROP_MARKETS = [
  ["pass_yd", "Pass yds"],
  ["pass_td", "Pass TDs"],
  ["rush_yd", "Rush yds"],
  ["rec", "Receptions"],
  ["rec_yd", "Rec yds"],
  ["anytime_td", "Anytime TD"],
];

function FantasyPlayerSearch({ onSelect }) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState([]);
  const timer = useRef(null);
  useEffect(() => {
    clearTimeout(timer.current);
    if (q.trim().length < 2) {
      setHits([]);
      return undefined;
    }
    timer.current = setTimeout(() => {
      searchFantasyPlayers(q.trim()).then(setHits).catch(() => setHits([]));
    }, 200);
    return () => clearTimeout(timer.current);
  }, [q]);
  return (
    <div className="field ff-search">
      <label>Player</label>
      <input value={q} placeholder="search any NFL player" onChange={(e) => setQ(e.target.value)} />
      {hits.length > 0 && (
        <ul className="dropdown">
          {hits.map((h) => (
            <li
              key={h.player_id}
              onClick={() => {
                onSelect(h);
                setQ("");
                setHits([]);
              }}
            >
              <span>{h.player_name}</span>
              <span className="muted">
                {h.pos} · {h.team}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function PropLegForm({ onAdd }) {
  const [player, setPlayer] = useState(null);
  const [market, setMarket] = useState("rec_yd");
  const [line, setLine] = useState("");
  const [side, setSide] = useState("over");
  const [odds, setOdds] = useState("-110");
  const add = () => {
    onAdd({
      kind: "prop",
      player_id: player.player_id,
      player: player.player_name,
      market,
      line: market === "anytime_td" ? 0.5 : Number(line),
      side: market === "anytime_td" ? "over" : side,
      odds: odds === "" ? null : Number(odds),
    });
    setPlayer(null);
    setLine("");
  };
  return (
    <div className="ff-legform">
      {!player ? (
        <FantasyPlayerSearch onSelect={setPlayer} />
      ) : (
        <div className="ff-row">
          <span className="ff-grow">
            <b>{player.player_name}</b> <span className="muted">{player.pos} · {player.team}</span>
          </span>
          <button className="ff-link" onClick={() => setPlayer(null)}>
            change
          </button>
        </div>
      )}
      {player && (
        <>
          <div className="row">
            <div className="field">
              <label>Market</label>
              <select value={market} onChange={(e) => setMarket(e.target.value)}>
                {PROP_MARKETS.map(([k, l]) => (
                  <option key={k} value={k}>
                    {l}
                  </option>
                ))}
              </select>
            </div>
            {market !== "anytime_td" && (
              <>
                <div className="field">
                  <label>Line</label>
                  <input type="number" step="0.5" value={line} onChange={(e) => setLine(e.target.value)} />
                </div>
                <div className="field">
                  <label>Side</label>
                  <select value={side} onChange={(e) => setSide(e.target.value)}>
                    <option value="over">Over</option>
                    <option value="under">Under</option>
                  </select>
                </div>
              </>
            )}
            <div className="field">
              <label>Odds</label>
              <input value={odds} onChange={(e) => setOdds(e.target.value)} />
            </div>
          </div>
          <button
            className="go ff-secondary"
            onClick={add}
            disabled={market !== "anytime_td" && line === ""}
          >
            Add leg
          </button>
        </>
      )}
    </div>
  );
}

function GameLines({ games, onAdd, propsAvailable, onProps }) {
  if (!games.length) return <div className="note">No lines posted for this week yet.</div>;
  return (
    <div className="ff-list">
      {games.map((g) => {
        const key = `${g.away}@${g.home}`;
        const awaySpread = g.spread_home == null ? null : -g.spread_home;
        return (
          <div key={key} className="ff-game">
            <div className="ff-row">
              <span className="ff-grow">
                <b>
                  {g.away} @ {g.home}
                </b>
                <div className="muted ff-small">
                  {g.kickoff ? new Date(g.kickoff).toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" }) : ""}
                  {` · ${g.source}`}
                </div>
              </span>
              <span className="ff-num ff-small">
                {g.away} {fmt(g.implied_away)} · {g.home} {fmt(g.implied_home)}
                <div className="muted">implied pts</div>
              </span>
            </div>
            <div className="ff-chips">
              {g.spread_home != null && (
                <>
                  <button onClick={() => onAdd({ kind: "game", game: key, market: "spread", team: g.away, line: awaySpread, odds: -110 })}>
                    {g.away} {spreadStr(awaySpread)}
                  </button>
                  <button onClick={() => onAdd({ kind: "game", game: key, market: "spread", team: g.home, line: g.spread_home, odds: -110 })}>
                    {g.home} {spreadStr(g.spread_home)}
                  </button>
                </>
              )}
              {g.total != null && (
                <>
                  <button onClick={() => onAdd({ kind: "game", game: key, market: "total", side: "over", line: g.total, odds: -110 })}>
                    O {g.total}
                  </button>
                  <button onClick={() => onAdd({ kind: "game", game: key, market: "total", side: "under", line: g.total, odds: -110 })}>
                    U {g.total}
                  </button>
                </>
              )}
              {g.ml_away != null && (
                <button onClick={() => onAdd({ kind: "game", game: key, market: "ml", team: g.away, odds: g.ml_away })}>
                  {g.away} ML {american(g.ml_away)}
                </button>
              )}
              {g.ml_home != null && (
                <button onClick={() => onAdd({ kind: "game", game: key, market: "ml", team: g.home, odds: g.ml_home })}>
                  {g.home} ML {american(g.ml_home)}
                </button>
              )}
              {propsAvailable && g.event_id && (
                <button className="ff-chip-accent" onClick={() => onProps(g)}>
                  Props ›
                </button>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function PropsBoard({ game, onAdd, onClose }) {
  const [props, setProps] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    fetchFantasyProps(game.event_id).then(setProps).catch((e) => setError(e.message));
  }, [game.event_id]);
  return (
    <div className="card">
      <div className="ff-row">
        <h3 className="ff-h ff-grow">
          Props: {game.away} @ {game.home}
        </h3>
        <button className="ff-link" onClick={onClose}>
          close
        </button>
      </div>
      {error && <div className="error">{error}</div>}
      {!props && !error && <div className="note">Loading lines…</div>}
      {props && props.length === 0 && <div className="note">No props posted yet.</div>}
      {props &&
        props.map((p, i) => (
          <div key={i} className="ff-row">
            <div className="ff-grow">
              <b>{p.player}</b> {p.side} {p.market === "anytime_td" ? "" : p.line} {p.label}
              <div className="muted ff-small">
                model {fmt(p.model_mean)} · hit {pct(p.prob)} vs book {pct(p.book_prob)}
              </div>
            </div>
            <div className="ff-num">
              <b className={p.edge > 0 ? "ff-pos" : "ff-neg"}>{signed(p.edge * 100, 1)}%</b>
              <div>
                <button
                  className="ff-link"
                  onClick={() =>
                    onAdd({ kind: "prop", player_id: p.player_id, player: p.player, market: p.market, line: p.line, side: p.side, odds: p.odds })
                  }
                >
                  + {american(p.odds)}
                </button>
              </div>
            </div>
          </div>
        ))}
    </div>
  );
}

function legLabel(l) {
  if (l.kind === "prop") {
    const m = (PROP_MARKETS.find(([k]) => k === l.market) || [, l.market])[1];
    return l.market === "anytime_td" ? `${l.player} anytime TD` : `${l.player} ${l.side} ${l.line} ${m}`;
  }
  if (l.market === "total") return `${l.game} ${l.side} ${l.line}`;
  if (l.market === "ml") return `${l.team} ML`;
  return `${l.team} ${spreadStr(l.line)}`;
}

function VegasParlays() {
  const [odds, setOdds] = useState(null);
  const [error, setError] = useState("");
  const [legs, setLegs] = useState([]);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [propsGame, setPropsGame] = useState(null);

  useEffect(() => {
    fetchFantasyOdds().then(setOdds).catch((e) => setError(e.message));
  }, []);

  const add = (leg) => {
    setLegs((ls) => [...ls, leg].slice(0, 12));
    setResult(null);
  };
  const remove = (i) => {
    setLegs((ls) => ls.filter((_, j) => j !== i));
    setResult(null);
  };
  const grade = async () => {
    setBusy(true);
    setError("");
    try {
      setResult(await gradeFantasyParlay(legs));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="card">
        <h3 className="ff-h">
          Vegas lines {odds && <span className="muted">· Week {odds.week}</span>}
        </h3>
        <p className="muted ff-hint">
          Implied points = what the spread and total say each offense scores. Tap a line to add it to
          your parlay.
        </p>
        {!odds && !error && <div className="note">Loading lines…</div>}
        {odds && (
          <GameLines games={odds.games} onAdd={add} propsAvailable={odds.props_available} onProps={setPropsGame} />
        )}
        {odds && !odds.props_available && (
          <p className="note">Add an ODDS_API_KEY on the server to pull live player-prop lines here.</p>
        )}
      </div>

      {propsGame && <PropsBoard game={propsGame} onAdd={add} onClose={() => setPropsGame(null)} />}

      <div className="card controls">
        <h3 className="ff-h">Parlay builder</h3>
        <PropLegForm onAdd={add} />
        {legs.length > 0 && (
          <div className="ff-list" style={{ marginTop: 12 }}>
            {legs.map((l, i) => {
              const g = result && result.legs[i];
              return (
                <div key={i} className="ff-row">
                  <span className="ff-grow">
                    {legLabel(l)} <span className="muted">{american(l.odds)}</span>
                    {g && g.note && <div className="error ff-small">{g.note}</div>}
                    {g && !g.note && (
                      <div className="muted ff-small">
                        hit {pct(g.prob)}
                        {g.book_prob != null && ` vs book ${pct(g.book_prob)}`}
                        {g.model_mean != null && ` · model ${fmt(g.model_mean)}`}
                        {g.source === "market" && " · market price"}
                      </div>
                    )}
                  </span>
                  {g && g.edge != null && (
                    <b className={g.edge > 0 ? "ff-pos" : "ff-neg"}>{signed(g.edge * 100, 1)}%</b>
                  )}
                  <button className="ff-link" onClick={() => remove(i)}>
                    ✕
                  </button>
                </div>
              );
            })}
          </div>
        )}
        <button className="go" onClick={grade} disabled={busy || !legs.length}>
          {busy ? "Grading…" : `Grade ${legs.length || ""} leg${legs.length === 1 ? "" : "s"}`}
        </button>
        {error && <div className="error">{error}</div>}
        {result && (
          <div className="ff-parlay">
            <div className="ff-trade-stats">
              <span>
                Hit chance <b>{pct(result.hit_prob)}</b>
              </span>
              <span>Fair odds {american(result.fair_odds)}</span>
              {result.book_american != null && <span>Book pays {american(result.book_american)}</span>}
              {result.ev != null && (
                <span className={result.ev > 0 ? "ff-pos" : "ff-neg"}>EV {signed(result.ev * 100, 1)}%</span>
              )}
            </div>
            {result.correlations.map((c, i) => (
              <div key={i} className={`ff-small ff-corr ${c.type}`}>
                {c.type === "positive" ? "🔗" : c.type === "negative" ? "⚠️" : "ℹ️"} {c.note}
              </div>
            ))}
            <p className="note">
              Props use this week's projection adjusted for the game's Vegas total; game lines are
              priced off the market itself, so they carry no model edge.
            </p>
          </div>
        )}
      </div>
    </>
  );
}

// ===========================================================================
export default function FantasyView() {
  const [tab, setTab] = useState("league");
  return (
    <>
      <div className="tabs ff-subtabs">
        <button className={tab === "league" ? "tab active" : "tab"} onClick={() => setTab("league")}>
          My League
        </button>
        <button className={tab === "vegas" ? "tab active" : "tab"} onClick={() => setTab("vegas")}>
          Vegas & Parlays
        </button>
      </div>
      {tab === "league" ? <MyLeague /> : <VegasParlays />}
    </>
  );
}

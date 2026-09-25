"""
Fantasy football data sources: Sleeper, ESPN fantasy, and Vegas lines.

Everything here is a thin, cached HTTP layer that turns each platform's JSON
into one normalized league shape the engine (fantasy_engine.py) works on:

    {
      "platform": "sleeper" | "espn",
      "league_id", "name", "season", "week",
      "slots": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"],
      "bench": 6,
      "scoring": {"pass_yd": 0.04, "rec": 1.0, ...},   # Sleeper stat keys
      "teams": [{"team_id", "name", "owner", "players": [...],
                 "starters": [...], "wins", "losses", "ties", "points_for"}],
    }

Player ids are always Sleeper ids (ESPN rosters are mapped onto Sleeper's
player database by name/position/team) so projections, stats, and values
come from one consistent source regardless of which site the league is on.

No Supabase dependency: the fantasy side works on a fresh checkout with
nothing but network access. ESPN private leagues need the `espn_s2` and
`SWID` cookies from a logged-in browser; public leagues need neither.

Optional: set ODDS_API_KEY (the-odds-api.com) for multi-book game lines and
player-prop lines. Without it, game lines come from ESPN's scoreboard feed.
"""

import json
import os
import re
import threading
import time
from datetime import date, datetime

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, ".cache", "fantasy")

SLEEPER = "https://api.sleeper.app/v1"
SLEEPER_DATA = "https://api.sleeper.com"
ESPN_FANTASY = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ODDS_API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")
TIMEOUT = 20


class SourceError(Exception):
    """A platform call failed in a way the user can act on (bad id, private
    league without cookies, platform down). The API turns it into a 4xx/503."""


# ---------------------------------------------------------------------------
# Caching: in-memory TTL, plus disk for the big Sleeper player database
# ---------------------------------------------------------------------------
_mem = {}
_mem_lock = threading.Lock()


def _cached(key, ttl, fn):
    now = time.time()
    with _mem_lock:
        hit = _mem.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = fn()
    with _mem_lock:
        _mem[key] = (now, value)
    return value


def clear_cache():
    with _mem_lock:
        _mem.clear()


def _get(url, params=None, cookies=None, what="request"):
    try:
        res = requests.get(url, params=params, cookies=cookies, timeout=TIMEOUT,
                           headers={"User-Agent": "money-from-a-baby/1.0"})
    except requests.RequestException as exc:
        raise SourceError(f"{what} failed: {exc}") from exc
    if res.status_code in (401, 403):
        raise SourceError(f"{what}: access denied ({res.status_code}).")
    if res.status_code == 404:
        raise SourceError(f"{what}: not found.")
    if not res.ok:
        raise SourceError(f"{what} failed ({res.status_code}).")
    try:
        return res.json()
    except ValueError as exc:
        raise SourceError(f"{what}: unexpected response.") from exc


# ---------------------------------------------------------------------------
# Team abbreviations. Sleeper's are canonical here (they key DEF players).
# ---------------------------------------------------------------------------
_ABBR_FIX = {"WSH": "WAS", "JAC": "JAX", "LA": "LAR", "OAK": "LV", "SD": "LAC", "STL": "LAR"}


def canon_abbr(abbr):
    a = (abbr or "").upper().strip()
    return _ABBR_FIX.get(a, a)


# ESPN fantasy proTeamId -> abbreviation.
ESPN_PRO_TEAMS = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN",
    8: "DET", 9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LAR",
    15: "MIA", 16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI",
    22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS",
    29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU",
}
ESPN_POSITIONS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DEF"}
# ESPN lineupSlotId -> our slot names (IDP / TQB slots are ignored).
ESPN_SLOTS = {
    0: "QB", 2: "RB", 3: "WRRB_FLEX", 4: "WR", 5: "REC_FLEX", 6: "TE",
    7: "SUPER_FLEX", 16: "DEF", 17: "K", 23: "FLEX",
}
ESPN_BENCH, ESPN_IR = 20, 21
# ESPN scoring statId -> Sleeper stat key (the ones that matter for skill
# players; K/DEF fall back to Sleeper's own standard scoring).
ESPN_STAT_KEYS = {
    3: "pass_yd", 4: "pass_td", 19: "pass_2pt", 20: "pass_int",
    24: "rush_yd", 25: "rush_td", 26: "rush_2pt",
    42: "rec_yd", 43: "rec_td", 44: "rec_2pt", 53: "rec", 72: "fum_lost",
}


# ---------------------------------------------------------------------------
# Name matching (ESPN -> Sleeper)
# ---------------------------------------------------------------------------
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def name_key(name):
    s = (name or "").lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9 ]", "", s.replace("-", " "))
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Sleeper: NFL state, player database, projections, stats, trending
# ---------------------------------------------------------------------------
def nfl_state():
    """{'season': '2026', 'week': 3, 'season_type': 'regular'} from Sleeper,
    with a date-based fallback so an outage never blocks the whole report."""
    def load():
        try:
            s = _get(f"{SLEEPER}/state/nfl", what="Sleeper NFL state")
            return {
                "season": str(s.get("season") or s.get("league_season")),
                "week": int(s.get("display_week") or s.get("week") or 1),
                "season_type": s.get("season_type") or "regular",
            }
        except SourceError:
            return estimate_state()
    return _cached("state", 1800, load)


def estimate_state(today=None):
    """Regular season starts the Thursday after Labor Day (first Monday of
    September); week N starts on the Tuesday before its Thursday game."""
    today = today or date.today()
    year = today.year if today.month >= 3 else today.year - 1
    sept1 = date(year, 9, 1)
    labor_day = sept1.toordinal() + ((0 - sept1.weekday()) % 7)
    kickoff = labor_day + 3
    days = today.toordinal() - (kickoff - 2)
    week = max(1, min(18, days // 7 + 1)) if days >= 0 else 1
    return {"season": str(year), "week": week,
            "season_type": "regular" if days >= 0 else "pre"}


def sleeper_players():
    """Sleeper's full NFL player database, trimmed to fantasy positions.

    It's ~5MB and Sleeper asks callers to fetch it at most once a day, so it's
    cached on disk for 24h (and in memory for the process)."""
    def load():
        path = os.path.join(CACHE_DIR, "sleeper_players.json")
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < 86400:
            with open(path) as f:
                return json.load(f)
        raw = _get(f"{SLEEPER}/players/nfl", what="Sleeper player database")
        trimmed = {}
        for pid, p in raw.items():
            pos = p.get("position")
            if pos not in FANTASY_POSITIONS:
                continue
            name = p.get("full_name") or " ".join(
                x for x in (p.get("first_name"), p.get("last_name")) if x)
            trimmed[pid] = {
                "name": name,
                "pos": pos,
                "team": canon_abbr(p.get("team")) if p.get("team") else None,
                "age": p.get("age"),
                "years_exp": p.get("years_exp"),
                "injury": p.get("injury_status"),
                "status": p.get("status"),
                "rank": p.get("search_rank"),
            }
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(trimmed, f)
        return trimmed
    return _cached("players", 3600, load)


def search_players(query, limit=10):
    """Name autocomplete over the Sleeper database, best-known players first."""
    q = name_key(query)
    if len(q) < 2:
        return []
    hits = [(pid, p) for pid, p in sleeper_players().items()
            if q in name_key(p["name"]) and (p.get("team") or p["pos"] == "DEF")]
    hits.sort(key=lambda x: (x[1].get("rank") or 10**6, x[1]["name"]))
    return [{"player_id": pid, "player_name": p["name"], "pos": p["pos"], "team": p["team"]}
            for pid, p in hits[:limit]]


def _position_params(season_type="regular"):
    return [("season_type", season_type)] + [("position[]", p) for p in FANTASY_POSITIONS]


def sleeper_week_projections(season, week, season_type="regular"):
    """{player_id: {stat: value}} -- Sleeper's weekly projected stat lines."""
    def load():
        rows = _get(f"{SLEEPER_DATA}/projections/nfl/{season}/{week}",
                    params=_position_params(season_type),
                    what=f"Sleeper projections (week {week})")
        return {str(r["player_id"]): r.get("stats") or {}
                for r in rows or [] if r.get("player_id")}
    return _cached(f"proj:{season}:{week}:{season_type}", 6 * 3600, load)


def sleeper_week_stats(season, week, season_type="regular"):
    """{player_id: {stat: value}} -- actual box-score lines for a finished week."""
    def load():
        rows = _get(f"{SLEEPER_DATA}/stats/nfl/{season}/{week}",
                    params=_position_params(season_type),
                    what=f"Sleeper stats (week {week})")
        return {str(r["player_id"]): r.get("stats") or {}
                for r in rows or [] if r.get("player_id")}
    return _cached(f"stats:{season}:{week}:{season_type}", 3 * 3600, load)


def sleeper_trending(kind="add", hours=48, limit=40):
    def load():
        try:
            rows = _get(f"{SLEEPER}/players/nfl/trending/{kind}",
                        params={"lookback_hours": hours, "limit": limit},
                        what="Sleeper trending")
        except SourceError:
            return []
        return [{"player_id": str(r["player_id"]), "count": r.get("count", 0)}
                for r in rows or []]
    return _cached(f"trend:{kind}:{hours}:{limit}", 1800, load)


# ---------------------------------------------------------------------------
# Sleeper leagues
# ---------------------------------------------------------------------------
def sleeper_user_leagues(username, season):
    user = _get(f"{SLEEPER}/user/{username}", what="Sleeper user lookup")
    if not user or not user.get("user_id"):
        raise SourceError(f"No Sleeper user named '{username}'.")
    leagues = _get(f"{SLEEPER}/user/{user['user_id']}/leagues/nfl/{season}",
                   what="Sleeper leagues") or []
    return {
        "user_id": user["user_id"],
        "display_name": user.get("display_name") or username,
        "leagues": [{
            "league_id": lg["league_id"],
            "name": lg.get("name"),
            "season": lg.get("season"),
            "teams": lg.get("total_rosters"),
            "status": lg.get("status"),
        } for lg in leagues],
    }


def sleeper_league(league_id):
    lg = _get(f"{SLEEPER}/league/{league_id}", what="Sleeper league")
    if not lg:
        raise SourceError(f"Sleeper league {league_id} not found.")
    rosters = _get(f"{SLEEPER}/league/{league_id}/rosters", what="Sleeper rosters") or []
    users = _get(f"{SLEEPER}/league/{league_id}/users", what="Sleeper league users") or []
    return normalize_sleeper(lg, rosters, users)


def normalize_sleeper(lg, rosters, users):
    by_user = {u["user_id"]: u for u in users}
    positions = lg.get("roster_positions") or []
    slots = [p for p in positions if p not in ("BN", "IR", "TAXI")]
    teams = []
    for r in rosters:
        u = by_user.get(r.get("owner_id")) or {}
        meta = u.get("metadata") or {}
        s = r.get("settings") or {}
        owner = u.get("display_name") or f"Team {r.get('roster_id')}"
        teams.append({
            "team_id": str(r.get("roster_id")),
            "owner_id": r.get("owner_id"),
            "owner": owner,
            "name": meta.get("team_name") or owner,
            "players": [str(p) for p in (r.get("players") or [])],
            "starters": [str(p) for p in (r.get("starters") or []) if p and p != "0"],
            "reserve": [str(p) for p in (r.get("reserve") or [])],
            "wins": s.get("wins", 0), "losses": s.get("losses", 0),
            "ties": s.get("ties", 0),
            "points_for": (s.get("fpts") or 0) + (s.get("fpts_decimal") or 0) / 100,
        })
    return {
        "platform": "sleeper",
        "league_id": str(lg.get("league_id")),
        "name": lg.get("name"),
        "season": str(lg.get("season")),
        "slots": slots,
        "bench": positions.count("BN"),
        "scoring": {k: float(v) for k, v in (lg.get("scoring_settings") or {}).items()
                    if isinstance(v, (int, float))},
        "teams": teams,
    }


# ---------------------------------------------------------------------------
# ESPN fantasy leagues
# ---------------------------------------------------------------------------
def espn_league(league_id, season, espn_s2=None, swid=None):
    cookies = {}
    if espn_s2 and swid:
        cookies = {"espn_s2": espn_s2, "SWID": swid if swid.startswith("{") else "{%s}" % swid}
    url = f"{ESPN_FANTASY}/seasons/{season}/segments/0/leagues/{league_id}"
    params = [("view", v) for v in ("mTeam", "mRoster", "mSettings", "mStatus")]
    try:
        data = _get(url, params=params, cookies=cookies or None, what="ESPN league")
    except SourceError as exc:
        if "access denied" in str(exc) and not cookies:
            raise SourceError(
                "This ESPN league is private. Add your espn_s2 and SWID cookies "
                "(from espn.com while logged in) to connect it.") from exc
        raise
    return normalize_espn(data, sleeper_players())


def _espn_scoring(settings):
    items = ((settings or {}).get("scoringSettings") or {}).get("scoringItems") or []
    out = {}
    for it in items:
        key = ESPN_STAT_KEYS.get(it.get("statId"))
        if key is not None:
            out[key] = float(it.get("points", 0))
    return out


def build_name_index(players):
    idx = {}
    for pid, p in players.items():
        idx.setdefault((name_key(p["name"]), p["pos"]), []).append(pid)
    return idx


def match_espn_player(espn_player, players, index):
    """ESPN player object -> Sleeper player id (None if no confident match)."""
    pos = ESPN_POSITIONS.get(espn_player.get("defaultPositionId"))
    team = ESPN_PRO_TEAMS.get(espn_player.get("proTeamId"))
    if pos == "DEF":
        return team if team in players else None
    cands = index.get((name_key(espn_player.get("fullName")), pos)) or []
    if len(cands) > 1 and team:
        same = [c for c in cands if players[c].get("team") == team]
        cands = same or cands
    return cands[0] if cands else None


def normalize_espn(data, players):
    settings = data.get("settings") or {}
    counts = ((settings.get("rosterSettings") or {}).get("lineupSlotCounts")) or {}
    slots, bench = [], 0
    for sid, n in sorted(((int(k), v) for k, v in counts.items()), key=lambda x: x[0]):
        if sid == ESPN_BENCH:
            bench = n
        elif sid in ESPN_SLOTS:
            slots += [ESPN_SLOTS[sid]] * int(n)

    members = {m.get("id"): m for m in data.get("members") or []}
    index = build_name_index(players)
    teams, unmatched = [], []
    for t in data.get("teams") or []:
        owner_ids = t.get("owners") or []
        m = members.get(owner_ids[0]) if owner_ids else None
        owner = (m or {}).get("displayName") or " ".join(
            x for x in ((m or {}).get("firstName"), (m or {}).get("lastName")) if x
        ) or f"Team {t.get('id')}"
        name = t.get("name") or " ".join(
            x for x in (t.get("location"), t.get("nickname")) if x) or owner
        ids, starters, reserve = [], [], []
        for e in ((t.get("roster") or {}).get("entries")) or []:
            pl = ((e.get("playerPoolEntry") or {}).get("player")) or {}
            pid = match_espn_player(pl, players, index)
            if pid is None:
                if pl.get("fullName"):
                    unmatched.append(pl["fullName"])
                continue
            ids.append(pid)
            slot = e.get("lineupSlotId")
            if slot == ESPN_IR:
                reserve.append(pid)
            elif slot != ESPN_BENCH:
                starters.append(pid)
        rec = ((t.get("record") or {}).get("overall")) or {}
        teams.append({
            "team_id": str(t.get("id")),
            "owner_id": owner_ids[0] if owner_ids else None,
            "owner": owner,
            "name": name,
            "players": ids,
            "starters": starters,
            "reserve": reserve,
            "wins": rec.get("wins", 0), "losses": rec.get("losses", 0),
            "ties": rec.get("ties", 0),
            "points_for": rec.get("pointsFor", 0),
        })
    status = data.get("status") or {}
    return {
        "platform": "espn",
        "league_id": str(data.get("id")),
        "name": settings.get("name") or f"ESPN league {data.get('id')}",
        "season": str(data.get("seasonId")),
        "week": data.get("scoringPeriodId") or status.get("currentMatchupPeriod"),
        "slots": slots,
        "bench": bench,
        "scoring": _espn_scoring(settings),
        "teams": teams,
        "unmatched_players": unmatched,
    }


# ---------------------------------------------------------------------------
# Vegas lines
# ---------------------------------------------------------------------------
_DETAILS = re.compile(r"^\s*([A-Z]{2,4})\s+(-?\d+(?:\.\d+)?)\s*$")


def american_to_prob(odds):
    odds = float(odds)
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def _game_from_lines(home, away, spread_home, total, ml_home=None, ml_away=None,
                     kickoff=None, source="espn", event_id=None):
    g = {
        "event_id": event_id, "home": home, "away": away, "kickoff": kickoff,
        "spread_home": spread_home, "total": total,
        "ml_home": ml_home, "ml_away": ml_away, "source": source,
    }
    if total is not None and spread_home is not None:
        g["implied_home"] = round(total / 2 - spread_home / 2, 2)
        g["implied_away"] = round(total / 2 + spread_home / 2, 2)
    if ml_home is not None and ml_away is not None:
        ph, pa = american_to_prob(ml_home), american_to_prob(ml_away)
        g["win_prob_home"] = round(ph / (ph + pa), 4)
    return g


def parse_espn_scoreboard(data):
    games = []
    for ev in data.get("events") or []:
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): canon_abbr((c.get("team") or {}).get("abbreviation"))
                 for c in comp.get("competitors") or []}
        home, away = teams.get("home"), teams.get("away")
        if not home or not away:
            continue
        odds = (comp.get("odds") or [None])[0] or {}
        total = odds.get("overUnder")
        spread_home = None
        m = _DETAILS.match(odds.get("details") or "")
        if m:
            fav, pts = canon_abbr(m.group(1)), abs(float(m.group(2)))
            spread_home = -pts if fav == home else pts
        elif (odds.get("details") or "").upper() == "EVEN":
            spread_home = 0.0
        ml_home = ((odds.get("homeTeamOdds") or {}).get("moneyLine"))
        ml_away = ((odds.get("awayTeamOdds") or {}).get("moneyLine"))
        status = (((comp.get("status") or {}).get("type")) or {}).get("state")
        g = _game_from_lines(home, away, spread_home,
                             float(total) if total is not None else None,
                             ml_home, ml_away, kickoff=ev.get("date"),
                             source=((odds.get("provider") or {}).get("name")) or "espn",
                             event_id=ev.get("id"))
        g["state"] = status
        games.append(g)
    return games


def _odds_api_key():
    return os.environ.get("ODDS_API_KEY", "").strip()


_ODDS_API_NAMES = None


def _odds_api_abbr(full_name):
    global _ODDS_API_NAMES
    if _ODDS_API_NAMES is None:
        from_names = {
            "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
            "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
            "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
            "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
            "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
            "Kansas City Chiefs": "KC", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR",
            "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
            "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
            "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
            "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
            "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
        }
        _ODDS_API_NAMES = from_names
    return _ODDS_API_NAMES.get(full_name, full_name)


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def parse_odds_api_games(rows):
    """The Odds API /odds response -> our game dicts, using the median line
    across books (a cheap consensus that ignores one-off stale books)."""
    games = []
    for ev in rows or []:
        home, away = _odds_api_abbr(ev.get("home_team")), _odds_api_abbr(ev.get("away_team"))
        spreads, totals, mlh, mla = [], [], [], []
        for bk in ev.get("bookmakers") or []:
            for mk in bk.get("markets") or []:
                for o in mk.get("outcomes") or []:
                    team = _odds_api_abbr(o.get("name"))
                    if mk.get("key") == "spreads" and team == home and o.get("point") is not None:
                        spreads.append(float(o["point"]))
                    elif mk.get("key") == "totals" and o.get("name") == "Over":
                        totals.append(float(o["point"]))
                    elif mk.get("key") == "h2h":
                        (mlh if team == home else mla).append(float(o["price"]))
        g = _game_from_lines(home, away, _median(spreads), _median(totals),
                             _median(mlh), _median(mla), kickoff=ev.get("commence_time"),
                             source=f"consensus of {len(ev.get('bookmakers') or [])} books",
                             event_id=ev.get("id"))
        g["state"] = "pre"
        games.append(g)
    return games


def game_lines(season, week, season_type="regular"):
    """This week's games with spread / total / moneyline / implied team totals.

    Uses The Odds API consensus when ODDS_API_KEY is set (it only lists games
    that haven't kicked off), otherwise ESPN's scoreboard odds."""
    def load():
        key = _odds_api_key()
        if key:
            try:
                rows = _get(f"{ODDS_API}/odds", params={
                    "apiKey": key, "regions": "us", "oddsFormat": "american",
                    "markets": "h2h,spreads,totals"}, what="Odds API")
                games = parse_odds_api_games(rows)
                if games:
                    return games
            except SourceError:
                pass
        stype = {"pre": 1, "regular": 2, "post": 3}.get(season_type, 2)
        data = _get(ESPN_SCOREBOARD, params={"week": week, "seasontype": stype,
                                             "dates": season}, what="ESPN scoreboard")
        return parse_espn_scoreboard(data)
    return _cached(f"lines:{season}:{week}:{season_type}", 900, load)


PROP_MARKETS = {
    "player_pass_yds": "pass_yd",
    "player_pass_tds": "pass_td",
    "player_rush_yds": "rush_yd",
    "player_receptions": "rec",
    "player_reception_yds": "rec_yd",
    "player_anytime_td": "anytime_td",
}


def parse_odds_api_props(event):
    """One event's props -> [{player, market, line, over, under, book}], best
    price per side across books."""
    best = {}
    for bk in event.get("bookmakers") or []:
        for mk in bk.get("markets") or []:
            market = PROP_MARKETS.get(mk.get("key"))
            if not market:
                continue
            for o in mk.get("outcomes") or []:
                player = o.get("description")
                side = (o.get("name") or "").lower()
                if market == "anytime_td":
                    side, point = "over", 0.5
                else:
                    point = o.get("point")
                if not player or point is None or side not in ("over", "under"):
                    continue
                k = (player, market, float(point))
                row = best.setdefault(k, {"player": player, "market": market,
                                          "line": float(point), "over": None, "under": None})
                price = float(o.get("price"))
                if row[side] is None or price > row[side]:
                    row[side] = price
    return list(best.values())


def event_props(event_id):
    key = _odds_api_key()
    if not key:
        raise SourceError("Player-prop lines need an ODDS_API_KEY (the-odds-api.com).")

    def load():
        ev = _get(f"{ODDS_API}/events/{event_id}/odds", params={
            "apiKey": key, "regions": "us", "oddsFormat": "american",
            "markets": ",".join(PROP_MARKETS)}, what="Odds API props")
        return parse_odds_api_props(ev)
    return _cached(f"props:{event_id}", 900, load)


def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

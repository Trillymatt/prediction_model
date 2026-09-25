"""Offline tests for the fantasy advisor (no network: sources are faked).

    python -m pytest tests/test_fantasy.py -q
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fantasy_engine as fe  # noqa: E402
import fantasy_sources as src  # noqa: E402

TEAMS = ["KC", "BUF", "PHI", "SF", "DAL", "DET", "MIA", "CIN", "BAL", "GB", "LAR", "HOU"]
PPR = {"pass_yd": 0.04, "pass_td": 4, "pass_int": -1, "rush_yd": 0.1, "rush_td": 6,
       "rec": 1.0, "rec_yd": 0.1, "rec_td": 6, "fum_lost": -2}
SLOTS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]


def _make_players():
    rng = random.Random(7)
    players, proj_base = {}, {}
    pid = 1000
    for pos, count in (("QB", 24), ("RB", 48), ("WR", 60), ("TE", 20), ("K", 14)):
        for i in range(count):
            pid += 1
            team = TEAMS[i % len(TEAMS)]
            players[str(pid)] = {"name": f"{pos} Player{i}", "pos": pos, "team": team,
                                 "age": 22 + (i % 10), "injury": None, "status": "Active"}
            proj_base[str(pid)] = (pos, i, rng.random())
    for t in TEAMS:
        players[t] = {"name": f"{t} Defense", "pos": "DEF", "team": t, "age": None, "injury": None}
        proj_base[t] = ("DEF", TEAMS.index(t), rng.random())
    return players, proj_base


def _stat_line(pos, tier, noise, week, actual=False):
    scale = max(0.15, 1 - tier / 40)
    jitter = (1 + (noise - 0.5) * 0.3) if actual else 1.0
    if pos == "QB":
        s = {"pass_att": 34 * scale, "pass_yd": 260 * scale * jitter, "pass_td": 1.9 * scale * jitter,
             "pass_int": 0.7, "rush_att": 4, "rush_yd": 18 * scale, "rush_td": 0.15}
    elif pos == "RB":
        s = {"rush_att": 16 * scale, "rush_yd": 75 * scale * jitter, "rush_td": 0.6 * scale * jitter,
             "rec_tgt": 4 * scale, "rec": 3 * scale, "rec_yd": 22 * scale}
    elif pos in ("WR", "TE"):
        m = 1.0 if pos == "WR" else 0.7
        s = {"rec_tgt": 8 * scale * m, "rec": 5.5 * scale * m, "rec_yd": 70 * scale * m * jitter,
             "rec_td": 0.5 * scale * m * jitter}
    else:
        s = {}
    s = {k: round(v, 2) for k, v in s.items()}
    base_pts = 8 * scale + 2
    s.update(pts_ppr=base_pts, pts_half_ppr=base_pts, pts_std=base_pts)
    if actual:
        s.update(gp=1, off_snp=50, tm_off_snp=65)
    return s


@pytest.fixture(scope="module")
def world():
    players, base = _make_players()
    current = 4
    projs = {w: {pid: _stat_line(*base[pid], w) for pid in base} for w in range(current, 18)}
    stats = {w: {pid: _stat_line(*base[pid], w, actual=True) for pid in base} for w in range(1, current)}

    # A buy-low: huge volume, no production. A sell-high: TD-fueled.
    buy_id = next(p for p, v in players.items() if v["name"] == "WR Player3")
    sell_id = next(p for p, v in players.items() if v["name"] == "RB Player6")
    for w in stats:
        stats[w][buy_id].update(rec_tgt=13, rec=4, rec_yd=31, rec_td=0)
        stats[w][sell_id].update(rush_att=9, rush_yd=38, rush_td=2, rec_tgt=1, rec=1, rec_yd=5)

    # 10-team league, snake-ish draft by projection.
    uni0 = fe.build_universe(players, PPR, projs, {}, current)
    order = sorted(uni0.values(), key=lambda p: -p["proj_pg"])
    rosters = [[] for _ in range(10)]
    need = {"QB": 2, "RB": 4, "WR": 6, "TE": 2, "K": 1, "DEF": 1}
    counts = [dict.fromkeys(need, 0) for _ in range(10)]
    rnd = 0
    pool = list(order)
    while any(sum(c.values()) < 16 for c in counts) and rnd < 40:
        seq = range(10) if rnd % 2 == 0 else range(9, -1, -1)
        for i in seq:
            pick = next((p for p in pool if counts[i][p["pos"]] < need[p["pos"]]), None)
            if pick:
                pool.remove(pick)
                rosters[i].append(pick["id"])
                counts[i][pick["pos"]] += 1
        rnd += 1
    lg = {"league_id": "L1", "name": "Test League", "season": "2026",
          "roster_positions": SLOTS + ["BN"] * 7, "scoring_settings": PPR}
    sl_rosters = [{"roster_id": i + 1, "owner_id": f"u{i}", "players": r, "starters": r[:9],
                   "settings": {"wins": i % 3, "losses": 3 - i % 3, "fpts": 300 + i}}
                  for i, r in enumerate(rosters)]
    users = [{"user_id": f"u{i}", "display_name": f"owner{i}",
              "metadata": {"team_name": f"Team {i}"}} for i in range(10)]
    league = src.normalize_sleeper(lg, sl_rosters, users)
    games = [src._game_from_lines(TEAMS[i], TEAMS[i + 1], -3.5 - i, 44.5 + i, -180, 150,
                                  event_id=str(i)) for i in range(0, 12, 2)]
    return {"players": players, "projs": projs, "stats": stats, "league": league,
            "games": games, "current": current, "buy_id": buy_id, "sell_id": sell_id}


@pytest.fixture(scope="module")
def ctx(world):
    lg = world["league"]
    rostered = {p for t in lg["teams"] for p in t["players"]}
    uni = fe.build_universe(world["players"], lg["scoring"], world["projs"], world["stats"],
                            world["current"], rostered=rostered)
    repl, per_team = fe.replacement_levels(uni, lg["slots"], len(lg["teams"]))
    fe.attach_values(uni, repl)
    needs = fe.team_needs(lg, uni, repl, per_team)
    return {"uni": uni, "repl": repl, "per_team": per_team, "needs": needs, "rostered": rostered}


def test_scoring_uses_league_settings():
    line = {"pass_yd": 300, "pass_td": 2, "rec": 5, "rec_yd": 50, "pts_ppr": 99}
    assert fe.fantasy_points(line, PPR, "QB") == pytest.approx(12 + 8 + 5 + 5)
    six = dict(PPR, pass_td=6)
    assert fe.fantasy_points(line, six, "QB") == pytest.approx(12 + 12 + 5 + 5)
    te = dict(PPR, bonus_rec_te=0.5)
    assert fe.fantasy_points(line, te, "TE") - fe.fantasy_points(line, te, "WR") == pytest.approx(2.5)
    assert fe.fantasy_points({"pts_std": 9, "pts_ppr": 9}, PPR, "K") == 9


def test_normalize_sleeper(world):
    lg = world["league"]
    assert lg["slots"] == SLOTS and lg["bench"] == 7 and len(lg["teams"]) == 10
    assert lg["teams"][0]["name"] == "Team 0" and lg["teams"][0]["owner"] == "owner0"


def test_normalize_espn_maps_players(world):
    players = world["players"]
    wr = next(p for p, v in players.items() if v["name"] == "WR Player0")
    data = {
        "id": 99, "seasonId": 2026, "scoringPeriodId": 4,
        "settings": {"name": "ESPN L",
                     "rosterSettings": {"lineupSlotCounts": {"0": 1, "2": 2, "4": 2, "6": 1,
                                                             "23": 1, "16": 1, "17": 1, "20": 6}},
                     "scoringSettings": {"scoringItems": [{"statId": 53, "points": 0.5},
                                                          {"statId": 42, "points": 0.1},
                                                          {"statId": 4, "points": 6}]}},
        "members": [{"id": "{A}", "displayName": "matt"}],
        "teams": [{"id": 1, "name": "Matt's Team", "owners": ["{A}"],
                   "record": {"overall": {"wins": 2, "losses": 1, "pointsFor": 350.5}},
                   "roster": {"entries": [
                       {"lineupSlotId": 4, "playerPoolEntry": {"player": {
                           "fullName": "WR Player0", "defaultPositionId": 3, "proTeamId": 12}}},
                       {"lineupSlotId": 16, "playerPoolEntry": {"player": {
                           "fullName": "Chiefs D/ST", "defaultPositionId": 16, "proTeamId": 12}}},
                       {"lineupSlotId": 20, "playerPoolEntry": {"player": {
                           "fullName": "Nobody Known", "defaultPositionId": 2, "proTeamId": 2}}},
                   ]}}],
    }
    lg = src.normalize_espn(data, players)
    assert lg["slots"] == ["QB", "RB", "RB", "WR", "WR", "TE", "DEF", "K", "FLEX"]
    assert lg["scoring"] == {"rec": 0.5, "rec_yd": 0.1, "pass_td": 6.0}
    t = lg["teams"][0]
    assert t["players"] == [wr, "KC"] and t["starters"] == [wr, "KC"]
    assert t["owner"] == "matt" and lg["unmatched_players"] == ["Nobody Known"]


def test_espn_scoreboard_parsing():
    data = {"events": [{"id": "1", "date": "2026-09-27T17:00Z", "competitions": [{
        "competitors": [{"homeAway": "home", "team": {"abbreviation": "WSH"}},
                        {"homeAway": "away", "team": {"abbreviation": "KC"}}],
        "odds": [{"details": "KC -3.5", "overUnder": 47.5, "provider": {"name": "DraftKings"},
                  "homeTeamOdds": {"moneyLine": 150}, "awayTeamOdds": {"moneyLine": -180}}],
        "status": {"type": {"state": "pre"}}}]}]}
    g = src.parse_espn_scoreboard(data)[0]
    assert g["home"] == "WAS" and g["away"] == "KC" and g["spread_home"] == 3.5
    assert g["implied_away"] == pytest.approx(25.5) and g["implied_home"] == pytest.approx(22.0)
    assert 0.35 < g["win_prob_home"] < 0.42


def test_estimate_state():
    from datetime import date
    s = src.estimate_state(date(2026, 9, 25))
    assert s["season"] == "2026" and s["week"] == 3


def test_optimal_lineup_fills_flex():
    pts = {"a": (20, "RB"), "b": (15, "RB"), "c": (14, "RB"), "d": (18, "WR"), "e": (9, "WR"),
           "f": (10, "TE"), "g": (8, "QB")}
    total, lineup = fe.optimal_lineup(list(pts), ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"], pts)
    assert dict(lineup)["FLEX"] == "c" and total == pytest.approx(94)


def test_values_and_needs(ctx):
    uni, repl = ctx["uni"], ctx["repl"]
    assert repl["QB"] > 0 and repl["RB"] > 0
    top_rb = max((p for p in uni.values() if p["pos"] == "RB"), key=lambda p: p["ros_ppg"])
    assert top_rb["value"] > 0
    for tid, n in ctx["needs"].items():
        assert set(n["positions"]) == {"QB", "RB", "WR", "TE"}
        assert all(1 <= v["rank"] <= 10 for v in n["positions"].values())


def test_trade_finder_is_win_win_ish(world, ctx):
    lg = world["league"]
    trades = fe.find_trades(lg, ctx["uni"], ctx["needs"], "1")
    assert trades, "expected at least one trade idea"
    for t in trades:
        assert t["my_gain_ppg"] >= 0.3 and t["their_gain_ppg"] >= -0.25
        assert t["partner_id"] != "1" and t["pitch"]
        ev = fe.evaluate_trade(lg, ctx["uni"], ctx["needs"], ctx["repl"], ctx["per_team"], "1",
                               t["partner_id"], [p["id"] for p in t["give"]],
                               [p["id"] for p in t["get"]])
        assert ev["my_gain_ppg"] == pytest.approx(t["my_gain_ppg"], abs=0.01)


def test_evaluate_trade_rejects_wrong_players(world, ctx):
    lg = world["league"]
    with pytest.raises(ValueError):
        fe.evaluate_trade(lg, ctx["uni"], ctx["needs"], ctx["repl"], ctx["per_team"], "1", "2",
                          [lg["teams"][1]["players"][0]], [])


def test_buy_low_sell_high(world, ctx):
    lg = world["league"]
    owners = {p: t["team_id"] for t in lg["teams"] for p in t["players"]}
    bs = fe.buy_sell(ctx["uni"], ctx["repl"], owners, "1", limit=50)
    buys = {b["id"] for b in bs["buy_low"]}
    sells = {s["id"] for s in bs["sell_high"]}
    if owners.get(world["buy_id"]) != "1":
        assert world["buy_id"] in buys
    assert world["sell_id"] in sells
    assert world["buy_id"] not in sells and world["sell_id"] not in buys


def test_start_sit_uses_vegas_and_injuries(world, ctx):
    lg = world["league"]
    team = dict(lg["teams"][0])
    uni = dict(ctx["uni"])
    hurt = team["players"][0]
    uni[hurt] = dict(uni[hurt], injury="Out")
    ss = fe.start_sit(lg, uni, team, world["games"])
    assert len(ss["lineup"]) == len(SLOTS)
    started = {s["player"]["id"] for s in ss["lineup"] if s["player"]}
    assert hurt not in started
    row = next(r for r in ss["bench"] + [s["player"] for s in ss["lineup"] if s["player"]]
               if r["id"] != hurt and r["pos"] in ("QB", "RB", "WR", "TE"))
    assert row["vegas_mult"] != 1.0 or row["team"] not in fe.lines_by_team(world["games"])


def test_vegas_multiplier_direction():
    hi = {"implied": 30, "opp_implied": 17, "spread": -10}
    lo = {"implied": 16, "opp_implied": 27, "spread": 10}
    assert fe.vegas_multiplier("WR", hi, 22.5)[0] > 1 > fe.vegas_multiplier("WR", lo, 22.5)[0]
    assert fe.vegas_multiplier("DEF", hi, 22.5)[0] > 1 > fe.vegas_multiplier("DEF", lo, 22.5)[0]


def test_waivers(world, ctx):
    lg = world["league"]
    w = fe.waiver_targets(lg, ctx["uni"], lg["teams"][9], ctx["rostered"])
    for t in w["targets"]:
        assert t["id"] not in ctx["rostered"]


def test_parlay_grading(world):
    players, projs, games = world["players"], world["projs"], world["games"]
    qb = next(p for p, v in players.items() if v["name"] == "QB Player0")
    wr = next(p for p, v in players.items() if v["name"] == "WR Player0")
    g0 = games[0]
    legs = [
        {"kind": "prop", "player_id": qb, "market": "pass_yd", "line": 249.5, "side": "over", "odds": -115},
        {"kind": "prop", "player_id": wr, "market": "rec_yd", "line": 60.5, "side": "over", "odds": -110},
        {"kind": "prop", "player_id": wr, "market": "anytime_td", "line": 0.5, "side": "over", "odds": 150},
        {"kind": "game", "game": f"{g0['away']}@{g0['home']}", "market": "total", "side": "over",
         "odds": -110},
        {"kind": "game", "game": f"{g0['away']}@{g0['home']}", "market": "spread", "team": g0["home"],
         "odds": -110},
    ]
    r = fe.grade_parlay(legs, projs[world["current"]], players, games)
    assert all(0 < leg["prob"] < 1 for leg in r["legs"])
    assert r["legs"][3]["prob"] == pytest.approx(0.5, abs=1e-6)
    assert r["legs"][4]["prob"] == pytest.approx(0.5, abs=1e-6)
    assert r["hit_prob"] == pytest.approx(
        __import__("math").prod(leg["prob"] for leg in r["legs"]), rel=1e-3)
    assert r["book_decimal"] > 1 and r["ev"] is not None
    assert any(c["type"] == "positive" for c in r["correlations"])


def test_props_board(world):
    players, projs, games = world["players"], world["projs"], world["games"]
    props = src.parse_odds_api_props({"bookmakers": [
        {"markets": [{"key": "player_reception_yds", "outcomes": [
            {"name": "Over", "description": "WR Player0", "point": 55.5, "price": -110},
            {"name": "Under", "description": "WR Player0", "point": 55.5, "price": -120}]}]},
        {"markets": [{"key": "player_reception_yds", "outcomes": [
            {"name": "Over", "description": "WR Player0", "point": 55.5, "price": -105}]}]},
    ]})
    assert props == [{"player": "WR Player0", "market": "rec_yd", "line": 55.5,
                      "over": -105.0, "under": -120.0}]
    board = fe.grade_props_board(props, projs[world["current"]], players, games)
    assert {b["side"] for b in board} == {"over", "under"}
    assert board[0]["edge"] >= board[-1]["edge"]


def test_analyze_league_end_to_end(world, monkeypatch):
    monkeypatch.setattr(src, "nfl_state", lambda: {"season": "2026", "week": world["current"],
                                                   "season_type": "regular"})
    monkeypatch.setattr(src, "sleeper_players", lambda: world["players"])
    monkeypatch.setattr(src, "sleeper_week_projections", lambda s, w: world["projs"].get(w, {}))
    monkeypatch.setattr(src, "sleeper_week_stats", lambda s, w: world["stats"].get(w, {}))
    monkeypatch.setattr(src, "game_lines", lambda s, w: world["games"])
    monkeypatch.setattr(src, "sleeper_trending", lambda: [])
    r = fe.analyze_league(world["league"], "3")
    assert r["league"]["scoring"] == "PPR" and r["league"]["week"] == world["current"]
    assert len(r["power_rankings"]) == 10
    assert r["start_sit"]["lineup"] and r["my_team"]["roster"]
    for key in ("trades", "buy_sell", "waivers", "games"):
        assert key in r


# ---------------------------------------------------------------------------
# Full stack through the real source functions + FastAPI routes, faking only
# the HTTP layer.
# ---------------------------------------------------------------------------
def _fake_http(world):
    players = world["players"]
    raw_players = {pid: {"full_name": p["name"], "position": p["pos"],
                         "team": "WAS" if p["team"] == "WAS" else p["team"],
                         "age": p["age"], "injury_status": p["injury"], "search_rank": i}
                   for i, (pid, p) in enumerate(players.items())}
    lg = world["league"]
    raw_league = {"league_id": "123", "name": "Test League", "season": "2026",
                  "roster_positions": SLOTS + ["BN"] * 7, "scoring_settings": PPR}
    rosters = [{"roster_id": int(t["team_id"]), "owner_id": f"u{i}", "players": t["players"],
                "starters": t["starters"], "settings": {"wins": 1, "losses": 2, "fpts": 250}}
               for i, t in enumerate(lg["teams"])]
    users = [{"user_id": f"u{i}", "display_name": f"owner{i}", "metadata": {"team_name": f"Team {i}"}}
             for i in range(len(lg["teams"]))]
    g0 = world["games"][0]
    board = {"events": [{"id": "e1", "date": "2026-09-27T17:00Z", "competitions": [{
        "competitors": [{"homeAway": "home", "team": {"abbreviation": g0["home"]}},
                        {"homeAway": "away", "team": {"abbreviation": g0["away"]}}],
        "odds": [{"details": f"{g0['home']} -3.5", "overUnder": 47.5}],
        "status": {"type": {"state": "pre"}}}]}]}

    def fake_get(url, params=None, cookies=None, what="request"):
        if url.endswith("/state/nfl"):
            return {"season": "2026", "week": world["current"], "season_type": "regular"}
        if url.endswith("/players/nfl"):
            return raw_players
        if "/trending/" in url:
            return []
        if "/projections/nfl/" in url or "/stats/nfl/" in url:
            wk = int(url.rsplit("/", 1)[1])
            src_ = world["projs"] if "/projections/" in url else world["stats"]
            return [{"player_id": pid, "stats": s} for pid, s in src_.get(wk, {}).items()]
        if url.endswith("/league/123"):
            return raw_league
        if url.endswith("/league/123/rosters"):
            return rosters
        if url.endswith("/league/123/users"):
            return users
        if url.endswith("/user/mattk"):
            return {"user_id": "u0", "display_name": "mattk"}
        if "/user/u0/leagues/nfl/" in url:
            return [{"league_id": "123", "name": "Test League", "season": "2026",
                     "total_rosters": 10, "status": "in_season"}]
        if url == src.ESPN_SCOREBOARD:
            return board
        raise src.SourceError(f"{what}: not found.", status=404)
    return fake_get


def test_api_routes(world, monkeypatch, tmp_path):
    monkeypatch.setenv("SUPABASE_URL", os.environ.get("SUPABASE_URL", "https://x.supabase.co"))
    monkeypatch.setenv("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", "x.y.z"))
    try:
        from fastapi.testclient import TestClient
        import api
    except Exception as exc:  # noqa: BLE001 - backend deps not installed
        pytest.skip(f"api.py not importable here: {exc}")
    monkeypatch.setattr(src, "_get", _fake_http(world))
    monkeypatch.setattr(src, "CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    src.clear_cache()
    api._league_cache.clear()
    c = TestClient(api.app)

    r = c.get("/api/fantasy/sleeper/leagues", params={"username": "mattk"})
    assert r.status_code == 200 and r.json()["leagues"][0]["league_id"] == "123"

    body = {"platform": "sleeper", "league_id": "123"}
    r = c.post("/api/fantasy/league", json=body)
    assert r.status_code == 200, r.text
    assert len(r.json()["teams"]) == 10 and r.json()["scoring"] == "PPR"

    r = c.post("/api/fantasy/analyze", json={**body, "team_id": "2"})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["start_sit"]["lineup"] and rep["power_rankings"]
    assert rep["games"] and rep["games"][0]["implied_home"] == pytest.approx(25.5)

    if rep["trades"]:
        t = rep["trades"][0]
        r = c.post("/api/fantasy/trade", json={**body, "team_id": "2", "partner_id": t["partner_id"],
                                               "give": [p["id"] for p in t["give"]],
                                               "get": [p["id"] for p in t["get"]]})
        assert r.status_code == 200, r.text
        assert r.json()["my_gain_ppg"] == pytest.approx(t["my_gain_ppg"], abs=0.01)

    r = c.post("/api/fantasy/analyze", json={**body, "team_id": "999"})
    assert r.status_code == 400
    assert c.post("/api/fantasy/league", json={"platform": "yahoo", "league_id": "1"}).status_code == 400

    r = c.get("/api/fantasy/odds")
    assert r.status_code == 200 and r.json()["props_available"] is False

    r = c.get("/api/fantasy/players", params={"q": "wr player1"})
    hits = r.json()["players"]
    assert hits and all(h["pos"] == "WR" for h in hits)

    g = rep["games"][0]
    r = c.post("/api/fantasy/parlay", json={"legs": [
        {"kind": "prop", "player_id": hits[0]["player_id"], "market": "rec", "line": 4.5,
         "side": "over", "odds": -120},
        {"kind": "game", "game": f"{g['away']}@{g['home']}", "market": "ml", "team": g["home"]},
    ]})
    assert r.status_code == 200, r.text
    assert 0 < r.json()["hit_prob"] < 1

    r = c.get("/api/fantasy/props", params={"event_id": "e1"})
    assert r.status_code == 503 and "ODDS_API_KEY" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Regression tests for bugs fixed in the review pass
# ---------------------------------------------------------------------------
def test_bad_odds_rejected_not_crash(world):
    leg = {"kind": "prop", "player_id": "KC", "market": "rec", "line": 4.5, "odds": 0}
    with pytest.raises(ValueError, match="American odds"):
        fe.grade_parlay([leg], {}, world["players"], [])
    for bad in ([], [{"kind": "prop", "market": "rec", "line": 1}],
                [{"kind": "prop", "player_id": "1", "market": "bogus", "line": 1}],
                [{"kind": "prop", "player_id": "1", "market": "rec", "line": "abc"}],
                [{"kind": "game", "game": "A@B", "market": "spread"}],
                [{"kind": "nope"}]):
        with pytest.raises(ValueError):
            fe.validate_legs(bad)


def test_merge_odds_keeps_espn_week():
    espn = [src._game_from_lines("KC", "BUF", -3.0, 47.0, event_id="e1"),
            src._game_from_lines("DAL", "PHI", 1.0, 44.0, event_id="e2")]
    espn[0]["state"], espn[1]["state"] = "pre", "post"
    odds = [src._game_from_lines("KC", "BUF", -4.5, 49.5, event_id="o1"),
            src._game_from_lines("DAL", "PHI", 3.0, 41.0, event_id="o2"),
            # next week's KC game must not leak into this week
            src._game_from_lines("KC", "DEN", -7.0, 45.0, event_id="o3")]
    merged = src.merge_odds(espn, odds)
    assert len(merged) == 2
    assert merged[0]["spread_home"] == -4.5 and merged[0]["odds_event_id"] == "o1"
    assert merged[1]["spread_home"] == 1.0 and "odds_event_id" not in merged[1]
    assert fe.lines_by_team(merged)["KC"]["opponent"] == "BUF"


def test_anytime_td_yes_no_mapping():
    props = src.parse_odds_api_props({"bookmakers": [{"markets": [{"key": "player_anytime_td", "outcomes": [
        {"name": "Yes", "description": "A B", "price": 150},
        {"name": "No", "description": "A B", "price": -200}]}]}]})
    assert props == [{"player": "A B", "market": "anytime_td", "line": 0.5,
                      "over": 150.0, "under": -200.0}]


def test_week_18_keeps_current_week(world, monkeypatch):
    projs = {18: world["projs"][world["current"]]}
    monkeypatch.setattr(src, "nfl_state", lambda: {"season": "2026", "week": 18,
                                                   "season_type": "regular"})
    monkeypatch.setattr(src, "sleeper_players", lambda: world["players"])
    monkeypatch.setattr(src, "sleeper_week_projections", lambda s, w: projs.get(w, {}))
    monkeypatch.setattr(src, "sleeper_week_stats", lambda s, w: {})
    monkeypatch.setattr(src, "game_lines", lambda s, w: [])
    ctx = fe.prepare(world["league"])
    assert ctx["week"] == 18
    assert any(p["proj_week"] > 0 for p in ctx["uni"].values())


def test_player_outlook(world):
    players, projs = world["players"], world["projs"][world["current"]]
    wr = fe.find_player(players, "wr player0", "", "WR")
    assert players[wr]["name"] == "WR Player0"
    out = fe.player_outlook(players, projs, world["games"], wr, world["current"])
    assert out["has_projection"] and out["stats"]["rec_yd"] > 0
    assert out["fantasy"]["ppr"]["proj"] > out["fantasy"]["std"]["proj"]
    assert 0 < out["anytime_td_prob"] < 1


def test_api_validation_and_player(world, monkeypatch, tmp_path):
    monkeypatch.setenv("SUPABASE_URL", os.environ.get("SUPABASE_URL", "https://x.supabase.co"))
    monkeypatch.setenv("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", "x.y.z"))
    try:
        from fastapi.testclient import TestClient
        import api
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"api.py not importable here: {exc}")
    fake = _fake_http(world)
    calls = []
    monkeypatch.setattr(src, "_get", lambda url, **kw: calls.append(url) or fake(url, **kw))
    monkeypatch.setattr(src, "CACHE_DIR", str(tmp_path))
    src.clear_cache()
    api._league_cache.clear()
    c = TestClient(api.app)

    assert c.get("/api/fantasy/sleeper/leagues", params={"username": "../league/1"}).status_code == 400
    assert c.get("/api/fantasy/props", params={"event_id": "../../sports"}).status_code == 400
    assert c.post("/api/fantasy/league", json={"platform": "sleeper", "league_id": "123",
                                               "season": "../x"}).status_code == 400
    body = {"platform": "sleeper", "league_id": "123"}
    assert c.post("/api/fantasy/analyze", json={**body, "team_id": "1", "week": 99}).status_code == 400
    r = c.post("/api/fantasy/parlay", json={"legs": [{"kind": "prop", "player_id": "KC",
                                                       "market": "rec", "line": 1, "odds": 50}]})
    assert r.status_code == 400 and "American odds" in r.json()["detail"]

    t1 = world["league"]["teams"][0]
    r = c.post("/api/fantasy/trade", json={**body, "team_id": "1", "partner_id": "2",
                                           "give": [t1["players"][0]], "get": [t1["players"][0]]})
    assert r.status_code == 400

    # refresh bypasses the 5-minute league cache
    c.post("/api/fantasy/league", json=body)
    n = sum(u.endswith("/league/123") for u in calls)
    c.post("/api/fantasy/league", json=body)
    assert sum(u.endswith("/league/123") for u in calls) == n
    c.post("/api/fantasy/analyze", json={**body, "team_id": "1", "refresh": True})
    assert sum(u.endswith("/league/123") for u in calls) == n + 1

    r = c.get("/api/fantasy/player", params={"name": "WR Player0", "team": "Kansas City Chiefs",
                                             "position": "WR"})
    assert r.status_code == 200 and r.json()["available"] and r.json()["has_projection"]
    r = c.get("/api/fantasy/player", params={"name": "Some Linebacker", "position": "LB"})
    assert r.json()["available"] is False
    r = c.get("/api/fantasy/player", params={"name": "Nobody Here", "position": "WR"})
    assert r.json()["available"] is False


def test_parlay_long_shot_not_zeroed_and_conflicts_flagged(world):
    players, projs, games = world["players"], world["projs"][world["current"]], world["games"]
    qb = next(p for p, v in players.items() if v["name"] == "QB Player0")
    g0 = games[0]
    key = f"{g0['away']}@{g0['home']}"
    # A near-impossible leg (QB receptions) plus a normal one: the parlay is
    # tiny but not zero, and it still gets fair odds.
    r = fe.grade_parlay([
        {"kind": "prop", "player_id": qb, "market": "rec", "line": 5.5, "side": "over", "odds": 900},
        {"kind": "game", "game": key, "market": "total", "side": "over", "odds": -110},
    ], projs, players, games)
    assert 0 < r["legs"][0]["prob"] < 1e-4 and r["hit_prob"] > 0 and r["fair_odds"] is not None

    r = fe.grade_parlay([
        {"kind": "game", "game": key, "market": "ml", "team": g0["home"], "odds": -150},
        {"kind": "game", "game": key, "market": "ml", "team": g0["away"], "odds": 130},
    ], projs, players, games)
    assert r["hit_prob"] == 0 and any(c["type"] == "conflict" for c in r["correlations"])

    r = fe.grade_parlay([
        {"kind": "game", "game": key, "market": "total", "side": "over", "line": 44, "odds": -110},
        {"kind": "game", "game": key, "market": "total", "side": "under", "line": 50, "odds": -110},
    ], projs, players, games)
    assert r["hit_prob"] > 0 and not any(c["type"] == "conflict" for c in r["correlations"])

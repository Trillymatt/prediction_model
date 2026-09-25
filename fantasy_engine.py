"""
Fantasy football advisor engine.

Pure functions over the normalized league shape from fantasy_sources.py plus
Sleeper's weekly projections/stats and this week's Vegas lines:

  build_universe()   every relevant player's weekly projection, rest-of-season
                     points per game, season results, usage, and value over
                     replacement -- all scored with *your league's* settings
  team_needs()       each team's strength rank at QB/RB/WR/TE and its
                     tradeable depth, so trades target real needs
  find_trades()      1-for-1 / 2-for-1 / 1-for-2 / 2-for-2 offers that improve
                     your starting lineup AND theirs (so they'll accept)
  evaluate_trade()   grade any proposed trade (incoming requests too)
  buy_sell()         buy-low / sell-high from usage vs. production and
                     projections vs. results
  start_sit()        this week's optimal lineup, adjusted by Vegas implied
                     team totals, spreads, and injury status
  waiver_targets()   free agents who'd crack your lineup (+ trending adds)
  grade_parlay()     prop and game-line legs -> hit probability, fair odds,
                     EV, and same-game correlation warnings

analyze_league() stitches it all together for the API.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

SKILL = ("QB", "RB", "WR", "TE")
SLOT_ELIGIBLE = {
    "QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"},
    "K": {"K"}, "DEF": {"DEF"},
    "WRRB_FLEX": {"RB", "WR"}, "REC_FLEX": {"WR", "TE"},
    "FLEX": {"RB", "WR", "TE"}, "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
}
# Fill the most restrictive slots first; with nested eligibility this makes
# the greedy fill optimal.
SLOT_PRIORITY = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 0, "DEF": 0,
                 "WRRB_FLEX": 1, "REC_FLEX": 1, "FLEX": 2, "SUPER_FLEX": 3}
END_WEEK = 17
INJURY_MULT = {"out": 0.0, "ir": 0.0, "pup": 0.0, "sus": 0.0, "na": 0.0,
               "cov": 0.0, "doubtful": 0.25, "questionable": 0.85}
NOT_PLAYING = {"out", "ir", "pup", "sus", "na", "cov"}


def _supported_slots(slots):
    return [s for s in slots if s in SLOT_ELIGIBLE]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _preset_points(stats, scoring):
    rec = (scoring or {}).get("rec", 1.0)
    key = "pts_ppr" if rec >= 0.75 else "pts_half_ppr" if rec >= 0.25 else "pts_std"
    return float(stats.get(key) or stats.get("pts_ppr") or stats.get("pts_std") or 0.0)


_CORE = ("pass_yd", "pass_td", "rush_yd", "rush_td", "rec_yd", "rec_td")


def _dot(stats, scoring, pos):
    pts = 0.0
    for key, mult in scoring.items():
        if key.startswith("bonus_rec_"):
            if key == f"bonus_rec_{pos.lower()}":
                pts += float(stats.get("rec") or 0) * mult
            continue
        val = stats.get(key)
        if isinstance(val, (int, float)):
            pts += val * mult
    return pts


def fantasy_points(stats, scoring, pos):
    """A stat line scored with the league's settings. K/DEF (and leagues with
    no recognizable offensive scoring) use Sleeper's precomputed totals."""
    if not stats:
        return 0.0
    if pos in ("K", "DEF") or not any(k in (scoring or {}) for k in _CORE):
        return _preset_points(stats, scoring)
    return _dot(stats, scoring, pos)


def _part_points(stats, scoring, keys, pos):
    """Points from one slice of the stat line (passing, rushing, receiving)."""
    sub = {k: v for k, v in scoring.items()
           if k in keys or (k == f"bonus_rec_{pos.lower()}" and "rec" in keys)}
    return _dot(stats, sub, pos)


PASS_KEYS = ("pass_yd", "pass_td", "pass_int", "pass_2pt", "pass_cmp", "pass_inc", "pass_att")
RUSH_KEYS = ("rush_yd", "rush_td", "rush_2pt", "rush_att")
REC_KEYS = ("rec", "rec_yd", "rec_td", "rec_2pt")


def _played(stats, pos):
    if not stats:
        return False
    if (stats.get("gp") or 0) > 0 or (stats.get("off_snp") or 0) > 0:
        return True
    return pos in ("K", "DEF") and any(v for v in stats.values() if isinstance(v, (int, float)))


# ---------------------------------------------------------------------------
# Universe: projections + results + value for every relevant player
# ---------------------------------------------------------------------------
def build_universe(players, scoring, week_projs, week_stats, current_week,
                   end_week=END_WEEK, rostered=()):
    """players: Sleeper db {pid: {name,pos,team,age,injury,...}}
    week_projs: {week: {pid: stats}} for current_week..end_week
    week_stats: {week: {pid: stats}} for completed weeks"""
    rostered = set(rostered)
    relevant = set(rostered)
    for d in list(week_projs.values()) + list(week_stats.values()):
        relevant.update(d.keys())
    uni = {}
    for pid in relevant:
        p = players.get(pid)
        if not p or p.get("pos") not in SLOT_ELIGIBLE:
            continue
        pos = p["pos"]
        ros = [fantasy_points(week_projs.get(w, {}).get(pid), scoring, pos)
               for w in range(current_week, end_week + 1)]
        proj_week = ros[0] if ros else 0.0
        playing = [x for x in ros if x > 0.5]
        proj_pg = sum(playing) / len(playing) if playing else 0.0

        games, pts, opp_weeks = [], [], []
        for w in sorted(week_stats):
            s = week_stats[w].get(pid)
            if not _played(s, pos):
                continue
            games.append(w)
            pts.append(fantasy_points(s, scoring, pos))
            opp_weeks.append(s)
        n = len(games)
        ppg = sum(pts) / n if n else 0.0
        w_act = min(0.35, 0.06 * n) if n and proj_pg > 0 else 0.0
        ros_ppg = (1 - w_act) * proj_pg + w_act * ppg if proj_pg > 0 else 0.0
        if p.get("injury") and p["injury"].lower() in ("ir", "pup", "sus") and not playing:
            ros_ppg = 0.0
        uni[pid] = {
            "id": pid, "name": p.get("name") or pid, "pos": pos, "team": p.get("team"),
            "age": p.get("age"), "injury": p.get("injury"),
            "proj_week": round(proj_week, 2),
            "proj_pg": round(proj_pg, 2),
            "ros_ppg": round(ros_ppg, 2),
            "ros_games": len(playing),
            "games": n, "ppg": round(ppg, 2),
            "last3_ppg": round(sum(pts[-3:]) / len(pts[-3:]), 2) if pts else 0.0,
            "weekly": [{"week": w, "pts": round(x, 1)} for w, x in zip(games, pts)],
            "_stat_weeks": opp_weeks,
            "rostered": pid in rostered,
        }
    _attach_usage(uni, scoring)
    return uni


def _attach_usage(uni, scoring):
    """Expected fantasy points from opportunity (pass attempts, carries,
    targets) at each position's league-wide per-opportunity rate -- the
    backbone of buy-low / sell-high."""
    totals = {}
    for pl in uni.values():
        if pl["pos"] not in SKILL:
            continue
        t = totals.setdefault(pl["pos"], {"pa": 0, "pp": 0.0, "ra": 0, "rp": 0.0, "tg": 0, "tp": 0.0})
        for s in pl["_stat_weeks"]:
            t["pa"] += s.get("pass_att") or 0
            t["pp"] += _part_points(s, scoring, PASS_KEYS, pl["pos"])
            t["ra"] += s.get("rush_att") or 0
            t["rp"] += _part_points(s, scoring, RUSH_KEYS, pl["pos"])
            t["tg"] += s.get("rec_tgt") or 0
            t["tp"] += _part_points(s, scoring, REC_KEYS, pl["pos"])
    rates = {pos: {"pass": t["pp"] / t["pa"] if t["pa"] else 0.0,
                   "rush": t["rp"] / t["ra"] if t["ra"] else 0.0,
                   "tgt": t["tp"] / t["tg"] if t["tg"] else 0.0}
             for pos, t in totals.items()}
    for pl in uni.values():
        weeks = pl.pop("_stat_weeks")
        r = rates.get(pl["pos"])
        if not r or not weeks:
            pl.update(xfp_pg=None, opp_pg=0.0, opp_last3=0.0, td_share=0.0, snap_pct=None)
            continue

        def opp(s):
            return ((s.get("pass_att") or 0) * (0.25 if pl["pos"] == "QB" else 1)
                    + (s.get("rush_att") or 0) + (s.get("rec_tgt") or 0))

        xfp = [(s.get("pass_att") or 0) * r["pass"] + (s.get("rush_att") or 0) * r["rush"]
               + (s.get("rec_tgt") or 0) * r["tgt"] for s in weeks]
        tds = sum((s.get("pass_td") or 0) * scoring.get("pass_td", 4)
                  + ((s.get("rush_td") or 0) + (s.get("rec_td") or 0)) * scoring.get("rush_td", 6)
                  for s in weeks)
        total_pts = pl["ppg"] * pl["games"]
        snaps = [s["off_snp"] / s["tm_off_snp"] for s in weeks
                 if s.get("off_snp") and s.get("tm_off_snp")]
        opps = [opp(s) for s in weeks]
        pl["xfp_pg"] = round(sum(xfp) / len(xfp), 2)
        pl["opp_pg"] = round(sum(opps) / len(opps), 1)
        pl["opp_last3"] = round(sum(opps[-3:]) / len(opps[-3:]), 1)
        pl["td_share"] = round(tds / total_pts, 2) if total_pts > 0 else 0.0
        pl["snap_pct"] = round(sum(snaps) / len(snaps), 2) if snaps else None


def replacement_levels(uni, slots, n_teams):
    """Per-position replacement ppg: the best player left once every team has
    filled its starting lineup (flex slots go to the best remaining eligible
    players league-wide). Also returns how many starters each position
    supplies per team -- the 'k' used for team-strength comparisons."""
    slots = _supported_slots(slots)
    by_pos = {pos: sorted((p["ros_ppg"] for p in uni.values() if p["pos"] == pos), reverse=True)
              for pos in SLOT_ELIGIBLE if pos in ("QB", "RB", "WR", "TE", "K", "DEF")}
    taken = {pos: 0 for pos in by_pos}
    for slot in sorted(set(slots), key=lambda s: SLOT_PRIORITY[s]):
        need = slots.count(slot) * n_teams
        elig = SLOT_ELIGIBLE[slot]
        for _ in range(need):
            best, best_pos = -1.0, None
            for pos in elig:
                lst = by_pos.get(pos, [])
                if taken[pos] < len(lst) and lst[taken[pos]] > best:
                    best, best_pos = lst[taken[pos]], pos
            if best_pos is None:
                break
            taken[best_pos] += 1
    repl = {pos: (lst[taken[pos]] if taken[pos] < len(lst) else 0.0)
            for pos, lst in by_pos.items()}
    per_team = {pos: taken[pos] / max(1, n_teams) for pos in taken}
    return repl, per_team


def attach_values(uni, repl):
    for p in uni.values():
        par = max(0.0, p["ros_ppg"] - repl.get(p["pos"], 0.0))
        p["value"] = round(par * p["ros_games"], 1)
        p["par_pg"] = round(p["ros_ppg"] - repl.get(p["pos"], 0.0), 2)


# ---------------------------------------------------------------------------
# Lineups
# ---------------------------------------------------------------------------
def optimal_lineup(pids, slots, pts):
    """Greedy slot fill (most restrictive first). pts: {pid: points}.
    Returns (total, [(slot, pid|None)])."""
    slots = _supported_slots(slots)
    pool = sorted((pid for pid in pids if pid in pts), key=lambda x: -pts[x][0])
    used, lineup, total = set(), [], 0.0
    for slot in sorted(slots, key=lambda s: SLOT_PRIORITY[s]):
        elig = SLOT_ELIGIBLE[slot]
        pick = next((pid for pid in pool if pid not in used and pts[pid][1] in elig), None)
        if pick is not None:
            used.add(pick)
            total += pts[pick][0]
        lineup.append((slot, pick))
    return total, lineup


def _ros_pts(uni, pids):
    return {pid: (uni[pid]["ros_ppg"], uni[pid]["pos"]) for pid in pids if pid in uni}


def lineup_ppg(uni, pids, slots):
    return optimal_lineup(pids, slots, _ros_pts(uni, pids))[0]


# ---------------------------------------------------------------------------
# Team needs
# ---------------------------------------------------------------------------
def _strength(vals, k):
    vals = sorted(vals, reverse=True)
    whole = int(k)
    s = sum(vals[:whole])
    if k - whole > 0 and len(vals) > whole:
        s += (k - whole) * vals[whole]
    return s


def team_needs(league, uni, repl, per_team):
    teams = league["teams"]
    n = len(teams)
    strengths = {}
    for t in teams:
        for pos in SKILL:
            k = per_team.get(pos, 0)
            if k <= 0:
                continue
            vals = [uni[p]["ros_ppg"] for p in t["players"] if p in uni and uni[p]["pos"] == pos]
            strengths.setdefault(pos, {})[t["team_id"]] = _strength(vals, k)
    out = {}
    for t in teams:
        tid = t["team_id"]
        profile = {}
        for pos, by_team in strengths.items():
            ordered = sorted(by_team.values(), reverse=True)
            mine = by_team[tid]
            rank = ordered.index(mine) + 1
            avg = sum(ordered) / len(ordered) if ordered else 0.0
            k = per_team[pos]
            mine_sorted = sorted((uni[p] for p in t["players"]
                                  if p in uni and uni[p]["pos"] == pos),
                                 key=lambda p: -p["ros_ppg"])
            depth = [p for p in mine_sorted[math.ceil(k):]
                     if p["ros_ppg"] > repl.get(pos, 0) + 0.5]
            if rank > n * 2 / 3 or (avg and mine < avg * 0.88):
                grade = "need"
            elif rank <= max(1, n / 3) or depth:
                grade = "strong"
            else:
                grade = "ok"
            profile[pos] = {
                "rank": rank, "of": n, "grade": grade,
                "strength": round(mine, 1), "league_avg": round(avg, 1),
                "depth": [{"id": p["id"], "name": p["name"], "ros_ppg": p["ros_ppg"]} for p in depth],
            }
        needs = sorted((p for p in profile if profile[p]["grade"] == "need"),
                       key=lambda p: profile[p]["strength"] - profile[p]["league_avg"])
        surplus = [p for p in profile if profile[p]["depth"]]
        out[tid] = {"positions": profile, "needs": needs, "surplus": surplus}
    return out


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------
def _logistic(x):
    return 1 / (1 + math.exp(-x))


def _trade_math(league, uni, mine, theirs, give, get):
    slots = league["slots"]
    me_after = [p for p in mine if p not in give] + list(get)
    them_after = [p for p in theirs if p not in get] + list(give)
    d_me = lineup_ppg(uni, me_after, slots) - lineup_ppg(uni, mine, slots)
    d_them = lineup_ppg(uni, them_after, slots) - lineup_ppg(uni, theirs, slots)
    v_give = sum(uni[p]["value"] for p in give if p in uni)
    v_get = sum(uni[p]["value"] for p in get if p in uni)
    # Managers accept trades that help their lineup and don't look like a
    # value loss on paper; both terms matter.
    accept = _logistic(1.1 * d_them + 0.035 * (v_give - v_get) + 0.2)
    return d_me, d_them, v_give, v_get, accept


def _brief(uni, pid):
    p = uni.get(pid) or {"id": pid, "name": pid, "pos": "?", "team": None}
    return {"id": pid, "name": p["name"], "pos": p["pos"], "team": p.get("team"),
            "ros_ppg": p.get("ros_ppg"), "value": p.get("value"), "injury": p.get("injury")}


def _pitch(uni, give, get, my_needs, their_needs, partner):
    give_pos = sorted({uni[p]["pos"] for p in give})
    get_pos = sorted({uni[p]["pos"] for p in get})
    bits = []
    helped = [pos for pos in give_pos if pos in their_needs["needs"]]
    if helped:
        pr = their_needs["positions"][helped[0]]
        bits.append(f"{partner} ranks {pr['rank']}/{pr['of']} at {helped[0]}; "
                    f"{' + '.join(uni[p]['name'] for p in give)} would start for them.")
    deep = [pos for pos in get_pos if pos in their_needs["surplus"]]
    if deep:
        bits.append(f"They're deep at {deep[0]}, so {' + '.join(uni[p]['name'] for p in get)} "
                    f"is expendable to them.")
    fills = [pos for pos in get_pos if pos in my_needs["needs"]]
    if fills:
        mr = my_needs["positions"][fills[0]]
        bits.append(f"You rank {mr['rank']}/{mr['of']} at {fills[0]} -- this plugs that hole.")
    return " ".join(bits) or "Both starting lineups project higher after the swap."


def find_trades(league, uni, needs, my_team_id, per_partner=3, limit=15, pool=7):
    teams = {t["team_id"]: t for t in league["teams"]}
    me = teams[my_team_id]
    mine = [p for p in me["players"] if p in uni]

    def candidates(pids):
        c = [p for p in pids if p in uni and uni[p]["pos"] in SKILL and uni[p]["value"] > 0]
        return sorted(c, key=lambda p: -uni[p]["value"])[:pool]

    my_c = candidates(mine)
    results = []
    for tid, t in teams.items():
        if tid == my_team_id:
            continue
        theirs = [p for p in t["players"] if p in uni]
        their_c = candidates(theirs)
        shapes = [(1, 1), (2, 1), (1, 2), (2, 2)]
        found = []
        for ng, nr in shapes:
            gl = my_c if ng == 1 or len(my_c) <= 6 else my_c[:6]
            rl = their_c if nr == 1 or len(their_c) <= 6 else their_c[:6]
            for give in combinations(gl, ng):
                for get in combinations(rl, nr):
                    d_me, d_them, v_give, v_get, accept = _trade_math(
                        league, uni, mine, theirs, give, get)
                    if d_me < 0.3 or d_them < -0.25 or accept < 0.35:
                        continue
                    found.append({
                        "partner_id": tid, "partner": t["name"], "partner_owner": t["owner"],
                        "give": list(give), "get": list(get),
                        "my_gain_ppg": round(d_me, 2), "their_gain_ppg": round(d_them, 2),
                        "value_give": round(v_give, 1), "value_get": round(v_get, 1),
                        "accept_prob": round(accept, 2),
                        "score": d_me * accept,
                    })
        found.sort(key=lambda x: -x["score"])
        kept, seen_give, seen_get = [], set(), set()
        for f in found:
            give_k, get_k = frozenset(f["give"]), frozenset(f["get"])
            if give_k & seen_give or get_k & seen_get:
                continue
            seen_give |= give_k
            seen_get |= get_k
            kept.append(f)
            if len(kept) >= per_partner:
                break
        results.extend(kept)
    results.sort(key=lambda x: -x["score"])
    # Don't let one player headline every idea.
    offered, picked = {}, []
    for r in results:
        if any(offered.get(p, 0) >= 3 for p in r["give"]):
            continue
        for p in r["give"]:
            offered[p] = offered.get(p, 0) + 1
        picked.append(r)
    out = []
    for r in picked[:limit]:
        r["pitch"] = _pitch(uni, r["give"], r["get"], needs[my_team_id], needs[r["partner_id"]],
                            r["partner"])
        r["give"] = [_brief(uni, p) for p in r["give"]]
        r["get"] = [_brief(uni, p) for p in r["get"]]
        r["verdict"] = _verdict(r["my_gain_ppg"], r["their_gain_ppg"],
                                r["value_get"] - r["value_give"], r["accept_prob"])
        r.pop("score")
        out.append(r)
    return out


def _verdict(d_me, d_them, value_edge, accept=1.0):
    if d_me > 0.3 and accept < 0.2:
        return "they'll likely decline"
    if d_me > 0.3 and d_them > 0.3:
        return "win-win"
    if d_me > 0.3 and value_edge > 20:
        return "you win (they may balk)"
    if d_me > 0.3:
        return "good for you"
    if d_me > -0.3:
        return "roughly even"
    return "hurts your lineup"


def evaluate_trade(league, uni, needs, repl, per_team, my_team_id, partner_id, give, get):
    """Grade a specific trade -- one you're considering or one you've been
    offered. give = players leaving your team, get = players arriving."""
    teams = {t["team_id"]: t for t in league["teams"]}
    if my_team_id not in teams or partner_id not in teams or my_team_id == partner_id:
        raise ValueError("Pick your team and a different trade partner.")
    if not give and not get:
        raise ValueError("Add at least one player to the trade.")
    mine = [p for p in teams[my_team_id]["players"] if p in uni]
    theirs = [p for p in teams[partner_id]["players"] if p in uni]
    bad = [p for p in give if p not in mine] + [p for p in get if p not in theirs]
    if bad:
        raise ValueError(f"Players not on the expected rosters: {', '.join(bad)}")
    d_me, d_them, v_give, v_get, accept = _trade_math(league, uni, mine, theirs, give, get)

    swapped = {my_team_id: [p for p in mine if p not in give] + list(get),
               partner_id: [p for p in theirs if p not in get] + list(give)}
    after = team_needs({"teams": [{**t, "players": swapped.get(t["team_id"], t["players"])}
                                  for t in league["teams"]]}, uni, repl, per_team)
    rank_change = {pos: {"before": v["rank"],
                         "after": after[my_team_id]["positions"][pos]["rank"]}
                   for pos, v in needs[my_team_id]["positions"].items()}
    return {
        "give": [_brief(uni, p) for p in give], "get": [_brief(uni, p) for p in get],
        "partner": teams[partner_id]["name"],
        "my_gain_ppg": round(d_me, 2), "their_gain_ppg": round(d_them, 2),
        "value_give": round(v_give, 1), "value_get": round(v_get, 1),
        "accept_prob": round(accept, 2),
        "verdict": _verdict(d_me, d_them, v_get - v_give, accept),
        "rank_change": rank_change,
        "pitch": _pitch(uni, give, get, needs[my_team_id], needs[partner_id],
                        teams[partner_id]["name"]) if give and get else "",
    }


# ---------------------------------------------------------------------------
# Buy low / sell high
# ---------------------------------------------------------------------------
def buy_sell(uni, repl, owners, my_team_id, limit=10):
    buys, sells = [], []
    for p in uni.values():
        if p["pos"] not in SKILL or p["games"] < 2 or p["xfp_pg"] is None:
            continue
        if (p.get("injury") or "").lower() in NOT_PLAYING:
            continue
        r = repl.get(p["pos"], 0.0)
        luck = p["ppg"] - p["xfp_pg"]
        outlook = p["proj_pg"] - p["ppg"]
        trend = p["opp_last3"] - p["opp_pg"]
        owner = owners.get(p["id"])
        reasons_b, reasons_s = [], []
        if luck < -1.5:
            reasons_b.append(f"Usage says {p['xfp_pg']:.1f} pts/game, he's scoring {p['ppg']:.1f}"
                             f" -- production should catch up to the volume.")
        if outlook > 1.5:
            reasons_b.append(f"Projected {p['proj_pg']:.1f}/game rest of season vs "
                             f"{p['ppg']:.1f} so far.")
        if trend > 1.5:
            reasons_b.append(f"Opportunities trending up ({p['opp_last3']:.1f} last 3 vs "
                             f"{p['opp_pg']:.1f} season).")
        if p["last3_ppg"] < p["ppg"] - 3 and luck < 0:
            reasons_b.append("Recent cold stretch is likely driving his price down.")
        if luck > 1.5:
            reasons_s.append(f"Scoring {p['ppg']:.1f}/game on volume worth {p['xfp_pg']:.1f}"
                             f" -- efficiency this high rarely holds.")
        if p["td_share"] >= 0.4 and p["ppg"] > 8:
            reasons_s.append(f"{int(p['td_share'] * 100)}% of his points are touchdowns.")
        if outlook < -1.5:
            reasons_s.append(f"Projected {p['proj_pg']:.1f}/game rest of season vs "
                             f"{p['ppg']:.1f} so far.")
        if trend < -1.5:
            reasons_s.append(f"Opportunities trending down ({p['opp_last3']:.1f} last 3 vs "
                             f"{p['opp_pg']:.1f} season).")
        if p["pos"] == "RB" and (p.get("age") or 0) >= 28:
            reasons_s.append(f"Age-{p['age']} running back -- sell before the decline shows.")
        buy_score = -luck * 0.6 + outlook * 0.8 + trend * 0.3
        sell_score = luck * 0.6 - outlook * 0.8 - trend * 0.3 + (
            1.0 if p["pos"] == "RB" and (p.get("age") or 0) >= 28 else 0.0)
        entry = {"id": p["id"], "name": p["name"], "pos": p["pos"], "team": p["team"],
                 "ppg": p["ppg"], "xfp_pg": p["xfp_pg"], "proj_pg": p["proj_pg"],
                 "last3_ppg": p["last3_ppg"], "opp_pg": p["opp_pg"],
                 "snap_pct": p["snap_pct"], "owner_team_id": owner,
                 "mine": owner == my_team_id}
        if buy_score > 1.5 and reasons_b and p["proj_pg"] >= r and owner != my_team_id:
            buys.append({**entry, "score": round(buy_score, 2), "reasons": reasons_b})
        if sell_score > 1.5 and reasons_s and p["ppg"] >= r + 1 and owner is not None:
            sells.append({**entry, "score": round(sell_score, 2), "reasons": reasons_s})
    buys.sort(key=lambda x: -x["score"])
    sells.sort(key=lambda x: (not x["mine"], -x["score"]))
    return {"buy_low": buys[:limit], "sell_high": sells[:limit]}


# ---------------------------------------------------------------------------
# Start / sit with Vegas context
# ---------------------------------------------------------------------------
def lines_by_team(games):
    out = {}
    for g in games or []:
        for side, other in (("home", "away"), ("away", "home")):
            team = g[side]
            spread = g.get("spread_home")
            out[team] = {
                "opponent": g[other], "home": side == "home",
                "implied": g.get(f"implied_{side}"),
                "opp_implied": g.get(f"implied_{other}"),
                "spread": (spread if side == "home" else -spread) if spread is not None else None,
                "total": g.get("total"), "kickoff": g.get("kickoff"),
            }
    return out


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def vegas_multiplier(pos, line, avg_implied):
    """Scale a projection by how many points Vegas expects the offense to
    score (and, for DEF, how few the opponent should score)."""
    if not line or line.get("implied") is None or not avg_implied:
        return 1.0, []
    notes = []
    s = line.get("spread") or 0.0
    if pos == "DEF":
        m = _clamp((avg_implied / max(line["opp_implied"], 10)) ** 1.0, 0.7, 1.3)
        notes.append(f"opponent implied {line['opp_implied']:.1f} pts")
        return m, notes
    alpha = {"QB": 0.8, "RB": 0.6, "WR": 0.6, "TE": 0.5, "K": 0.7}.get(pos, 0.5)
    m = _clamp((line["implied"] / avg_implied) ** alpha, 0.8, 1.2)
    notes.append(f"team implied {line['implied']:.1f} pts")
    if pos == "RB":
        m *= 1 + _clamp(-s * 0.005, -0.04, 0.04)
        if s <= -6:
            notes.append(f"{abs(s):g}-pt favorite: run-heavy script")
    elif pos in ("WR", "TE", "QB"):
        m *= 1 + _clamp(s * 0.003, -0.03, 0.03)
        if s >= 6:
            notes.append(f"{s:g}-pt underdog: pass-heavy script")
    return m, notes


def start_sit(league, uni, team, games):
    lines = lines_by_team(games)
    implieds = [g[k] for g in games or [] for k in ("implied_home", "implied_away")
                if g.get(k) is not None]
    avg = sum(implieds) / len(implieds) if implieds else 22.5
    rows = []
    for pid in team["players"]:
        p = uni.get(pid)
        if not p:
            continue
        base = p["proj_week"]
        line = lines.get(p["team"])
        mult, notes = vegas_multiplier(p["pos"], line, avg)
        inj = (p.get("injury") or "").lower()
        imult = INJURY_MULT.get(inj, 1.0)
        if inj:
            notes.append(p["injury"])
        if base <= 0.5 and not inj and lines and line is None:
            notes.append("bye / no game")
        adj = base * mult * imult
        sd = 0.45 * adj + 1.0
        rows.append({
            **_brief(uni, pid), "proj": round(base, 1), "adj_proj": round(adj, 1),
            "floor": round(max(0.0, adj - 0.674 * sd), 1),
            "ceiling": round(adj + 0.674 * sd, 1),
            "vegas_mult": round(mult, 3), "opponent": (line or {}).get("opponent"),
            "home": (line or {}).get("home"), "implied": (line or {}).get("implied"),
            "spread": (line or {}).get("spread"), "notes": notes,
        })
    pts = {r["id"]: (r["adj_proj"], r["pos"]) for r in rows}
    total, lineup = optimal_lineup([r["id"] for r in rows], league["slots"], pts)
    starting = {pid for _, pid in lineup if pid}
    by_id = {r["id"]: r for r in rows}
    for r in rows:
        r["start"] = r["id"] in starting
    current = set(team.get("starters") or [])
    swaps = []
    if current:
        cur_total = optimal_lineup([p for p in current if p in by_id], league["slots"], pts)[0]
        ins = sorted((p for p in starting - current), key=lambda p: -by_id[p]["adj_proj"])
        outs = sorted((p for p in current - starting if p in by_id), key=lambda p: by_id[p]["adj_proj"])
        for i in ins:
            if not outs:
                swaps.append({"start": by_id[i]["name"], "bench": None,
                              "gain": by_id[i]["adj_proj"]})
                continue
            o = next((x for x in outs if by_id[x]["pos"] == by_id[i]["pos"]), outs[0])
            outs.remove(o)
            swaps.append({"start": by_id[i]["name"], "bench": by_id[o]["name"],
                          "gain": round(by_id[i]["adj_proj"] - by_id[o]["adj_proj"], 1)})
        swaps += [{"start": None, "bench": by_id[o]["name"], "gain": -by_id[o]["adj_proj"]}
                  for o in outs]
    else:
        cur_total = None
    close = []
    bench = [r for r in rows if not r["start"]]
    for slot, pid in lineup:
        if not pid:
            continue
        s = by_id[pid]
        for b in bench:
            if b["pos"] in SLOT_ELIGIBLE[slot] and 0 <= s["adj_proj"] - b["adj_proj"] <= 1.5:
                close.append({"slot": slot, "starter": s["name"], "alternative": b["name"],
                              "margin": round(s["adj_proj"] - b["adj_proj"], 1),
                              "tiebreak": "ceiling" if b["ceiling"] > s["ceiling"] else "floor"})
    return {
        "lineup": [{"slot": slot, "player": by_id[pid] if pid else None} for slot, pid in lineup],
        "bench": sorted(bench, key=lambda r: -r["adj_proj"]),
        "projected_total": round(total, 1),
        "current_total": round(cur_total, 1) if cur_total is not None else None,
        "swaps": swaps,
        "close_calls": close[:6],
        "avg_implied": round(avg, 1),
    }


# ---------------------------------------------------------------------------
# Waivers
# ---------------------------------------------------------------------------
def waiver_targets(league, uni, team, rostered, trending=(), limit=10):
    trend = {t["player_id"]: t["count"] for t in trending}
    mine = [p for p in team["players"] if p in uni]
    base = lineup_ppg(uni, mine, league["slots"])
    starters = {pid for _, pid in optimal_lineup(mine, league["slots"], _ros_pts(uni, mine))[1] if pid}
    bench = sorted((p for p in mine if p not in starters), key=lambda p: uni[p]["value"])
    drop = bench[0] if bench else None
    fas = [p for p in uni.values()
           if p["id"] not in rostered and p["team"] and p["ros_ppg"] > 0
           and (p.get("injury") or "").lower() not in NOT_PLAYING]
    fas.sort(key=lambda p: -(p["ros_ppg"] + 0.02 * trend.get(p["id"], 0) ** 0.5))
    out = []
    for p in fas[:80]:
        after = [x for x in mine if x != drop] + [p["id"]]
        gain = lineup_ppg(uni, after, league["slots"]) - base
        stash = p["value"] > (uni[drop]["value"] if drop else 0)
        if gain > 0.2 or stash:
            out.append({**_brief(uni, p["id"]), "lineup_gain_ppg": round(gain, 2),
                        "trending_adds": trend.get(p["id"], 0),
                        "drop": _brief(uni, drop) if drop else None})
    out.sort(key=lambda x: (-x["lineup_gain_ppg"], -x["value"]))
    hot = [{**_brief(uni, pid), "trending_adds": n} for pid, n in
           sorted(trend.items(), key=lambda kv: -kv[1]) if pid in uni and pid not in rostered][:10]
    return {"targets": out[:limit], "trending": hot}


# ---------------------------------------------------------------------------
# Parlays
# ---------------------------------------------------------------------------
def _phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _poisson_over(mean, line):
    """P(X > line) for Poisson X."""
    k_max = math.floor(line)
    cdf = sum(math.exp(-mean) * mean ** k / math.factorial(k) for k in range(k_max + 1))
    return 1 - cdf


def american_to_decimal(odds):
    odds = float(odds)
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def prob_to_american(p):
    if p <= 0 or p >= 1:
        return None
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


PROP_SD = {"pass_yd": (0.22, 20), "rush_yd": (0.45, 8), "rec_yd": (0.55, 6), "rec": (0.4, 1.0),
           "rush_att": (0.3, 2), "pass_cmp": (0.2, 3)}
PROP_LABELS = {"pass_yd": "Pass yds", "pass_td": "Pass TDs", "rush_yd": "Rush yds",
               "rec_yd": "Rec yds", "rec": "Receptions", "anytime_td": "Anytime TD",
               "rush_att": "Rush att", "pass_cmp": "Completions"}


def prop_probability(market, mean, line, side="over"):
    if mean is None:
        return None
    if market == "anytime_td":
        p = 1 - math.exp(-max(mean, 0))
    elif market == "pass_td":
        p = _poisson_over(max(mean, 1e-6), line)
    else:
        cv, base = PROP_SD.get(market, (0.4, 2))
        sd = cv * mean + base
        p = 1 - _phi((line - mean) / sd)
    return p if side == "over" else 1 - p


def prop_mean(market, stats, mult=1.0):
    if not stats:
        return None
    if market == "anytime_td":
        v = (stats.get("rush_td") or 0) + (stats.get("rec_td") or 0)
    else:
        v = stats.get(market) or 0.0
    return v * mult


def grade_parlay(legs, week_proj_stats, players, games, uni=None):
    """legs: [{kind:'prop', player_id, market, line, side, odds?}
              {kind:'game', game:'AWY@HOM', market:'spread'|'total'|'ml',
               team?, line?, side?, odds?}]"""
    lines = lines_by_team(games)
    implieds = [g[k] for g in games or [] for k in ("implied_home", "implied_away")
                if g.get(k) is not None]
    avg = sum(implieds) / len(implieds) if implieds else 22.5
    graded = []
    for leg in legs:
        g = dict(leg)
        odds = leg.get("odds")
        if leg.get("kind") == "prop":
            pid = str(leg.get("player_id") or "")
            p = players.get(pid) or {}
            pos, team = p.get("pos"), p.get("team")
            line = lines.get(team)
            mult = vegas_multiplier(pos, line, avg)[0] if pos else 1.0
            mean = prop_mean(leg["market"], week_proj_stats.get(pid), mult)
            prob = prop_probability(leg["market"], mean, float(leg.get("line") or 0.5),
                                    leg.get("side", "over"))
            if mean is None:
                g["note"] = "no projection this week (bye, injury, or unknown player)"
            g.update(player=p.get("name") or leg.get("player"), pos=pos, team=team,
                     game=_game_key(line, team), model_mean=round(mean, 2) if mean is not None else None,
                     label=f"{p.get('name') or leg.get('player')} {leg.get('side', 'over')} "
                           f"{leg.get('line')} {PROP_LABELS.get(leg['market'], leg['market'])}",
                     source="model")
        else:
            prob, label, gk = _game_leg_prob(leg, games)
            g.update(label=label, game=gk, source="market")
        g["prob"] = round(prob, 4) if prob is not None else None
        if odds not in (None, "") and prob is not None:
            book = 1 / american_to_decimal(odds)
            g["book_prob"] = round(book, 4)
            g["edge"] = round(prob - book, 4)
            g["ev"] = round(prob * american_to_decimal(odds) - 1, 4)
        g["fair_odds"] = prob_to_american(prob) if prob is not None else None
        graded.append(g)

    probs = [g["prob"] for g in graded if g["prob"] is not None]
    parlay_p = math.prod(probs) if probs and len(probs) == len(graded) else None
    dec = None
    if graded and all(g.get("odds") not in (None, "") for g in graded):
        dec = math.prod(american_to_decimal(g["odds"]) for g in graded)
    return {
        "legs": graded,
        "hit_prob": round(parlay_p, 4) if parlay_p is not None else None,
        "fair_odds": prob_to_american(parlay_p) if parlay_p else None,
        "book_decimal": round(dec, 3) if dec else None,
        "book_american": prob_to_american(1 / dec) if dec else None,
        "ev": round(parlay_p * dec - 1, 4) if parlay_p and dec else None,
        "correlations": _correlations(graded),
    }


def _game_key(line, team):
    if not line or not team:
        return None
    return f"{line['opponent']}@{team}" if line["home"] else f"{team}@{line['opponent']}"


def _find_game(key, games):
    for g in games or []:
        if key in (f"{g['away']}@{g['home']}", g.get("event_id")):
            return g
    return None


def _game_leg_prob(leg, games):
    g = _find_game(leg.get("game"), games)
    market = leg.get("market")
    if not g:
        return None, f"{leg.get('game')} {market} (no line found)", leg.get("game")
    gk = f"{g['away']}@{g['home']}"
    if market == "total":
        if g.get("total") is None:
            return None, f"{gk} total (no line)", gk
        line = float(leg.get("line") or g["total"])
        side = leg.get("side", "over")
        p = 1 - _phi((line - g["total"]) / 10.5)
        return (p if side == "over" else 1 - p), f"{gk} {side} {line:g}", gk
    team = leg.get("team")
    if team not in (g["home"], g["away"]):
        return None, f"{gk} {market}: pick {g['away']} or {g['home']}", gk
    if g.get("spread_home") is None and market != "ml":
        return None, f"{gk} {team} spread (no line)", gk
    home_margin = -(g.get("spread_home") or 0.0)
    mu = home_margin if team == g["home"] else -home_margin
    if market == "ml":
        if g.get("win_prob_home") is not None:
            p = g["win_prob_home"] if team == g["home"] else 1 - g["win_prob_home"]
        else:
            p = 1 - _phi(-mu / 13.5)
        return p, f"{team} moneyline", gk
    line = float(leg["line"]) if leg.get("line") not in (None, "") else (
        g["spread_home"] if team == g["home"] else -g["spread_home"])
    p = 1 - _phi((-line - mu) / 13.5)
    return p, f"{team} {line:+g}", gk


def _correlations(legs):
    notes = []
    by_game = {}
    for g in legs:
        if g.get("game"):
            by_game.setdefault(g["game"], []).append(g)
    for game, gl in by_game.items():
        if len(gl) < 2:
            continue
        props = [g for g in gl if g.get("kind") == "prop"]
        total = next((g for g in gl if g.get("market") == "total"), None)
        passers = [g for g in props if g.get("pos") == "QB" and g["market"] in ("pass_yd", "pass_td")
                   and g.get("side", "over") == "over"]
        catchers = [g for g in props if g.get("pos") in ("WR", "TE", "RB") and
                    g["market"] in ("rec_yd", "rec") and g.get("side", "over") == "over"]
        for q in passers:
            for c in catchers:
                if c.get("team") == q.get("team"):
                    notes.append({"type": "positive", "game": game,
                                  "note": f"{q['player']} and {c['player']} overs move together "
                                          f"-- the true hit rate is higher than the independent math."})
        if total:
            overs = [g for g in props if g.get("side", "over") == "over"]
            if total.get("side", "over") == "over" and overs:
                notes.append({"type": "positive", "game": game,
                              "note": "Game over + player overs are positively correlated."})
            elif total.get("side") == "under" and overs:
                notes.append({"type": "negative", "game": game,
                              "note": "Game under fights the player overs -- these legs work "
                                      "against each other."})
        rbs = [g for g in props if g.get("pos") == "RB" and g["market"] == "rush_yd"
               and g.get("side", "over") == "over"]
        sides = [g for g in gl if g.get("kind") == "game" and g.get("market") in ("ml", "spread")]
        for r in rbs:
            for s in sides:
                if s.get("team") == r.get("team"):
                    notes.append({"type": "positive", "game": game,
                                  "note": f"{r['player']} rushing + {s['team']} winning go together "
                                          f"(teams with a lead run the ball)."})
        if not any(n["game"] == game for n in notes):
            notes.append({"type": "info", "game": game,
                          "note": "Multiple legs from one game -- outcomes aren't fully independent."})
    return notes


def grade_props_board(props, week_proj_stats, players, games, limit=25):
    """Odds API prop lines for one game -> every side graded, best edges first."""
    index = {}
    for pid, p in players.items():
        index.setdefault(_simple(p["name"]), []).append(pid)
    lines = lines_by_team(games)
    implieds = [g[k] for g in games or [] for k in ("implied_home", "implied_away")
                if g.get(k) is not None]
    avg = sum(implieds) / len(implieds) if implieds else 22.5
    out = []
    for pr in props:
        pids = index.get(_simple(pr["player"])) or []
        pid = next((x for x in pids if x in week_proj_stats), pids[0] if pids else None)
        if not pid:
            continue
        p = players[pid]
        mult = vegas_multiplier(p["pos"], lines.get(p["team"]), avg)[0]
        mean = prop_mean(pr["market"], week_proj_stats.get(pid), mult)
        if mean is None:
            continue
        for side in ("over", "under"):
            odds = pr.get(side)
            if odds is None:
                continue
            prob = prop_probability(pr["market"], mean, pr["line"], side)
            book = 1 / american_to_decimal(odds)
            out.append({"player_id": pid, "player": p["name"], "pos": p["pos"], "team": p["team"],
                        "market": pr["market"], "label": PROP_LABELS.get(pr["market"], pr["market"]),
                        "line": pr["line"], "side": side, "odds": odds,
                        "model_mean": round(mean, 2), "prob": round(prob, 4),
                        "book_prob": round(book, 4), "edge": round(prob - book, 4),
                        "ev": round(prob * american_to_decimal(odds) - 1, 4)})
    out.sort(key=lambda x: -x["edge"])
    return out[:limit]


def _simple(name):
    import fantasy_sources as src
    return src.name_key(name)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def load_week_data(season, current_week, end_week=END_WEEK):
    import fantasy_sources as src
    proj_weeks = list(range(current_week, end_week + 1))
    stat_weeks = list(range(1, current_week))
    with ThreadPoolExecutor(max_workers=8) as ex:
        pf = {w: ex.submit(src.sleeper_week_projections, season, w) for w in proj_weeks}
        sf = {w: ex.submit(src.sleeper_week_stats, season, w) for w in stat_weeks}
        projs, stats = {}, {}
        for w, f in pf.items():
            try:
                projs[w] = f.result()
            except src.SourceError:
                projs[w] = {}
        for w, f in sf.items():
            try:
                stats[w] = f.result()
            except src.SourceError:
                stats[w] = {}
    return projs, stats


def load_league(platform, league_id, season, espn_s2=None, swid=None):
    import fantasy_sources as src
    if platform == "sleeper":
        return src.sleeper_league(league_id)
    if platform == "espn":
        return src.espn_league(league_id, season, espn_s2=espn_s2, swid=swid)
    raise ValueError("platform must be 'sleeper' or 'espn'")


def prepare(league, week=None):
    """Load everything the analyses share and build the valued universe."""
    import fantasy_sources as src
    state = src.nfl_state()
    season = league.get("season") or state["season"]
    current = int(week or state["week"] or 1)
    players = src.sleeper_players()
    projs, stats = load_week_data(season, current)
    rostered = {p for t in league["teams"] for p in t["players"]}
    uni = build_universe(players, league["scoring"], projs, stats, current, rostered=rostered)
    repl, per_team = replacement_levels(uni, league["slots"], len(league["teams"]))
    attach_values(uni, repl)
    try:
        games = src.game_lines(season, current)
    except src.SourceError:
        games = []
    return {"players": players, "uni": uni, "repl": repl, "per_team": per_team,
            "projs": projs, "games": games, "season": season, "week": current,
            "rostered": rostered}


def analyze_league(league, my_team_id, week=None):
    import fantasy_sources as src
    teams = {t["team_id"]: t for t in league["teams"]}
    if my_team_id not in teams:
        raise ValueError("Pick your team from the league's team list.")
    ctx = prepare(league, week)
    uni, repl, per_team = ctx["uni"], ctx["repl"], ctx["per_team"]
    needs = team_needs(league, uni, repl, per_team)
    owners = {p: t["team_id"] for t in league["teams"] for p in t["players"]}
    me = teams[my_team_id]
    power = sorted(({"team_id": t["team_id"], "name": t["name"], "owner": t["owner"],
                     "record": f"{t['wins']}-{t['losses']}" + (f"-{t['ties']}" if t["ties"] else ""),
                     "points_for": round(t["points_for"] or 0, 1),
                     "lineup_ppg": round(lineup_ppg(uni, t["players"], league["slots"]), 1),
                     "needs": needs[t["team_id"]]["needs"],
                     "surplus": needs[t["team_id"]]["surplus"],
                     "positions": needs[t["team_id"]]["positions"]}
                    for t in league["teams"]), key=lambda x: -x["lineup_ppg"])
    roster = sorted((_brief(uni, p) | {"proj_week": uni[p]["proj_week"], "ppg": uni[p]["ppg"],
                                       "games": uni[p]["games"], "age": uni[p]["age"]}
                     for p in me["players"] if p in uni), key=lambda x: -(x["value"] or 0))
    return {
        "league": {"platform": league["platform"], "league_id": league["league_id"],
                   "name": league["name"], "season": ctx["season"], "week": ctx["week"],
                   "slots": league["slots"], "teams": len(league["teams"]),
                   "scoring": _scoring_label(league["scoring"]),
                   "unmatched_players": league.get("unmatched_players") or []},
        "my_team": {"team_id": my_team_id, "name": me["name"], "owner": me["owner"],
                    "needs": needs[my_team_id], "roster": roster},
        "rosters": {t["team_id"]: sorted((_brief(uni, p) for p in t["players"] if p in uni),
                                         key=lambda x: -(x["value"] or 0))
                    for t in league["teams"]},
        "replacement": {k: round(v, 2) for k, v in repl.items()},
        "power_rankings": power,
        "start_sit": start_sit(league, uni, me, ctx["games"]),
        "trades": find_trades(league, uni, needs, my_team_id),
        "buy_sell": buy_sell(uni, repl, owners, my_team_id),
        "waivers": waiver_targets(league, uni, me, ctx["rostered"], src.sleeper_trending()),
        "games": ctx["games"],
    }


def _scoring_label(scoring):
    rec = scoring.get("rec", 0)
    base = "PPR" if rec >= 0.75 else "Half-PPR" if rec >= 0.25 else "Standard"
    extras = []
    if scoring.get("pass_td", 4) >= 6:
        extras.append("6-pt pass TD")
    if scoring.get("bonus_rec_te"):
        extras.append("TE premium")
    return base + (f" ({', '.join(extras)})" if extras else "")

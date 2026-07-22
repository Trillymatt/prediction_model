"""
Build a POINT-IN-TIME training dataset for NFL GAME OUTCOMES.

    python 36_nfl_build_game_training.py     # writes nfl_game_training_data.csv

The football sibling of 12_build_game_training_data.py. One row per completed
regular-season/playoff game (from the home team's perspective) built from
nfl_schedule scores, with features computed using ONLY games strictly before
that game's date. Two ingredients set it apart from the NBA builder:

  * an as-of Simple Rating System (opponent-adjusted point margin) for each
    team, so the model sees strength-of-schedule directly -- the same rating
    nfl_common.team_ratings serves live during the season; and
  * team offensive touchdowns per game, derived from nfl_player_game_logs
    (rushing + receiving TDs), so the touchdown market trains on real data.

Targets
-------
  target_margin     home_score - away_score   (sign = winner, size = spread)
  target_total      home_score + away_score
  target_total_td   home + away offensive touchdowns (NaN if logs missing)
  target_home_win   1/0
  target_tie        1/0   (rare, but a real NFL outcome)

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import os
from collections import defaultdict, deque
from datetime import datetime

import pandas as pd

import nfl_common as nc


OUTPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "nfl_game_training_data.csv"
)

MIN_TEAM_GAMES = 3
ROLLING_MAX = 6
REST_CAP = 14
SRS_ITERS = 25
HFA = nc.HFA_POINTS
TD_STATS = ("rush_td", "rec_td")   # passing TD == the receiver's TD; don't double


def num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_date(raw):
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def avg(total, count):
    return total / count if count else None


def fetch_team_tds() -> dict:
    """{game_id: {team_name: offensive_tds}} from the player logs."""
    columns = "game_id,team," + ",".join(TD_STATS)
    rows = nc.fetch_all(nc.LOGS_TABLE, columns, order_col="game_id")
    out = defaultdict(lambda: defaultdict(float))
    for r in rows:
        gid = r.get("game_id")
        team = nc.normalize_team(r.get("team"))
        if gid is None or not team:
            continue
        out[gid][team] += sum((r.get(s) or 0) for s in TD_STATS)
    return out


def load_games():
    """Completed rated games grouped by day, with both scores + TD totals."""
    schedule = nc.fetch_all(
        nc.SCHEDULE_TABLE,
        "game_id,game_date,season,season_type,home_team,away_team,status,"
        "home_score,away_score",
        order_col="game_date",
    )
    tds = fetch_team_tds()
    by_day = defaultdict(list)
    for g in schedule:
        if g.get("status") != "completed":
            continue
        if (g.get("season_type") or "regular") not in nc.RATED_SEASON_TYPES:
            continue
        hs, as_ = num(g.get("home_score")), num(g.get("away_score"))
        d = parse_date(g.get("game_date"))
        home, away = nc.normalize_team(g.get("home_team")), nc.normalize_team(g.get("away_team"))
        if hs is None or as_ is None or d is None or not home or not away:
            continue
        gtd = tds.get(g.get("game_id"), {})
        home_td = gtd.get(home)
        away_td = gtd.get(away)
        by_day[d].append({
            "date": d, "season": str(g.get("season")),
            "season_type": g.get("season_type") or "regular",
            "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_,
            "total_td": (home_td + away_td) if (home_td is not None and away_td is not None) else None,
        })
    return by_day


class TeamState:
    def __init__(self):
        self.recent = defaultdict(lambda: deque(maxlen=ROLLING_MAX))
        self.season_sum = defaultdict(lambda: defaultdict(float))
        self.last_date = {}
        self.games = defaultdict(list)   # season -> [(home, away, neutral_margin)]

    def features(self, team, season, prefix):
        ssum = self.season_sum[(team, season)]
        n = ssum.get("n", 0)
        recent = list(self.recent[team])
        last5 = recent[-5:]
        ppg = avg(ssum.get("scored", 0.0), n)
        papg = avg(ssum.get("allowed", 0.0), n)
        return {
            f"{prefix}_season_games": n,
            f"{prefix}_season_ppg": ppg,
            f"{prefix}_season_papg": papg,
            f"{prefix}_season_net": (ppg - papg) if (ppg is not None and papg is not None) else None,
            f"{prefix}_season_win_pct": avg(ssum.get("wins", 0.0), n),
            f"{prefix}_l5_ppg": avg(sum(g["scored"] for g in last5), len(last5)),
            f"{prefix}_l5_papg": avg(sum(g["allowed"] for g in last5), len(last5)),
        }

    def rest_days(self, team, game_date):
        prev = self.last_date.get(team)
        return min((game_date - prev).days, REST_CAP) if prev else None

    def season_games(self, team, season):
        return self.season_sum[(team, season)].get("n", 0)

    def srs(self, season):
        """Opponent-adjusted rating from games so far this season (fixed-point)."""
        games = self.games[season]
        if not games:
            return {}
        margins = defaultdict(list)   # team -> neutral margins
        opps = defaultdict(list)      # team -> opponents faced
        for home, away, m in games:
            margins[home].append(m)
            margins[away].append(-m)
            opps[home].append(away)
            opps[away].append(home)
        avg_margin = {t: sum(v) / len(v) for t, v in margins.items()}
        ratings = dict(avg_margin)
        for _ in range(SRS_ITERS):
            new = {}
            for t in margins:
                sos = sum(ratings.get(o, 0.0) for o in opps[t]) / len(opps[t])
                new[t] = avg_margin[t] + sos
            mean_r = sum(new.values()) / len(new)
            ratings = {t: r - mean_r for t, r in new.items()}
        return ratings

    def add_result(self, home, away, hs, as_, season, game_date):
        for team, scored, allowed in ((home, hs, as_), (away, as_, hs)):
            won = 1.0 if scored > allowed else 0.0
            self.recent[team].append({"scored": scored, "allowed": allowed, "won": won})
            ssum = self.season_sum[(team, season)]
            ssum["scored"] += scored
            ssum["allowed"] += allowed
            ssum["wins"] += won
            ssum["n"] += 1
            self.last_date[team] = game_date
        self.games[season].append((home, away, (hs - as_) - HFA))


def main():
    print("Loading completed games + team touchdowns ...")
    by_day = load_games()
    n_games = sum(len(v) for v in by_day.values())
    print(f"  {n_games} games across {len(by_day)} days\n")

    state = TeamState()
    out_rows = []
    for day in sorted(by_day):
        games = by_day[day]
        ratings_by_season = {}   # computed once per day, reused for every game
        for g in games:
            season = g["season"]
            if state.season_games(g["home_team"], season) < MIN_TEAM_GAMES:
                continue
            if state.season_games(g["away_team"], season) < MIN_TEAM_GAMES:
                continue
            if season not in ratings_by_season:
                ratings_by_season[season] = state.srs(season)
            ratings = ratings_by_season[season]
            row = {
                "game_date": g["date"].isoformat(),
                "season": season,
                "season_type": g["season_type"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "home_days_rest": state.rest_days(g["home_team"], g["date"]),
                "away_days_rest": state.rest_days(g["away_team"], g["date"]),
                "home_rating": ratings.get(g["home_team"]),
                "away_rating": ratings.get(g["away_team"]),
            }
            row.update(state.features(g["home_team"], season, "home"))
            row.update(state.features(g["away_team"], season, "away"))
            row["target_margin"] = g["home_score"] - g["away_score"]
            row["target_total"] = g["home_score"] + g["away_score"]
            row["target_total_td"] = g["total_td"]
            row["target_home_win"] = 1 if g["home_score"] > g["away_score"] else 0
            row["target_tie"] = 1 if g["home_score"] == g["away_score"] else 0
            out_rows.append(row)

        for g in games:
            state.add_result(g["home_team"], g["away_team"], g["home_score"],
                             g["away_score"], g["season"], g["date"])

    df = pd.DataFrame(out_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print("=" * 60)
    print(f"DONE. Wrote {len(df)} game rows to {OUTPUT_CSV}")
    if not df.empty:
        print(f"Home team won {df['target_home_win'].mean()*100:.1f}% | "
              f"ties {df['target_tie'].mean()*100:.2f}% | "
              f"TD target present on {df['target_total_td'].notna().mean()*100:.0f}% of rows")


if __name__ == "__main__":
    main()

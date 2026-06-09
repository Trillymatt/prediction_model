"""
Build a POINT-IN-TIME training dataset for GAME OUTCOMES (winner + score).

    python 12_build_game_training_data.py        # writes game_training_data.csv

The team-level sibling of 10_build_training_data.py. Official team scores come
straight from nba_api's LeagueGameLog for every season (two calls per season:
regular season + playoffs), then we emit one row per GAME from the home team's
perspective, with features computed using ONLY games strictly before that
game's date -- the same as-of discipline as the player dataset, so the backtest
stays honest.

(Why not aggregate nba_player_game_logs like everything else? Those logs only
cover CURRENTLY ACTIVE players, so historical team scores are missing the
points of everyone who has since left the league -- 2022-23 'team scores'
came out 20+ points short. LeagueGameLog is the official per-team box line.)

Features per row (all as-of the game date, home_/away_ prefixed for both teams)
-------------------------------------------------------------------------------
  season:   points scored / allowed per game, win pct, net rating, games played
  form:     last-5 and last-10 scored / allowed per game, last-10 win pct
  context:  days of rest for each side, season type (regular vs playoffs)

Targets (labels)
----------------
  target_margin  home_score - away_score   (sign = winner, size = spread)
  target_total   home_score + away_score   (the over/under total)
  target_home_win 1/0                      (for reporting accuracy)

The model trains on margin and total; score projections and the win
probability are derived from those two at inference time, so the winner, the
spread and the projected score can never contradict each other.

Setup:
    pip install -r requirements.txt
    # fill in SUPABASE_URL and SUPABASE_KEY in .env
"""

import os
import time
from collections import defaultdict, deque
from datetime import datetime

import pandas as pd

from nba_api.stats.endpoints import leaguegamelog


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "game_training_data.csv"
)

def _seasons_through_current(first_start_year=2022) -> list:
    """Season labels from 2022-23 (same span as 01) through the current one."""
    today = datetime.now()
    last_start = today.year if today.month >= 10 else today.year - 1
    return [f"{y}-{str(y + 1)[2:]}" for y in range(first_start_year, last_start + 1)]


SEASONS = _seasons_through_current()
SEASON_TYPES = ["Regular Season", "Playoffs"]
API_DELAY = 0.6           # seconds between API calls (rate-limit guard)
API_TIMEOUT = 30          # seconds before an nba_api request times out

# Both teams need this many season games before a row is emitted, so the
# season-to-date features rest on something real (early-season noise dropped).
MIN_TEAM_GAMES = 5
ROLLING_MAX = 10          # longest rolling window we need (L10)
REST_CAP = 14             # cap days_rest so the offseason doesn't blow the scale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def avg(total, count):
    """Safe mean; None when there's nothing to average."""
    return total / count if count else None


def parse_date(raw):
    """ISO 'YYYY-MM-DD' -> date (None if unparseable)."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def fetch_all_games() -> dict:
    """Pull every completed game from LeagueGameLog, grouped {date: [game, ...]}.

    LeagueGameLog yields two rows per game (one per team); the MATCHUP string
    ('GSW vs. LAL' = home, 'GSW @ LAL' = away) tells us which side each row is.
    We merge the pair into one game record with both official scores.
    """
    games = {}   # game_id -> record
    for season in SEASONS:
        for season_type in SEASON_TYPES:
            print(f"  [{season} {season_type}] fetching ...", end=" ", flush=True)
            log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                timeout=API_TIMEOUT,
            )
            rows = log.get_normalized_dict().get("LeagueGameLog", [])
            print(f"{len(rows)} team-rows")
            for row in rows:
                game_id = row.get("GAME_ID")
                pts = row.get("PTS")
                matchup = row.get("MATCHUP") or ""
                team = (row.get("TEAM_ABBREVIATION") or "").strip()
                if not game_id or pts is None or not team:
                    continue
                rec = games.setdefault(game_id, {
                    "date": parse_date(row.get("GAME_DATE")),
                    "season": season,
                    "season_type": season_type,
                    "home_team": None, "away_team": None,
                    "home_score": None, "away_score": None,
                })
                if " vs. " in matchup:
                    rec["home_team"], rec["home_score"] = team, float(pts)
                elif " @ " in matchup:
                    rec["away_team"], rec["away_score"] = team, float(pts)
            time.sleep(API_DELAY)

    by_day = defaultdict(list)
    dropped = 0
    for rec in games.values():
        if (rec["date"] is None or rec["home_score"] is None
                or rec["away_score"] is None):
            dropped += 1
            continue
        by_day[rec["date"]].append(rec)
    if dropped:
        print(f"  ({dropped} games missing a side; skipped)")
    return by_day


# ---------------------------------------------------------------------------
# Running state (all "as-of" the day currently being processed)
# ---------------------------------------------------------------------------
class TeamState:
    """Per-team accumulators updated chronologically; every read is point-in-time."""

    def __init__(self):
        self.recent = defaultdict(lambda: deque(maxlen=ROLLING_MAX))  # team -> games
        self.season_sum = defaultdict(lambda: defaultdict(float))     # (team, season)
        self.last_date = {}                                           # team -> date

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
            f"{prefix}_l10_ppg": avg(sum(g["scored"] for g in recent), len(recent)),
            f"{prefix}_l10_papg": avg(sum(g["allowed"] for g in recent), len(recent)),
            f"{prefix}_l10_win_pct": avg(sum(g["won"] for g in recent), len(recent)),
            f"{prefix}_l5_ppg": avg(sum(g["scored"] for g in last5), len(last5)),
            f"{prefix}_l5_papg": avg(sum(g["allowed"] for g in last5), len(last5)),
        }

    def rest_days(self, team, game_date):
        prev = self.last_date.get(team)
        return min((game_date - prev).days, REST_CAP) if prev else None

    def season_games(self, team, season):
        return self.season_sum[(team, season)].get("n", 0)

    def add_result(self, team, season, scored, allowed, game_date):
        won = 1.0 if scored > allowed else 0.0
        self.recent[team].append({"scored": scored, "allowed": allowed, "won": won})
        ssum = self.season_sum[(team, season)]
        ssum["scored"] += scored
        ssum["allowed"] += allowed
        ssum["wins"] += won
        ssum["n"] += 1
        self.last_date[team] = game_date


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Fetching official team game logs ({', '.join(SEASONS)}) ...")
    by_day = fetch_all_games()
    n_games = sum(len(g) for g in by_day.values())
    print(f"  {n_games} games across {len(by_day)} days\n")

    state = TeamState()
    out_rows = []

    for day in sorted(by_day):
        games = by_day[day]

        # 1) READ: build features for every game on this day from prior state.
        for g in games:
            season = g["season"]
            if state.season_games(g["home_team"], season) < MIN_TEAM_GAMES:
                continue
            if state.season_games(g["away_team"], season) < MIN_TEAM_GAMES:
                continue

            row = {
                "game_date": g["date"].isoformat(),
                "season": season,
                "season_type": g["season_type"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "home_days_rest": state.rest_days(g["home_team"], g["date"]),
                "away_days_rest": state.rest_days(g["away_team"], g["date"]),
            }
            row.update(state.features(g["home_team"], season, "home"))
            row.update(state.features(g["away_team"], season, "away"))
            # Labels (the actual outcome we want to predict).
            row["target_margin"] = g["home_score"] - g["away_score"]
            row["target_total"] = g["home_score"] + g["away_score"]
            row["target_home_win"] = 1 if g["home_score"] > g["away_score"] else 0
            out_rows.append(row)

        # 2) WRITE: now that the day's features are locked, fold the day in.
        for g in games:
            state.add_result(g["home_team"], g["season"],
                             g["home_score"], g["away_score"], g["date"])
            state.add_result(g["away_team"], g["season"],
                             g["away_score"], g["home_score"], g["date"])

    df = pd.DataFrame(out_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("=" * 60)
    print(f"DONE. Wrote {len(df)} training rows to {OUTPUT_CSV}")
    if not df.empty:
        seasons = ", ".join(sorted(s for s in df["season"].dropna().unique()))
        print(f"Seasons covered: {seasons}")
        print(f"Home team won {df['target_home_win'].mean() * 100:.1f}% of games")


if __name__ == "__main__":
    main()

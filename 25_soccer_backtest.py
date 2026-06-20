"""
Backtest the soccer match model against completed results -- the World Cup first.

    # How well did the model call the 2026 World Cup so far?
    python 25_soccer_backtest.py

    # Compare the WC-aware model against the old flat one (the A/B):
    python 25_soccer_backtest.py --flat        # old behaviour
    python 25_soccer_backtest.py               # new behaviour

    # Everything, or one competition, since a date, with per-match detail:
    python 25_soccer_backtest.py --all
    python 25_soccer_backtest.py --competition "FIFA World Cup" --since 2026-06-01 -v

This is the measurement half of "make the model better based on this year's
World Cup matches": it replays the model on each completed match using ONLY
the data available before kickoff (point-in-time Elo, form and baselines --
no peeking at the result), then scores the predictions:

  Outcome (1X2)   Brier, log-loss, accuracy vs a uniform/Elo-favourite baseline
  Totals (O/U)    Brier, log-loss, accuracy at the chosen line
  BTTS            Brier, accuracy
  Calibration     predicted favourite prob vs realised, in buckets

Tune the knobs in soccer_common.py (COMPETITION_FORM_WEIGHTS, WC_BASELINE_*,
SPLIT_RHO, FORM_DAMPENING, ...) and re-run to see the numbers move. Lower
Brier / log-loss is better; a calibrated model's buckets track the diagonal.

Setup:
    pip install -r requirements.txt
    # same .env (SUPABASE_URL / SUPABASE_KEY) as the other scripts
"""

import argparse
import importlib.util
import math
import os

import soccer_common as sc

LOG_EPS = 1e-12          # clamp so a 0-probability outcome doesn't give -inf
UNIFORM_LOGLOSS = math.log(3)   # a coin-flip three-way model's log-loss (~1.099)


def _load_engine():
    """Load 23_soccer_game_projections (digit-leading name) for its grid math."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "soccer_game_projection_engine",
        os.path.join(here, "23_soccer_game_projections.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outcome_of(home_score, away_score) -> str:
    if home_score > away_score:
        return "home"
    if home_score < away_score:
        return "away"
    return "draw"


def predict(engine, home, away, competition, prior_rows, total_line):
    """Point-in-time prediction for one match from the rows before it."""
    # The Elo cache keys on id(list); short-lived per-match lists can recycle
    # ids, so clear it to force an honest recompute for every match.
    sc._elo_cache["key"] = None
    xg = sc.expected_goals(home, away, schedule_rows=prior_rows,
                           competition=competition or "FIFA World Cup")
    grid = engine.score_grid(xg["lambda_home"], xg["lambda_away"])
    markets = engine.grid_markets(grid, total_line=total_line)
    markets["elo_home"], markets["elo_away"] = xg["elo_home"], xg["elo_away"]
    return markets


class Metric:
    """Running Brier / log-loss / accuracy for a set of predictions."""

    def __init__(self, name):
        self.name = name
        self.n = 0
        self.brier = 0.0
        self.logloss = 0.0
        self.correct = 0

    def add_multiclass(self, probs: dict, actual: str):
        self.n += 1
        for cls, p in probs.items():
            y = 1.0 if cls == actual else 0.0
            self.brier += (p - y) ** 2
        self.logloss += -math.log(max(probs.get(actual, 0.0), LOG_EPS))
        if max(probs, key=probs.get) == actual:
            self.correct += 1

    def add_binary(self, p_yes: float, actual_yes: bool):
        self.n += 1
        y = 1.0 if actual_yes else 0.0
        self.brier += (p_yes - y) ** 2
        p = p_yes if actual_yes else 1.0 - p_yes
        self.logloss += -math.log(max(p, LOG_EPS))
        pick_yes = p_yes >= 0.5
        if pick_yes == actual_yes:
            self.correct += 1

    def row(self):
        if not self.n:
            return f"  {self.name:14} (no matches)"
        return (f"  {self.name:14} n={self.n:<4} "
                f"Brier={self.brier / self.n:.4f}  "
                f"LogLoss={self.logloss / self.n:.4f}  "
                f"Acc={self.correct / self.n * 100:5.1f}%")


def calibration_table(pairs, bins=5):
    """pairs = [(predicted_prob, hit_bool)]; show predicted vs realised."""
    buckets = [[] for _ in range(bins)]
    for p, hit in pairs:
        idx = min(int(p * bins), bins - 1)
        buckets[idx].append((p, hit))
    lines = ["  bucket        n    pred    actual"]
    for i, b in enumerate(buckets):
        lo, hi = i / bins, (i + 1) / bins
        if not b:
            lines.append(f"  {lo:.1f}-{hi:.1f}      0      -        -")
            continue
        pred = sum(p for p, _ in b) / len(b)
        act = sum(1 for _, h in b if h) / len(b)
        lines.append(f"  {lo:.1f}-{hi:.1f}   {len(b):4d}  "
                     f"{pred * 100:5.1f}%   {act * 100:5.1f}%")
    return "\n".join(lines)


def apply_flat_config():
    """Revert the WC-aware knobs to the old flat model, for an A/B run."""
    sc.COMPETITION_FORM_WEIGHTS = {k: 1.0 for k in sc.COMPETITION_FORM_WEIGHTS}
    sc.WC_BASELINE_BLEND = 0.0


def main():
    ap = argparse.ArgumentParser(description="Backtest the soccer match model.")
    ap.add_argument("--competition", default="FIFA World Cup",
                    help='Substring filter, e.g. "World Cup" (default), '
                         '"Qualifying", "Friendly". Ignored with --all.')
    ap.add_argument("--all", action="store_true",
                    help="Score every completed match, any competition.")
    ap.add_argument("--since", metavar="YYYY-MM-DD", default=None,
                    help="Only score matches on/after this date.")
    ap.add_argument("--total-line", type=float, default=2.5,
                    help="Over/under line to grade totals at (default 2.5).")
    ap.add_argument("--min-prior", type=int, default=0,
                    help="Skip matches with fewer than N prior completed games "
                         "(thin early history is noisy).")
    ap.add_argument("--flat", action="store_true",
                    help="Use the OLD flat config (no competition form "
                         "weighting, all-international baseline) for an A/B.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Print every match's call vs the result.")
    args = ap.parse_args()

    if args.flat:
        apply_flat_config()

    engine = _load_engine()
    rows = sc.fetch_schedule_rows(force=True)

    since = sc.parse_date(args.since) if args.since else None
    comp_filter = None if args.all else (args.competition or "").lower()

    # All completed, scoreable matches, oldest first -> the replay timeline.
    completed = []
    for g in rows:
        if g.get("status") != "completed":
            continue
        if g.get("home_score") is None or g.get("away_score") is None:
            continue
        d = sc.parse_date(g.get("match_date"))
        if d is None:
            continue
        completed.append((d, g))
    completed.sort(key=lambda t: t[0])
    comp_dates = [d for d, _ in completed]

    targets = []
    for d, g in completed:
        if since and d < since:
            continue
        if comp_filter and comp_filter not in (g.get("competition") or "").lower():
            continue
        targets.append((d, g))

    if not targets:
        raise SystemExit("No completed matches matched the filter. "
                         "Run 20_soccer_schedule.py, or widen --competition/--all.")

    label = "ALL competitions" if args.all else f'"{args.competition}"'
    cfg = "FLAT (old)" if args.flat else "WC-aware (new)"
    print(f"Backtesting {len(targets)} completed matches  [{label}]  "
          f"config={cfg}\n")

    outcome_m = Metric("Outcome 1X2")
    totals_m = Metric(f"Totals {args.total_line}")
    btts_m = Metric("BTTS")
    fav_cal, over_cal = [], []
    elo_fav_correct = 0
    skipped = 0

    import bisect
    for d, g in targets:
        # Strictly-earlier completed matches = everything knowable pre-kickoff.
        cut = bisect.bisect_left(comp_dates, d)
        prior_rows = [gg for _, gg in completed[:cut]]
        if len(prior_rows) < args.min_prior:
            skipped += 1
            continue

        home = sc.normalize_team(g.get("home_team"))
        away = sc.normalize_team(g.get("away_team"))
        hs, as_ = g["home_score"], g["away_score"]
        actual = outcome_of(hs, as_)
        try:
            m = predict(engine, home, away, g.get("competition"),
                        prior_rows, args.total_line)
        except Exception as exc:  # noqa: BLE001 - one bad match shouldn't stop the run
            skipped += 1
            if args.verbose:
                print(f"  !! {home} vs {away} ({d}): {exc}")
            continue

        probs = {"home": m["p_home"], "draw": m["p_draw"], "away": m["p_away"]}
        outcome_m.add_multiclass(probs, actual)

        actual_over = (hs + as_) > args.total_line
        totals_m.add_binary(m["p_over"], actual_over)
        over_cal.append((m["p_over"], actual_over))

        actual_btts = hs > 0 and as_ > 0
        btts_m.add_binary(m["p_btts"], actual_btts)

        # Favourite calibration: did the side the model favoured come in?
        fav = max(probs, key=probs.get)
        fav_cal.append((probs[fav], fav == actual))

        # Pure-Elo baseline: higher (host-adjusted) Elo as the pick.
        elo_pick = "home" if m["elo_home"] >= m["elo_away"] else "away"
        if elo_pick == actual:
            elo_fav_correct += 1

        if args.verbose:
            call = max(probs, key=probs.get)
            print(f"  {d}  {home:>16} {hs}-{as_} {away:<16}  "
                  f"model={call:4} "
                  f"({probs['home']*100:2.0f}/{probs['draw']*100:2.0f}/"
                  f"{probs['away']*100:2.0f})  actual={actual}")

    scored = outcome_m.n
    print("\n" + "=" * 60)
    print(f"Scored {scored} matches" + (f"  (skipped {skipped})" if skipped else ""))
    print("\nAccuracy + sharpness (lower Brier / log-loss is better):")
    print(outcome_m.row())
    print(totals_m.row())
    print(btts_m.row())

    if scored:
        print("\nBaselines (outcome):")
        print(f"  {'Uniform 1/3':14} n={scored:<4} "
              f"Brier={2/3:.4f}  LogLoss={UNIFORM_LOGLOSS:.4f}  Acc= 33.3%")
        print(f"  {'Elo favourite':14} n={scored:<4} "
              f"{'':25}Acc={elo_fav_correct / scored * 100:5.1f}%")
        skill = 1.0 - (outcome_m.logloss / scored) / UNIFORM_LOGLOSS
        print(f"\n  Log-loss skill vs uniform: {skill * 100:+.1f}%  "
              f"(higher = more informative than a coin flip)")

        print("\nFavourite calibration (model's pick prob vs how often it won):")
        print(calibration_table(fav_cal))
        print(f"\nOver {args.total_line} calibration:")
        print(calibration_table(over_cal))


if __name__ == "__main__":
    main()

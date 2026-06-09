"""
Train and BACKTEST the game-outcome models from game_training_data.csv.

    python 13_train_game_model.py

Two targets, each with three gradient-boosted models (mean / q16 / q84), the
same recipe as the player models in 11_train_model.py:

    margin  home_score - away_score    (sign = winner, size = spread)
    total   home_score + away_score    (the over/under total)

Everything downstream derives from those two, so it can never self-contradict:

    P(home win) = Phi(margin / sigma_margin)
    home score  = (total + margin) / 2
    away score  = (total - margin) / 2

Residual-over-anchor, like the player models: the margin model learns the
adjustment over the teams' season net-rating gap (so home-court advantage is
literally the residual it learns first), and the total model learns the
adjustment over the four-factor scoring pace of both teams. Anchors are
rate-based and stationary, so league-wide scoring drift between seasons can't
bias the model.

Backtest: train on seasons before TEST_SEASON, test on TEST_SEASON, report
against naive baselines (home-team-always-wins; anchor-only margin). Then refit
on ALL rows and ship those, exactly like 11.

Outputs: models/game_{margin,total}_{mean,q16,q84}.joblib + models/game_metadata.json.
"""

import os
import json
import math

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_CSV = os.path.join(HERE, "game_training_data.csv")
MODEL_DIR = os.path.join(HERE, "models")

TEST_SEASON = "2025-26"

NUMERIC_FEATURES = [
    "home_days_rest", "away_days_rest",
    "home_season_games", "away_season_games",
    "home_season_ppg", "home_season_papg", "home_season_net", "home_season_win_pct",
    "away_season_ppg", "away_season_papg", "away_season_net", "away_season_win_pct",
    "home_l10_ppg", "home_l10_papg", "home_l10_win_pct",
    "away_l10_ppg", "away_l10_papg", "away_l10_win_pct",
    "home_l5_ppg", "home_l5_papg",
    "away_l5_ppg", "away_l5_papg",
]
CATEGORICAL_FEATURES = ["season_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

Q_LOW, Q_HIGH = 0.159, 0.841
# NBA margins have a game-to-game std around 13 points; don't let the predicted
# spread collapse below something physically plausible.
SIGMA_FLOOR_MARGIN = 8.0
SIGMA_FLOOR_TOTAL = 10.0


def make_model(loss, quantile=None):
    """A HistGradientBoostingRegressor with our standard settings."""
    return HistGradientBoostingRegressor(
        loss=loss,
        quantile=quantile,
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        categorical_features=CATEGORICAL_FEATURES,
    )


def prepare(df):
    """Return X (typed for the model); categoricals as category dtype."""
    X = df[FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    return X


def anchors_for(df):
    """Return (anchor_margin, anchor_total) Series for a frame.

    margin anchor: the season net-rating gap (home net minus away net). Zero
    for two equally-good teams -- home-court advantage lives in the residual.
    total anchor: average of both teams' combined scoring pace (points scored
    + allowed covers both offense and defense quality).
    """
    a_margin = (df["home_season_net"] - df["away_season_net"]).astype(float)
    a_total = (
        df["home_season_ppg"] + df["home_season_papg"]
        + df["away_season_ppg"] + df["away_season_papg"]
    ).astype(float) / 2.0
    return a_margin, a_total


def normal_cdf(z):
    """Standard-normal CDF, vectorized (no scipy needed)."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z) / math.sqrt(2.0)))


def main():
    if not os.path.exists(TRAINING_CSV):
        raise SystemExit(f"Missing {TRAINING_CSV}. Run 12_build_game_training_data.py first.")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(TRAINING_CSV)
    print(f"Loaded {len(df)} games. Seasons: {sorted(df['season'].dropna().unique())}\n")

    train_df = df[df["season"] != TEST_SEASON]
    test_df = df[df["season"] == TEST_SEASON]
    if test_df.empty or train_df.empty:
        raise SystemExit(
            f"Need both a train split and a '{TEST_SEASON}' test split; "
            f"got train={len(train_df)} test={len(test_df)}."
        )
    print(f"Train rows: {len(train_df)} (seasons < {TEST_SEASON})")
    print(f"Test rows:  {len(test_df)} ({TEST_SEASON}, held out)\n")

    X_train, X_test, X_full = prepare(train_df), prepare(test_df), prepare(df)

    metadata = {
        "test_season": TEST_SEASON,
        "prediction_scheme": "residual_over_anchor",
        "anchors": {
            "margin": "home_season_net - away_season_net",
            "total": "(home_ppg + home_papg + away_ppg + away_papg) / 2",
        },
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "features": FEATURES,
        "sigma_floor": {"margin": SIGMA_FLOOR_MARGIN, "total": SIGMA_FLOOR_TOTAL},
        "quantiles": {"low": Q_LOW, "high": Q_HIGH},
        "production_trained_on": "all_seasons",
        "production_seasons": sorted(s for s in df["season"].dropna().unique()),
        "production_rows": len(df),
        "backtest_train_rows": len(train_df),
        "backtest_test_rows": len(test_df),
        "metrics": {},
    }

    a_margin_train, a_total_train = anchors_for(train_df)
    a_margin_test, a_total_test = anchors_for(test_df)
    a_margin_full, a_total_full = anchors_for(df)

    preds = {}
    for target, anchor_train, anchor_test, anchor_full, floor in [
        ("margin", a_margin_train, a_margin_test, a_margin_full, SIGMA_FLOOR_MARGIN),
        ("total", a_total_train, a_total_test, a_total_full, SIGMA_FLOOR_TOTAL),
    ]:
        col = f"target_{target}"
        y_test = test_df[col].to_numpy()
        resid_train = train_df[col].to_numpy() - anchor_train.to_numpy()

        print("=" * 64)
        print(f"TARGET: {target.upper()}")
        print("-" * 64)

        mean_model = make_model("squared_error").fit(X_train, resid_train)
        q16_model = make_model("quantile", Q_LOW).fit(X_train, resid_train)
        q84_model = make_model("quantile", Q_HIGH).fit(X_train, resid_train)

        anchor_np = anchor_test.to_numpy()
        pred = anchor_np + mean_model.predict(X_test)
        q16 = anchor_np + q16_model.predict(X_test)
        q84 = anchor_np + q84_model.predict(X_test)
        sigma = np.maximum((q84 - q16) / 2.0, floor)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        bias = float(np.mean(pred - y_test))
        anchor_mae = mean_absolute_error(y_test, anchor_np)
        coverage = float(np.mean((y_test >= q16) & (y_test <= q84)))

        print(f"  [backtest] Model MAE {mae:6.3f} | RMSE {rmse:6.3f} | bias {bias:+.3f}")
        print(f"  [backtest] Anchor-only MAE {anchor_mae:6.3f}  "
              f"(improvement {100 * (anchor_mae - mae) / anchor_mae:+.1f}%)")
        print(f"  [backtest] Interval [q16,q84] coverage: {coverage * 100:4.1f}%  (target ~68.2%)")

        preds[target] = {"pred": pred, "sigma": sigma, "y": y_test}
        metadata["metrics"][target] = {
            "model_mae": round(mae, 4),
            "model_rmse": round(rmse, 4),
            "bias": round(bias, 4),
            "anchor_mae": round(anchor_mae, 4),
            "improvement_vs_anchor_pct": round(100 * (anchor_mae - mae) / anchor_mae, 2),
            "interval_coverage": round(coverage, 4),
        }

        # Refit on ALL rows and ship those (metrics above stay from the backtest).
        resid_full = df[col].to_numpy() - anchor_full.to_numpy()
        joblib.dump(make_model("squared_error").fit(X_full, resid_full),
                    os.path.join(MODEL_DIR, f"game_{target}_mean.joblib"))
        joblib.dump(make_model("quantile", Q_LOW).fit(X_full, resid_full),
                    os.path.join(MODEL_DIR, f"game_{target}_q16.joblib"))
        joblib.dump(make_model("quantile", Q_HIGH).fit(X_full, resid_full),
                    os.path.join(MODEL_DIR, f"game_{target}_q84.joblib"))
        print(f"  [production] refit on all {len(df)} rows -> saved")

    # --- Winner metrics, derived from the margin model ----------------------
    print("=" * 64)
    print("WINNER (derived from margin)")
    print("-" * 64)
    m = preds["margin"]
    y_win = test_df["target_home_win"].to_numpy()
    p_home = normal_cdf(m["pred"] / m["sigma"])
    acc = float(np.mean((m["pred"] > 0) == (y_win == 1)))
    brier = float(np.mean((p_home - y_win) ** 2))
    home_always = float(np.mean(y_win))
    anchor_acc = float(np.mean((a_margin_test.to_numpy() > 0) == (y_win == 1)))

    print(f"  [backtest] Winner accuracy:  {acc * 100:4.1f}%")
    print(f"  [backtest] Baselines: home-team-always {home_always * 100:4.1f}% | "
          f"anchor-margin-only {anchor_acc * 100:4.1f}%")
    print(f"  [backtest] Brier score: {brier:.4f}  (0.25 = coin flip, lower is better)")

    metadata["metrics"]["winner"] = {
        "accuracy": round(acc, 4),
        "baseline_home_always": round(home_always, 4),
        "baseline_anchor_margin": round(anchor_acc, 4),
        "brier": round(brier, 4),
    }

    with open(os.path.join(MODEL_DIR, "game_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 64)
    print(f"DONE. Game models + game_metadata.json written to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

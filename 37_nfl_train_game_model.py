"""
Train and BACKTEST the NFL game-outcome models from nfl_game_training_data.csv.

    python 37_nfl_train_game_model.py

The football sibling of 13_train_game_model.py. Three targets, each with
mean/q16/q84 gradient-boosted models:

    margin    home_score - away_score     (sign = winner, size = spread)
    total     home_score + away_score      (the over/under total)
    total_td  home + away offensive TDs    (the touchdown market)

Everything downstream is derived so it can't self-contradict:

    P(home win) = Phi(margin / sigma_margin)
    P(tie)      ~ tie base-rate weighted by how close the margin is to 0
    home score  = (total + margin) / 2 ;  away score = (total - margin) / 2
    home TDs    = total_td * home_score / (home_score + away_score)

Residual-over-anchor, like every other model here:
    margin anchor    = home_rating - away_rating + home-field edge  (SRS gap)
    total anchor     = (home_ppg + home_papg + away_ppg + away_papg) / 2
    total_td anchor  = total anchor / 7      (a touchdown is worth ~7 points)

Backtest holds out the most recent season when two+ are present; otherwise it
ships on everything. Outputs: models/nfl_game_{margin,total,td}_{mean,q16,q84}
.joblib + models/nfl_game_metadata.json.
"""

import os
import json
import math

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import nfl_common as nc


HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_CSV = os.path.join(HERE, "nfl_game_training_data.csv")
MODEL_DIR = os.path.join(HERE, "models")

NUMERIC_FEATURES = [
    "home_days_rest", "away_days_rest",
    "home_rating", "away_rating",
    "home_season_games", "away_season_games",
    "home_season_ppg", "home_season_papg", "home_season_net", "home_season_win_pct",
    "away_season_ppg", "away_season_papg", "away_season_net", "away_season_win_pct",
    "home_l5_ppg", "home_l5_papg", "away_l5_ppg", "away_l5_papg",
]
CATEGORICAL_FEATURES = ["season_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

Q_LOW, Q_HIGH = 0.159, 0.841
SIGMA_FLOOR = {"margin": 9.0, "total": 9.0, "td": 1.2}


def make_model(loss, quantile=None):
    return HistGradientBoostingRegressor(
        loss=loss, quantile=quantile, learning_rate=0.05, max_iter=400,
        max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, random_state=42,
        categorical_features=CATEGORICAL_FEATURES,
    )


def prepare(df):
    X = df[FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    return X


def anchors_for(df):
    """(margin, total, total_td) anchor Series, mirroring the engine."""
    rating_gap = (df["home_rating"] - df["away_rating"])
    # Early rows may miss a rating -> fall back to the season net gap.
    rating_gap = rating_gap.fillna(df["home_season_net"] - df["away_season_net"]).fillna(0.0)
    a_margin = rating_gap.astype(float) + nc.HFA_POINTS
    a_total = (df["home_season_ppg"] + df["home_season_papg"]
               + df["away_season_ppg"] + df["away_season_papg"]).astype(float) / 2.0
    a_td = a_total / 7.0
    return a_margin, a_total, a_td


def normal_cdf(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z) / math.sqrt(2.0)))


def main():
    if not os.path.exists(TRAINING_CSV):
        raise SystemExit(f"Missing {TRAINING_CSV}. Run 36_nfl_build_game_training.py first.")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(TRAINING_CSV)
    df["season"] = df["season"].astype(str)
    seasons = sorted(df["season"].dropna().unique())
    print(f"Loaded {len(df)} games. Seasons: {seasons}\n")
    if df.empty:
        raise SystemExit("No game rows -- backfill the schedule/logs first.")

    test_season = seasons[-1] if len(seasons) >= 2 else None
    train_df = df[df["season"] != test_season] if test_season else df
    test_df = df[df["season"] == test_season] if test_season else df.iloc[0:0]

    X_full = prepare(df)
    X_train = prepare(train_df)
    X_test = prepare(test_df) if not test_df.empty else None

    a_margin_f, a_total_f, a_td_f = anchors_for(df)
    a_margin_tr, a_total_tr, a_td_tr = anchors_for(train_df)
    anchors_full = {"margin": a_margin_f, "total": a_total_f, "td": a_td_f}
    anchors_train = {"margin": a_margin_tr, "total": a_total_tr, "td": a_td_tr}
    if X_test is not None:
        a_margin_te, a_total_te, a_td_te = anchors_for(test_df)
        anchors_test = {"margin": a_margin_te, "total": a_total_te, "td": a_td_te}

    metadata = {
        "test_season": test_season,
        "prediction_scheme": "residual_over_anchor",
        "anchors": {
            "margin": "home_rating - away_rating + HFA",
            "total": "(home_ppg + home_papg + away_ppg + away_papg) / 2",
            "td": "total_anchor / 7",
        },
        "hfa_points": nc.HFA_POINTS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "features": FEATURES,
        "sigma_floor": SIGMA_FLOOR,
        "quantiles": {"low": Q_LOW, "high": Q_HIGH},
        "tie_base_rate": round(float(df["target_tie"].mean()), 5),
        "production_seasons": seasons,
        "production_rows": len(df),
        "metrics": {},
    }

    target_cols = {"margin": "target_margin", "total": "target_total", "td": "target_total_td"}
    backtest_pred = {}
    for target, col in target_cols.items():
        floor = SIGMA_FLOOR[target]
        # The TD target may be missing where logs weren't loaded; drop those rows.
        full_mask = df[col].notna()
        Xf = X_full[full_mask.to_numpy()]
        yf = df.loc[full_mask, col].to_numpy()
        af = anchors_full[target][full_mask].to_numpy()
        if len(yf) < 50:
            print(f"[{target}] only {len(yf)} labeled rows -- skipping this target.")
            continue

        print("=" * 60)
        print(f"TARGET: {target}")
        if X_test is not None:
            tr_mask = train_df[col].notna()
            te_mask = test_df[col].notna()
            if te_mask.any() and tr_mask.any():
                Xtr = X_train[tr_mask.to_numpy()]
                ytr = train_df.loc[tr_mask, col].to_numpy()
                atr = anchors_train[target][tr_mask].to_numpy()
                Xte = X_test[te_mask.to_numpy()]
                yte = test_df.loc[te_mask, col].to_numpy()
                ate = anchors_test[target][te_mask].to_numpy()
                mean_m = make_model("squared_error").fit(Xtr, ytr - atr)
                q16_m = make_model("quantile", Q_LOW).fit(Xtr, ytr - atr)
                q84_m = make_model("quantile", Q_HIGH).fit(Xtr, ytr - atr)
                pred = ate + mean_m.predict(Xte)
                q16 = ate + q16_m.predict(Xte)
                q84 = ate + q84_m.predict(Xte)
                sigma = np.maximum((q84 - q16) / 2.0, floor)
                mae = mean_absolute_error(yte, pred)
                anchor_mae = mean_absolute_error(yte, ate)
                coverage = float(np.mean((yte >= q16) & (yte <= q84)))
                print(f"  [backtest] MAE {mae:6.3f} | anchor MAE {anchor_mae:6.3f} "
                      f"({100*(anchor_mae-mae)/anchor_mae:+.1f}%) | coverage {coverage*100:4.1f}%")
                metadata["metrics"][target] = {
                    "model_mae": round(mae, 4), "anchor_mae": round(anchor_mae, 4),
                    "interval_coverage": round(coverage, 4),
                }
                if target == "margin":
                    backtest_pred["margin"] = {"pred": pred, "sigma": sigma,
                                               "y": yte, "anchor": ate}

        # Ship on all labeled rows.
        joblib.dump(make_model("squared_error").fit(Xf, yf - af),
                    os.path.join(MODEL_DIR, f"nfl_game_{target}_mean.joblib"))
        joblib.dump(make_model("quantile", Q_LOW).fit(Xf, yf - af),
                    os.path.join(MODEL_DIR, f"nfl_game_{target}_q16.joblib"))
        joblib.dump(make_model("quantile", Q_HIGH).fit(Xf, yf - af),
                    os.path.join(MODEL_DIR, f"nfl_game_{target}_q84.joblib"))
        print(f"  [production] refit on all {len(yf)} rows -> saved")

    # Winner metrics from the margin backtest.
    if "margin" in backtest_pred:
        m = backtest_pred["margin"]
        y_win = (m["y"] > 0).astype(int)
        acc = float(np.mean((m["pred"] > 0) == (m["y"] > 0)))
        anchor_acc = float(np.mean((m["anchor"] > 0) == (m["y"] > 0)))
        print("=" * 60)
        print(f"WINNER accuracy {acc*100:4.1f}%  (SRS-anchor {anchor_acc*100:4.1f}%)")
        metadata["metrics"]["winner"] = {"accuracy": round(acc, 4),
                                         "baseline_anchor": round(anchor_acc, 4)}

    with open(os.path.join(MODEL_DIR, "nfl_game_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("=" * 60)
    print(f"DONE. NFL game models + nfl_game_metadata.json written to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

"""Phase-2 duration model: alert time-to-all-clear via survival analysis.

Fits Kaplan-Meier (non-parametric baseline) and Cox proportional-hazards (with
covariates from the hourly feature matrix at alert-start time). Reports median
alert duration, C-index, and top hazard-ratio covariates.

Run: PYTHONUTF8=1 python run_survival.py
"""

from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config, features, index, loaders, survival


def main() -> None:
    t0 = time.time()
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)

    alerts = loaders.load_alerts()
    waves = loaders.load_massive_attacks()
    daily = loaders.load_missile_daily()

    start = alerts["start_utc"].min().floor("h").tz_convert("UTC").tz_localize(None)
    grid = index.build_master_index(start=start)
    grid = index.expand_alerts_to_grid(grid, alerts)
    fm = features.build_feature_matrix(grid, {"waves": waves, "daily": daily})
    print(f"features {fm.shape[1]} cols  {len(fm):,} rows  [{time.time()-t0:.0f}s]")

    surv_df = survival.build_survival_dataset(alerts, fm)
    print(f"survival dataset: {len(surv_df):,} events  "
          f"({surv_df['observed'].sum():,} uncensored, "
          f"{(~surv_df['observed'].astype(bool)).sum():,} censored)  [{time.time()-t0:.0f}s]")

    train, test = survival.temporal_split_events(surv_df)
    print(f"train {len(train):,}  test {len(test):,}")

    # Kaplan-Meier baseline
    kmf = survival.km_baseline(train)
    median_km = kmf.median_survival_time_
    print(f"\nKaplan-Meier median alert duration: {median_km:.1f} hours")

    # KM survival curve plot
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("Hours since alert start")
    ax.set_ylabel("P(alert still active)")
    ax.set_title("Kaplan-Meier: alert duration survival curve")
    fig.savefig(config.ARTIFACTS_DIR / "km_survival.png", bbox_inches="tight", dpi=120)
    plt.close(fig)

    # Cox PH model
    cph = survival.cox_model(train)
    print(f"\nCox PH  C-index: {cph.concordance_index_:.3f}")
    print(f"Top hazard ratios (exp(coef) > 1 = shorter alerts):")
    summary = cph.summary
    top = summary.nlargest(10, "exp(coef)")[["exp(coef)", "p"]]
    print(top.to_string())

    # Evaluate on test set
    test_pred = survival.predict_duration(cph, test)
    valid = test_pred.notna() & test["observed"].astype(bool)
    if valid.any():
        actual = test.loc[valid, "duration_hours"]
        predicted = test_pred[valid]
        mae = (actual - predicted).abs().mean()
        print(f"\nTest MAE (uncensored only): {mae:.1f} hours  (n={valid.sum()})")

    print(f"\nartifacts -> {config.ARTIFACTS_DIR}  [total {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

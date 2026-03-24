"""
Out-of-sample PMI test:
- Train all cluster logic on history up to 2022-12-31
- Backtest on 2023-2026

Purpose: longer OOS window and explicit sector->cluster mapping.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmi_cluster_backtest import (
    AllocationConfig,
    add_3m_forward_returns,
    add_3m_trailing_returns_for_training,
    backtest_pmi_strategy,
    build_cluster_criteria_profile,
    build_monthly_scores,
    build_phase_cluster_score_table,
    build_sector_features,
    classify_phases,
    cluster_and_score,
    load_monthly_master,
    plot_rolling_sharpe,
)

OUTPUT_DIR = Path(__file__).parent.parent / "pdf"


def plot_oos_cumulative(bt: pd.DataFrame, save_path: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bt["Date"], bt["cum_excess_index"], label="Cum index (3M forward)", lw=2)
    ax.plot(bt["Date"], bt["cum_excess_index_1m"], label="Cum index (next 1M)", lw=2, alpha=0.85)
    ax.set_title("PMI OOS cumulative performance (2023-2026)")
    ax.set_ylabel("Index")
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()


def main() -> None:
    cfg = AllocationConfig()

    train_end = pd.Timestamp("2022-12-31")
    test_start = pd.Timestamp("2023-01-01")
    test_end = pd.Timestamp("2026-12-31")

    df = load_monthly_master()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    excess_cols = [c for c in df.columns if c.startswith("EXCESS_")]
    if not excess_cols:
        raise ValueError("No EXCESS_ columns found in monthly dataset.")

    df = classify_phases(df, span=cfg.pmi_ewma_span)
    df, fwd_cols = add_3m_forward_returns(df, excess_cols, horizon=cfg.horizon_months)
    df, trail_cols = add_3m_trailing_returns_for_training(df, excess_cols, horizon=cfg.horizon_months)

    train_df = df[df["Date"] <= train_end].copy()
    if len(train_df) < 36:
        raise ValueError("Not enough training history before 2023.")

    feat_train = build_sector_features(train_df, fwd_cols)
    sector_clustered, cluster_table = cluster_and_score(feat_train, cfg)
    cluster_profile = build_cluster_criteria_profile(sector_clustered)

    ticker_to_cluster = dict(zip(sector_clustered["ticker"], sector_clustered["cluster_id"]))
    phase_cluster_scores = build_phase_cluster_score_table(train_df, ticker_to_cluster, cfg)

    test_df = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)].copy()
    if test_df.empty:
        raise ValueError("No rows in requested test window 2023-2026.")

    score_test = build_monthly_scores(
        test_df,
        phase_cluster_scores,
        trail_cols,
        ticker_to_cluster=ticker_to_cluster,
    )
    bt = backtest_pmi_strategy(test_df, score_test, cfg)

    print("\n=== OOS Setup ===")
    print(f"Train window: <= {train_end.date()}")
    print(f"Test window : {test_start.date()} -> {test_end.date()}")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    print("\n=== Sector -> Cluster Mapping (trained <=2022) ===")
    map_cols = ["ticker", "sector", "cluster_id", "cluster_score"]
    print(sector_clustered[map_cols].sort_values(["cluster_id", "ticker"]).to_string(index=False))

    print("\n=== Train Cluster Summary ===")
    print(cluster_table.to_string(index=False))

    print("\n=== Train Cluster Criteria Profile ===")
    print(cluster_profile.to_string(index=False))

    print("\n=== Stable Phase -> Cluster Score (trained up to 2022) ===")
    print(phase_cluster_scores.sort_values(["Cycle_Phase", "cluster_score"]).to_string(index=False))

    if bt.empty:
        print("\nNo OOS backtest rows generated.")
        return

    hit_3m = float((bt["portfolio_fwd_excess_3m"] > 0).mean())
    avg_3m = float(bt["portfolio_fwd_excess_3m"].mean())
    vol_3m = float(bt["portfolio_fwd_excess_3m"].std(ddof=0))
    ir_3m = avg_3m / vol_3m if vol_3m > 0 else np.nan

    hit_1m = float((bt["portfolio_next_1m_excess"] > 0).mean())
    avg_1m = float(bt["portfolio_next_1m_excess"].mean())
    vol_1m = float(bt["portfolio_next_1m_excess"].std(ddof=0))
    ir_1m_ann = (avg_1m / vol_1m) * np.sqrt(12) if vol_1m > 0 else np.nan

    print("\n=== OOS Backtest Summary (2023-2026) ===")
    print(f"Observations:       {len(bt)}")
    print(f"Hit rate 3M fwd:    {hit_3m:.1%}")
    print(f"Avg 3M alpha:       {avg_3m:.3%}")
    print(f"Vol 3M alpha:       {vol_3m:.3%}")
    print(f"IR 3M:              {ir_3m:.2f}")
    print(f"Final index (3M):   {bt['cum_excess_index'].iloc[-1]:.3f}")
    print(f"Hit rate next 1M:   {hit_1m:.1%}")
    print(f"Avg next 1M alpha:  {avg_1m:.3%}")
    print(f"Vol next 1M alpha:  {vol_1m:.3%}")
    print(f"Ann. IR next 1M:    {ir_1m_ann:.2f}")
    print(f"Final index (1M):   {bt['cum_excess_index_1m'].iloc[-1]:.3f}")

    rolling_window = max(6, min(12, len(bt) // 2))
    print(f"Rolling Sharpe window used: {rolling_window} months")

    bt = plot_rolling_sharpe(
        bt,
        window_months=rolling_window,
        save_path=OUTPUT_DIR / "pmi_strategy_rolling_1y_sharpe_oos_2023_2026.pdf",
    )

    plot_oos_cumulative(
        bt,
        save_path=OUTPUT_DIR / "pmi_oos_cumulative_2023_2026.pdf",
    )


if __name__ == "__main__":
    main()

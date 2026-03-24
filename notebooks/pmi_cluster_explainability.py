"""
Explainability notebook/script for PMI clusters.

Goal: provide a transparent, rule-based interpretation of each cluster with
relative (z-score) criteria and a textual "why" explanation.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from pmi_cluster_backtest import (
    AllocationConfig,
    add_3m_forward_returns,
    build_sector_features,
    classify_phases,
    cluster_and_score,
    load_monthly_master,
)


def build_explainability_table(sector_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Build a cluster-level table with z-score based interpretation.

    Features used for interpretation:
      - hit_ratio
      - avg_fwd_3m
      - downside_mean
      - recession sensitivity: p_out_3_Recession
      - recovery sensitivity: p_out_4_Recovery
    """
    cols = [
        "hit_ratio",
        "avg_fwd_3m",
        "downside_mean",
        "p_out_3_Recession",
        "p_out_4_Recovery",
    ]

    agg = (
        sector_clustered.groupby(["cluster_id", "cluster_score"], as_index=False)[cols]
        .mean()
        .sort_values("cluster_score")
    )

    # relative positioning vs other clusters
    for c in cols:
        mu = agg[c].mean()
        sd = agg[c].std(ddof=0)
        if sd == 0:
            agg[f"z_{c}"] = 0.0
        else:
            agg[f"z_{c}"] = (agg[c] - mu) / sd

    # explicit style from z-scores (more robust than hard absolute thresholds)
    def style(row: pd.Series) -> str:
        if row["z_hit_ratio"] >= 0.7 and row["z_avg_fwd_3m"] >= 0.7:
            return "Cyclique fort"
        if row["z_hit_ratio"] >= 0.2 and row["z_avg_fwd_3m"] >= 0.2:
            return "Cyclique modéré"
        if row["z_p_out_3_Recession"] > row["z_p_out_4_Recovery"] and row["z_avg_fwd_3m"] <= 0:
            return "Défensif récession"
        return "Neutre / hybride"

    def why(row: pd.Series) -> str:
        reasons = []
        if row["z_hit_ratio"] >= 0.5:
            reasons.append("hit_ratio élevé")
        if row["z_avg_fwd_3m"] >= 0.5:
            reasons.append("avg_fwd_3m élevé")
        if row["z_downside_mean"] <= -0.5:
            reasons.append("downside contrôlé")
        if row["z_p_out_3_Recession"] - row["z_p_out_4_Recovery"] >= 0.5:
            reasons.append("plus robuste en récession")
        if row["z_p_out_4_Recovery"] - row["z_p_out_3_Recession"] >= 0.5:
            reasons.append("plus favorable en recovery")
        if not reasons:
            reasons.append("profil intermédiaire")
        return "; ".join(reasons)

    agg["cluster_style_zscore"] = agg.apply(style, axis=1)
    agg["why"] = agg.apply(why, axis=1)

    return agg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain PMI clusters for a specific training window."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "oos_2023_2026", "oos_2024_2026"],
        default="full",
        help=(
            "full = all available history; "
            "oos_2023_2026 = train cutoff 2022-12-31; "
            "oos_2024_2026 = train cutoff 2023-12-31"
        ),
    )
    args = parser.parse_args()

    cfg = AllocationConfig()

    df = load_monthly_master()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    train_cutoff = None
    if args.mode == "oos_2023_2026":
        train_cutoff = pd.Timestamp("2022-12-31")
    elif args.mode == "oos_2024_2026":
        train_cutoff = pd.Timestamp("2023-12-31")

    if train_cutoff is not None:
        df = df[df["Date"] <= train_cutoff].copy()
        if df.empty:
            raise ValueError(f"No rows available up to training cutoff {train_cutoff.date()}.")

    excess_cols = [c for c in df.columns if c.startswith("EXCESS_")]
    if not excess_cols:
        raise ValueError("No EXCESS_ columns found in monthly dataset.")

    df = classify_phases(df, span=cfg.pmi_ewma_span)
    df, fwd_cols = add_3m_forward_returns(df, excess_cols, horizon=cfg.horizon_months)

    feat = build_sector_features(df, fwd_cols)
    sector_clustered, cluster_table = cluster_and_score(feat, cfg)

    explain = build_explainability_table(sector_clustered)

    print("\n=== Explainability Scope ===")
    if train_cutoff is None:
        print("Mode: full (all available history)")
    else:
        print(f"Mode: {args.mode} | Train cutoff: {train_cutoff.date()}")
    print(f"Rows used for clustering: {len(df)}")

    print("\n=== Cluster Summary ===")
    print(cluster_table.to_string(index=False))

    print("\n=== Explainability (z-score based) ===")
    cols_show = [
        "cluster_id", "cluster_score",
        "hit_ratio", "avg_fwd_3m", "downside_mean",
        "p_out_3_Recession", "p_out_4_Recovery",
        "cluster_style_zscore", "why",
    ]
    print(explain[cols_show].to_string(index=False))

    print("\n=== Sector -> Cluster Mapping ===")
    show_map = ["ticker", "sector", "cluster_id", "cluster_score"]
    print(sector_clustered[show_map].sort_values(["cluster_id", "ticker"]).to_string(index=False))


if __name__ == "__main__":
    main()

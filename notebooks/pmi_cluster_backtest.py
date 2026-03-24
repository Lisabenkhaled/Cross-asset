"""
PMI-only cluster + score + allocation backtest.

Goal
----
1) Build sector clusters from PMI conditional outperformance statistics.
2) Transform clusters into discrete notes/scores (1..5).
3) Convert scores into active tilts vs STOXX 600 initial sector weights.
4) Backtest a PMI-only allocation process with 3M forward logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stoxx600_initial_weights import normalized_initial_weights

SECTOR_NAMES = {
    "SXNP": "Industrial G&S",
    "SX7P": "Banks",
    "SXDP": "Health Care",
    "SXIP": "Insurance",
    "S600ENP": "Energy",
    "SX8P": "Technology",
    "S600FOP": "Food & Bev",
    "S600CPP": "Consumer Prod",
    "SX6P": "Utilities",
    "SXFP": "Financial Svc",
    "SXOP": "Construction",
    "SXKP": "Telecom",
    "SXPP": "Basic Resources",
    "SX4P": "Chemicals",
    "SXAP": "Autos & Parts",
    "SX86P": "Real Estate",
    "SXTP": "Travel & Leisure",
    "SXRP": "Retail",
    "SXMP": "Media",
    "S600PDP": "Pers & HH Goods",
}

DATA_DIR = Path(__file__).parent.parent / "data"
MONTHLY_PATH = DATA_DIR / "master_data_monthly.parquet"
DAILY_PATH = DATA_DIR / "master_data_daily.parquet"
OUTPUT_DIR = Path(__file__).parent.parent / "pdf"


def _monthly_compound(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    return float((1 + s).prod() - 1)


def load_monthly_master() -> pd.DataFrame:
    """
    Load monthly panel aligned with data_factory outputs.

    Priority:
      1) data/master_data_monthly.parquet
      2) data/master_data_daily.parquet -> resample to monthly
    """
    if MONTHLY_PATH.exists():
        df = pd.read_parquet(MONTHLY_PATH)
        if "Date" not in df.columns and df.index.name == "Date":
            df = df.reset_index()
        if "Date" not in df.columns:
            raise ValueError(f"'Date' column not found in {MONTHLY_PATH}.")
        return df.sort_values("Date").reset_index(drop=True)

    if DAILY_PATH.exists():
        daily = pd.read_parquet(DAILY_PATH)
        if "Date" not in daily.columns and daily.index.name == "Date":
            daily = daily.reset_index()
        if "Date" not in daily.columns:
            raise ValueError(f"'Date' column not found in {DAILY_PATH}.")

        daily["Date"] = pd.to_datetime(daily["Date"])
        daily = daily.sort_values("Date").set_index("Date")

        if "PMI" not in daily.columns:
            raise ValueError(f"'PMI' column not found in {DAILY_PATH}.")

        ret_cols = [c for c in daily.columns if c == "BENCHMARK" or c.startswith("EXCESS_")]
        if not ret_cols:
            raise ValueError(f"No BENCHMARK/EXCESS_ columns found in {DAILY_PATH}.")

        monthly_rets = daily[ret_cols].resample("ME").apply(_monthly_compound)
        monthly_pmi = daily["PMI"].resample("ME").last().ffill()
        monthly = pd.concat([monthly_pmi, monthly_rets], axis=1).dropna(subset=["PMI"])
        return monthly.reset_index()

    raise FileNotFoundError(
        f"Missing monthly and daily datasets. Expected {MONTHLY_PATH} or {DAILY_PATH}. "
        "Run `python src/data_pipeline/data_factory.py` first."
    )


@dataclass(slots=True)
class AllocationConfig:
    horizon_months: int = 3
    pmi_ewma_span: int = 3
    n_clusters: int = 5
    score_bins: tuple[float, float, float, float] = (0.40, 0.48, 0.56, 0.64)
    # note -> active multiplier applied to baseline weight
    score_to_multiplier: dict[int, float] | None = None
    min_weight_pct: float = 0.10

    def __post_init__(self):
        if self.score_to_multiplier is None:
            self.score_to_multiplier = {
                1: 0.75,  # underweight
                2: 0.90,
                3: 1.00,  # neutral
                4: 1.10,
                5: 1.25,  # overweight
            }


def classify_phases(df: pd.DataFrame, pmi_col: str = "PMI", span: int = 3) -> pd.DataFrame:
    """Classify each month in 4 PMI cycle regimes."""
    out = df.copy()
    out["PMI_Smoothed"] = out[pmi_col].ewm(span=span, adjust=False).mean()
    out["PMI_Change"] = out["PMI_Smoothed"].diff()

    conds = [
        (out["PMI_Smoothed"] >= 50) & (out["PMI_Change"] >= 0),
        (out["PMI_Smoothed"] >= 50) & (out["PMI_Change"] < 0),
        (out["PMI_Smoothed"] < 50) & (out["PMI_Change"] < 0),
        (out["PMI_Smoothed"] < 50) & (out["PMI_Change"] >= 0),
    ]
    labels = ["1_Expansion", "2_Slowdown", "3_Recession", "4_Recovery"]
    out["Cycle_Phase"] = np.select(conds, labels, default="Unknown")
    return out


def add_3m_forward_returns(df: pd.DataFrame, excess_cols: list[str], horizon: int) -> tuple[pd.DataFrame, list[str]]:
    """Add forward-looking compounded returns over `horizon` months."""
    out = df.copy()
    fwd_cols: list[str] = []
    for col in excess_cols:
        target = f"{col}_{horizon}M_FWD"
        compounded = (1 + out[col]).rolling(window=horizon).apply(np.prod, raw=True) - 1
        out[target] = compounded.shift(-horizon)
        fwd_cols.append(target)
    return out, fwd_cols


def add_3m_trailing_returns_for_training(df: pd.DataFrame, excess_cols: list[str], horizon: int) -> tuple[pd.DataFrame, list[str]]:
    """
    Build trailing 3M compounded returns used only for probability estimation.

    At date t we use returns up to t-1 (shift(1)) so training inputs contain no
    future information relative to decision date t.
    """
    out = df.copy()
    trail_cols: list[str] = []
    for col in excess_cols:
        trail_name = f"{col}_{horizon}M_TRAIL"
        trailing = (1 + out[col]).rolling(window=horizon).apply(np.prod, raw=True) - 1
        out[trail_name] = trailing.shift(1)
        trail_cols.append(trail_name)
    return out, trail_cols


def _simple_kmeans(x: np.ndarray, n_clusters: int, seed: int = 42, n_iter: int = 100) -> np.ndarray:
    """Small numpy-only k-means (to avoid extra dependencies)."""
    if len(x) < n_clusters:
        raise ValueError("n_clusters cannot exceed number of sectors")

    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=n_clusters, replace=False)]

    for _ in range(n_iter):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)

        new_centers = []
        for k in range(n_clusters):
            members = x[labels == k]
            if len(members) == 0:
                new_centers.append(x[rng.integers(0, len(x))])
            else:
                new_centers.append(members.mean(axis=0))
        new_centers = np.vstack(new_centers)

        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return labels


def build_sector_features(df: pd.DataFrame, fwd_cols: list[str]) -> pd.DataFrame:
    """Feature engineering based only on PMI phases + 3M forward excess returns."""
    clean = df[df["Cycle_Phase"] != "Unknown"].dropna(subset=fwd_cols)

    features = {}
    for col in fwd_cols:
        ticker = col.replace("EXCESS_", "").replace("_3M_FWD", "")
        s = clean[col]

        # Regime-level outperform probabilities
        by_phase = clean.groupby("Cycle_Phase")[col].apply(lambda x: (x > 0).mean())
        phase_probs = {
            f"p_out_{phase}": float(by_phase.get(phase, np.nan))
            for phase in ["1_Expansion", "2_Slowdown", "3_Recession", "4_Recovery"]
        }

        # Additional PMI-only features (still using only phase+return relation)
        hit_ratio = float((s > 0).mean())
        avg_fwd = float(s.mean())
        downside = float(s[s < 0].mean()) if (s < 0).any() else 0.0

        features[ticker] = {
            "ticker": ticker,
            "sector": SECTOR_NAMES.get(ticker, ticker),
            "hit_ratio": hit_ratio,
            "avg_fwd_3m": avg_fwd,
            "downside_mean": downside,
            **phase_probs,
        }

    feat = pd.DataFrame.from_dict(features, orient="index")
    feat = feat.sort_values("ticker").reset_index(drop=True)
    return feat


def cluster_and_score(feat: pd.DataFrame, cfg: AllocationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster sectors and assign score 1..5 from cluster quality."""
    use_cols = [
        "hit_ratio", "avg_fwd_3m", "downside_mean",
        "p_out_1_Expansion", "p_out_2_Slowdown", "p_out_3_Recession", "p_out_4_Recovery",
    ]

    x = feat[use_cols].to_numpy(dtype=float)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0] = 1.0
    z = (x - mu) / sigma

    labels = _simple_kmeans(z, n_clusters=cfg.n_clusters, seed=42)
    out = feat.copy()
    out["cluster_id"] = labels

    # Cluster quality = blended signal, then rank clusters to note 1..5
    out["sector_signal"] = (
        0.45 * out["hit_ratio"]
        + 0.35 * (out["avg_fwd_3m"].rank(pct=True))
        + 0.20 * (out["p_out_4_Recovery"] - out["p_out_3_Recession"])
    )

    cluster_signal = out.groupby("cluster_id")["sector_signal"].mean().sort_values()
    cluster_to_score = {
        cluster: score
        for score, cluster in enumerate(cluster_signal.index, start=1)
    }
    out["cluster_score"] = out["cluster_id"].map(cluster_to_score).astype(int)

    cluster_table = (
        out.groupby("cluster_id", as_index=False)
        .agg(
            n_sectors=("ticker", "count"),
            avg_hit_ratio=("hit_ratio", "mean"),
            avg_fwd_3m=("avg_fwd_3m", "mean"),
            avg_signal=("sector_signal", "mean"),
            score=("cluster_score", "first"),
        )
        .sort_values("score")
    )

    return out, cluster_table


def build_cluster_criteria_profile(sector_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Human-readable cluster definition based on average feature levels.

    This answers: "cluster k correspond à quels critères ?"
    """
    feature_cols = [
        "hit_ratio",
        "avg_fwd_3m",
        "downside_mean",
        "p_out_1_Expansion",
        "p_out_2_Slowdown",
        "p_out_3_Recession",
        "p_out_4_Recovery",
    ]
    prof = (
        sector_clustered.groupby(["cluster_id", "cluster_score"], as_index=False)[feature_cols]
        .mean()
        .sort_values("cluster_score")
    )

    # Text label for interpretation (simple rule-of-thumb)
    def _label(row: pd.Series) -> str:
        if row["avg_fwd_3m"] > 0 and row["hit_ratio"] >= 0.55:
            return "Pro-cyclique fort"
        if row["avg_fwd_3m"] > 0 and row["hit_ratio"] >= 0.50:
            return "Pro-cyclique modéré"
        if row["avg_fwd_3m"] <= 0 and row["p_out_3_Recession"] >= row["p_out_4_Recovery"]:
            return "Défensif / récession"
        return "Neutre / mixte"

    prof["cluster_style"] = prof.apply(_label, axis=1)
    return prof


def _score_from_probability(p: float, bins: tuple[float, float, float, float]) -> int:
    b1, b2, b3, b4 = bins
    if p < b1:
        return 1
    if p < b2:
        return 2
    if p < b3:
        return 3
    if p < b4:
        return 4
    return 5


def build_phase_cluster_score_table(
    df: pd.DataFrame,
    ticker_to_cluster: dict[str, int],
    cfg: AllocationConfig,
) -> pd.DataFrame:
    """
    Build a *stable* mapping: phase -> cluster_score.

    Process:
      historical analysis -> fixed clusters -> score clusters per phase.
    """
    phases = ["1_Expansion", "2_Slowdown", "3_Recession", "4_Recovery"]
    clusters = sorted(set(ticker_to_cluster.values()))
    rows: list[dict[str, float | int | str]] = []

    clean = df[df["Cycle_Phase"].isin(phases)].copy()
    for phase_t in phases:
        phase_hist = clean[clean["Cycle_Phase"] == phase_t]
        if phase_hist.empty:
            continue
        cluster_pout: dict[int, float] = {}
        for cluster_id in clusters:
            members = [t for t, c in ticker_to_cluster.items() if c == cluster_id]
            member_cols = [f"EXCESS_{t}_{cfg.horizon_months}M_TRAIL" for t in members]
            available_cols = [c for c in member_cols if c in phase_hist.columns]
            if not available_cols:
                cluster_pout[cluster_id] = np.nan
                continue

            m = phase_hist[available_cols].to_numpy(dtype=float)
            cluster_pout[cluster_id] = float(np.nanmean(m > 0))

        valid_clusters = [k for k, v in cluster_pout.items() if pd.notna(v)]
        ranked = sorted(valid_clusters, key=lambda k: cluster_pout[k])
        cluster_to_score = {cl: i + 1 for i, cl in enumerate(ranked)}

        for cluster_id in clusters:
            rows.append(
                {
                    "Cycle_Phase": phase_t,
                    "cluster_id": cluster_id,
                    "cluster_pout": cluster_pout.get(cluster_id, np.nan),
                    "cluster_score": int(cluster_to_score.get(cluster_id, 3)),
                }
            )

    return pd.DataFrame(rows)


def build_monthly_scores(
    df: pd.DataFrame,
    phase_cluster_scores: pd.DataFrame,
    trail_cols: list[str],
    ticker_to_cluster: dict[str, int],
) -> pd.DataFrame:
    """
    Apply stable phase->cluster_score map to each date.

    Cluster definitions and cluster ids are fixed through time.
    """
    phases = ["1_Expansion", "2_Slowdown", "3_Recession", "4_Recovery"]
    history = df[df["Cycle_Phase"].isin(phases)].copy()

    pcs = phase_cluster_scores.pivot(
        index="Cycle_Phase", columns="cluster_id", values="cluster_score"
    )
    pcp = phase_cluster_scores.pivot(
        index="Cycle_Phase", columns="cluster_id", values="cluster_pout"
    )

    rows: list[dict[str, float | int | str]] = []
    for _, row in history.iterrows():
        phase_t = row["Cycle_Phase"]
        entry: dict[str, float | int | str] = {
            "Date": row["Date"],
            "Cycle_Phase": phase_t,
        }
        for trail_col in trail_cols:
            ticker = trail_col.replace("EXCESS_", "").replace("_3M_TRAIL", "")
            cl = ticker_to_cluster.get(ticker)
            entry[f"score_{ticker}"] = int(pcs.loc[phase_t, cl])
            entry[f"pout_{ticker}"] = float(pcp.loc[phase_t, cl])
        rows.append(entry)
    return pd.DataFrame(rows)


def score_to_weights(score_row: pd.Series, cfg: AllocationConfig) -> dict[str, float]:
    """Convert per-sector notes into fully-invested weights around initial STOXX600 mix."""
    base = normalized_initial_weights()
    raw = {}

    for ticker, w0 in base.items():
        score = int(score_row.get(f"score_{ticker}", 3))
        mult = cfg.score_to_multiplier[score]
        raw_w = max(w0 * mult, cfg.min_weight_pct)
        raw[ticker] = raw_w

    total = sum(raw.values())
    return {k: 100 * v / total for k, v in raw.items()}


def backtest_pmi_strategy(df: pd.DataFrame, score_df: pd.DataFrame, cfg: AllocationConfig) -> pd.DataFrame:
    """Backtest monthly using notes at t and realised 3M fwd excess returns from t."""
    by_date = df.set_index("Date")
    index_by_date = {d: i for i, d in enumerate(df["Date"].tolist())}
    pnl_rows = []

    for _, srow in score_df.iterrows():
        dt = srow["Date"]
        if dt not in by_date.index:
            continue
        market_row = by_date.loc[dt]
        idx = index_by_date.get(dt)
        next_row = None
        if idx is not None and idx + 1 < len(df):
            next_row = df.iloc[idx + 1]

        weights = score_to_weights(srow, cfg)
        weighted_fwd = 0.0
        weighted_1m = 0.0
        for ticker, weight in weights.items():
            fwd_col = f"EXCESS_{ticker}_{cfg.horizon_months}M_FWD"
            r = market_row.get(fwd_col, np.nan)
            if pd.notna(r):
                weighted_fwd += (weight / 100.0) * float(r)
            if next_row is not None:
                one_m_col = f"EXCESS_{ticker}"
                r1 = next_row.get(one_m_col, np.nan)
                if pd.notna(r1):
                    weighted_1m += (weight / 100.0) * float(r1)

        pnl_rows.append(
            {
                "Date": dt,
                "Cycle_Phase": srow["Cycle_Phase"],
                "portfolio_fwd_excess_3m": weighted_fwd,
                "portfolio_fwd_excess_3m_pct": weighted_fwd * 100,
                "portfolio_next_1m_excess": weighted_1m,
                "portfolio_next_1m_excess_pct": weighted_1m * 100,
            }
        )

    bt = pd.DataFrame(pnl_rows).sort_values("Date")
    bt["cum_excess_index"] = (1 + bt["portfolio_fwd_excess_3m"]).cumprod()
    bt["cum_excess_index_1m"] = (1 + bt["portfolio_next_1m_excess"]).cumprod()
    return bt


def plot_rolling_sharpe(
    bt: pd.DataFrame,
    *,
    window_months: int = 12,
    annualization_factor: float = np.sqrt(12),
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Plot rolling Sharpe on strategy 3M-forward excess returns.

    We annualize a rolling Sharpe computed from the next-1M strategy returns:
        Sharpe_t = mean(r_{t-window+1:t}) / std(r_{t-window+1:t}) * sqrt(12)
    """
    out = bt.copy()
    series = out["portfolio_next_1m_excess"]
    roll_mean = series.rolling(window_months).mean()
    roll_std = series.rolling(window_months).std(ddof=0)
    out["rolling_1y_sharpe"] = (roll_mean / roll_std) * annualization_factor

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(out["Date"], out["rolling_1y_sharpe"], linewidth=1.8, color="#2C6DA4")
    ax.axhline(0.0, color="black", ls="--", lw=1.2, alpha=0.8)
    ax.set_title("Rolling 1Y Sharpe Ratio - PMI Strategy", fontsize=18)
    ax.set_ylabel("Sharpe Ratio", fontsize=14)
    ax.grid(True, ls="--", alpha=0.35)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.show()

    return out


def print_latest_regime_view(
    score_df: pd.DataFrame,
    sector_clustered: pd.DataFrame,
) -> None:
    """Print today's regime and cluster over/under-weight guidance."""
    if score_df.empty:
        return
    latest = score_df.sort_values("Date").iloc[-1]
    phase = latest["Cycle_Phase"]
    print("\n=== Latest Regime Allocation View ===")
    print(f"Current phase: {phase}")

    cl_table = sector_clustered.copy()
    cl_table["regime_score"] = cl_table["ticker"].map(
        lambda t: int(latest.get(f"score_{t}", 3))
    )
    by_cluster = (
        cl_table.groupby("cluster_id", as_index=False)
        .agg(
            regime_score=("regime_score", "mean"),
            sectors=("sector", lambda s: ", ".join(sorted(s.tolist()))),
        )
        .sort_values("regime_score", ascending=False)
    )
    print(by_cluster.to_string(index=False))


def main() -> None:
    cfg = AllocationConfig()

    df = load_monthly_master()
    if "PMI" not in df.columns:
        raise ValueError("Expected 'PMI' column in monthly dataset.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    excess_cols = [c for c in df.columns if c.startswith("EXCESS_")]
    if not excess_cols:
        raise ValueError("No EXCESS_ columns found in monthly dataset.")

    df = classify_phases(df, span=cfg.pmi_ewma_span)
    df, fwd_cols = add_3m_forward_returns(df, excess_cols, horizon=cfg.horizon_months)
    df, trail_cols = add_3m_trailing_returns_for_training(df, excess_cols, horizon=cfg.horizon_months)

    # Use full sample for static cluster analysis (exploration)
    feat = build_sector_features(df, fwd_cols)
    sector_clustered, cluster_table = cluster_and_score(feat, cfg)
    cluster_profile = build_cluster_criteria_profile(sector_clustered)
    ticker_to_cluster = dict(zip(sector_clustered["ticker"], sector_clustered["cluster_id"]))

    phase_cluster_scores = build_phase_cluster_score_table(df, ticker_to_cluster, cfg)
    score_df = build_monthly_scores(
        df,
        phase_cluster_scores,
        trail_cols,
        ticker_to_cluster=ticker_to_cluster,
    )
    bt = backtest_pmi_strategy(df, score_df, cfg)

    print("\n=== Cluster Summary (PMI-only features) ===")
    print(cluster_table.to_string(index=False))

    show_cols = ["ticker", "sector", "cluster_id", "cluster_score", "hit_ratio", "avg_fwd_3m"]
    print("\n=== Sector -> Cluster/Score Mapping ===")
    print(sector_clustered[show_cols].sort_values(["cluster_score", "ticker"]).to_string(index=False))
    print("\n=== Cluster Criteria Profile ===")
    print(cluster_profile.to_string(index=False))
    print("\n=== Stable Phase -> Cluster Score Mapping ===")
    print(phase_cluster_scores.sort_values(["Cycle_Phase", "cluster_score"]).to_string(index=False))

    print("\n=== Backtest Summary (3M forward excess returns) ===")
    if len(bt) == 0:
        print("No backtest rows generated (check data availability).")
        return

    hit = float((bt["portfolio_fwd_excess_3m"] > 0).mean())
    avg = float(bt["portfolio_fwd_excess_3m"].mean())
    vol = float(bt["portfolio_fwd_excess_3m"].std(ddof=0))
    ir = avg / vol if vol > 0 else np.nan

    print(f"Observations: {len(bt)}")
    print(f"Hit rate:     {hit:.1%}")
    print(f"Avg 3M alpha: {avg:.3%}")
    print(f"Vol 3M alpha: {vol:.3%}")
    print(f"Info ratio:   {ir:.2f}")
    print(f"Final index:  {bt['cum_excess_index'].iloc[-1]:.3f}")
    one_m_hit = float((bt["portfolio_next_1m_excess"] > 0).mean())
    one_m_avg = float(bt["portfolio_next_1m_excess"].mean())
    one_m_vol = float(bt["portfolio_next_1m_excess"].std(ddof=0))
    one_m_ir = (one_m_avg / one_m_vol) * np.sqrt(12) if one_m_vol > 0 else np.nan
    print(f"Hit rate (next 1M): {one_m_hit:.1%}")
    print(f"Avg 1M alpha:       {one_m_avg:.3%}")
    print(f"Vol 1M alpha:       {one_m_vol:.3%}")
    print(f"Ann. IR (1M):       {one_m_ir:.2f}")
    print(f"Final 1M index:     {bt['cum_excess_index_1m'].iloc[-1]:.3f}")

    bt = plot_rolling_sharpe(
        bt,
        window_months=12,
        save_path=OUTPUT_DIR / "pmi_strategy_rolling_1y_sharpe.pdf",
    )
    latest_sharpe = bt["rolling_1y_sharpe"].dropna()
    if not latest_sharpe.empty:
        print(f"Latest rolling 1Y Sharpe: {latest_sharpe.iloc[-1]:.2f}")

    print_latest_regime_view(score_df, sector_clustered)


if __name__ == "__main__":
    main()

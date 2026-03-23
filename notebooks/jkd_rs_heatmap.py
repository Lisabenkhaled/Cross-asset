# %%
"""
JdK Relative Rotation Graph (RRG) for STOXX 600 sectors.

Reads daily returns from master_data_daily.parquet, reconstructs cumulative
price indices, resamples to weekly, and plots the standard RRG quadrant chart.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TICKER_MAP = {
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

# Sectors to exclude from the RRG (too small / illiquid)
_EXCLUDE_SECTORS = {"SXMP", "SXRP"}

DATA_PATH = Path(__file__).parent.parent / "data" / "master_data_daily.parquet"
OUTPUT_PDF = Path(__file__).parent.parent / "pdf" / "jkd_rrg_quadrant.pdf"


# ---------------------------------------------------------------------------
# Core analytics
# ---------------------------------------------------------------------------


def _returns_to_prices(returns: pd.DataFrame) -> pd.DataFrame:
    """Convert daily simple returns to cumulative price index (base=100)."""
    return (1 + returns).cumprod() * 100


def calculate_jkd_rs_metrics(
    prices: pd.DataFrame,
    benchmark_col: str,
    sector_cols: List[str],
    window: int = 14,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    JdK RS-Ratio and RS-Momentum.

    Args:
        prices: DataFrame of price levels (not returns).
        benchmark_col: Benchmark column name.
        sector_cols: Sector column names.
        window: Smoothing window in bars (weeks when fed weekly data).

    Returns:
        (rs_ratio, rs_momentum) DataFrames scaled around 100.
    """
    benchmark = prices[benchmark_col]
    raw_rs = prices[sector_cols].div(benchmark, axis=0)

    # 52-bar rolling z-score normalisation
    long_window = max(window * 4, 52)
    rs_mean = raw_rs.rolling(long_window, min_periods=long_window // 2).mean()
    rs_std = raw_rs.rolling(long_window, min_periods=long_window // 2).std(ddof=0)
    rs_std = rs_std.replace(0, np.nan)
    rs_norm = (raw_rs - rs_mean) / rs_std

    # RS-Ratio: EMA-smoothed z-score, rescaled to [~90, ~110]
    rs_ratio = 100 + rs_norm.ewm(span=window, adjust=False).mean() * 5

    # RS-Momentum: z-scored rate-of-change of RS-Ratio
    roc = rs_ratio.diff(window)
    roc_mean = roc.rolling(long_window, min_periods=long_window // 2).mean()
    roc_std = (
        roc.rolling(long_window, min_periods=long_window // 2)
        .std(ddof=0)
        .replace(0, np.nan)
    )
    rs_momentum = (
        100 + ((roc - roc_mean) / roc_std).ewm(span=window, adjust=False).mean() * 5
    )

    return rs_ratio, rs_momentum


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_rrg(
    rs_ratio: pd.DataFrame,
    rs_momentum: pd.DataFrame,
    tail_length: int = 8,
    title: str = "Relative Rotation Graph (RRG)",
    save_path: Optional[str | Path] = None,
):
    """Plot a standard RRG with quadrant backgrounds and sector tails."""
    df_x = rs_ratio.tail(tail_length).dropna(how="all")
    df_y = rs_momentum.tail(tail_length).dropna(how="all")

    if df_x.empty or df_y.empty:
        logger.error("No valid data to plot.")
        return

    latest_date = df_x.index[-1].strftime("%Y-%m-%d")
    fig, ax = plt.subplots(figsize=(14, 10))

    # Dynamic axis limits centered at 100
    x_pad = max(abs(df_x.min().min() - 100), abs(df_x.max().max() - 100)) * 1.15
    y_pad = max(abs(df_y.min().min() - 100), abs(df_y.max().max() - 100)) * 1.15
    x_pad, y_pad = max(x_pad, 5), max(y_pad, 5)

    ax.set_xlim(100 - x_pad, 100 + x_pad)
    ax.set_ylim(100 - y_pad, 100 + y_pad)

    # Quadrant backgrounds
    quad_cfg = [
        ((100, 100), "#d4edda", "LEADING"),
        ((100, 100 - y_pad), "#fff3cd", "WEAKENING"),
        ((100 - x_pad, 100 - y_pad), "#f8d7da", "LAGGING"),
        ((100 - x_pad, 100), "#cce5ff", "IMPROVING"),
    ]
    for (ox, oy), color, label in quad_cfg:
        ax.add_patch(
            patches.Rectangle((ox, oy), x_pad, y_pad, color=color, alpha=0.6, zorder=0)
        )
    ax.axhline(100, color="black", lw=1.5, zorder=1)
    ax.axvline(100, color="black", lw=1.5, zorder=1)

    text_kws = dict(
        fontsize=24, alpha=0.15, ha="center", va="center", weight="bold", zorder=1
    )
    ax.text(100 + x_pad * 0.5, 100 + y_pad * 0.5, "LEADING", **text_kws)
    ax.text(100 + x_pad * 0.5, 100 - y_pad * 0.5, "WEAKENING", **text_kws)
    ax.text(100 - x_pad * 0.5, 100 - y_pad * 0.5, "LAGGING", **text_kws)
    ax.text(100 - x_pad * 0.5, 100 + y_pad * 0.5, "IMPROVING", **text_kws)

    # Plot each sector
    sectors = df_x.columns
    cmap = cm.get_cmap("tab20", len(sectors))

    for i, sector in enumerate(sectors):
        x = df_x[sector].values
        y = df_y[sector].values
        c = cmap(i)

        ax.plot(x, y, color=c, lw=2, alpha=0.5, zorder=2)
        ax.scatter(x, y, s=np.linspace(10, 60, len(x)), color=c, alpha=0.5, zorder=3)
        ax.scatter(
            x[-1],
            y[-1],
            s=150,
            color=c,
            edgecolor="black",
            lw=1.5,
            zorder=4,
            label=sector,
        )
        ax.annotate(
            sector,
            (x[-1], y[-1]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
            zorder=5,
        )

    ax.set_xlabel("JdK RS-Ratio (Trend)", fontsize=12, weight="bold")
    ax.set_ylabel("JdK RS-Momentum", fontsize=12, weight="bold")
    ax.set_title(f"{title}\n(As of {latest_date})", fontsize=16, pad=15)
    ax.grid(True, ls="--", alpha=0.4, zorder=1)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title="Sectors",
        frameon=True,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"RRG saved to: {save_path}")

    plt.show()


# %%
if __name__ == "__main__":
    # 1. Load daily returns
    df = pd.read_parquet(DATA_PATH)

    # 2. Keep only sector return columns + BENCHMARK (drop PMI, EXCESS_*)
    sector_tickers = [
        c
        for c in df.columns
        if not c.startswith("EXCESS_") and c not in ("PMI", "BENCHMARK")
    ]
    sector_tickers = [s for s in sector_tickers if s not in _EXCLUDE_SECTORS]
    cols = ["BENCHMARK"] + sector_tickers
    df_rets = df[cols]

    # 3. Reconstruct price levels from returns
    df_prices = _returns_to_prices(df_rets)

    # 4. Rename tickers to readable names
    df_prices = df_prices.rename(columns=TICKER_MAP)
    benchmark = "BENCHMARK"
    sectors = [c for c in df_prices.columns if c != benchmark]

    logger.info(f"Benchmark: {benchmark} | {len(sectors)} sectors")

    # 5. Resample to weekly (end-of-week Friday)
    df_weekly = df_prices.resample("W-FRI").last().dropna(how="all")

    # 6. Compute JdK RS metrics (14-week window)
    rs_ratio, rs_momentum = calculate_jkd_rs_metrics(
        prices=df_weekly,
        benchmark_col=benchmark,
        sector_cols=sectors,
        window=14,
    )

    # 7. Plot
    plot_rrg(
        rs_ratio,
        rs_momentum,
        tail_length=14,
        title=f"Relative Rotation Graph (RRG) vs {benchmark}",
        save_path=str(OUTPUT_PDF),
    )

"""
PMI cycle analysis: classify economic phases and score sector forward excess returns.

Reads monthly data from master_data.parquet (produced by data_factory.py).
Outputs:
  - PMI phase trajectory plot
  - Scored heatmap: Win Rate * Avg Forward Excess Return by phase and sector
  - Strategy summary table
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

DATA_PATH = Path(__file__).parent.parent / "data" / "master_data.parquet"
OUTPUT_DIR = Path(__file__).parent.parent / "pdf"


# ---------------------------------------------------------------------------
# Cycle classification
# ---------------------------------------------------------------------------

def classify_phases(df: pd.DataFrame, pmi_col: str = "PMI", span: int = 3) -> pd.DataFrame:
    """
    Assign an economic phase using smoothed PMI level and momentum.

    Phases (investment-clock order):
        1_Expansion  : PMI >= 50, rising
        2_Slowdown   : PMI >= 50, falling
        3_Recession  : PMI <  50, falling
        4_Recovery   : PMI <  50, rising
    """
    df = df.copy()
    df["PMI_Smoothed"] = df[pmi_col].ewm(span=span, adjust=False).mean()
    df["PMI_Change"] = df["PMI_Smoothed"].diff()

    conditions = [
        (df["PMI_Smoothed"] >= 50) & (df["PMI_Change"] >= 0),
        (df["PMI_Smoothed"] >= 50) & (df["PMI_Change"] < 0),
        (df["PMI_Smoothed"] < 50) & (df["PMI_Change"] < 0),
        (df["PMI_Smoothed"] < 50) & (df["PMI_Change"] >= 0),
    ]
    phases = ["1_Expansion", "2_Slowdown", "3_Recession", "4_Recovery"]
    df["Cycle_Phase"] = np.select(conditions, phases, default="Unknown")
    return df


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------

def add_forward_returns(
    df: pd.DataFrame, excess_cols: list[str], horizon: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute forward compound excess returns over `horizon` months.

    Uses geometric compounding: prod(1 + r_t+1 ... r_t+h) - 1.
    Shifts returns forward so there is no look-ahead bias: at time t we see
    the return realised over [t+1, t+horizon].
    """
    fwd_cols = []
    for col in excess_cols:
        fwd_name = f"{col}_{horizon}M_FWD"
        # Rolling compound return, then shift so row t holds the FUTURE return
        compound = (1 + df[col]).rolling(window=horizon).apply(np.prod, raw=True) - 1
        df[fwd_name] = compound.shift(-horizon)
        fwd_cols.append(fwd_name)
    return df, fwd_cols


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_scores(df: pd.DataFrame, fwd_cols: list[str]) -> pd.DataFrame:
    """
    Score = Avg Forward Excess Return (%) * Win Rate (%).

    Returns a (phases x sectors) DataFrame.
    """
    clean = df[df["Cycle_Phase"] != "Unknown"].dropna(subset=fwd_cols)
    grouped = clean.groupby("Cycle_Phase")[fwd_cols]
    avg_ret = grouped.mean() * 100
    win_rate = grouped.apply(lambda x: (x > 0).mean()) * 100
    return avg_ret * win_rate


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_phase_diagram(df: pd.DataFrame, span: int = 3, save_path: Path | None = None):
    """PMI phase trajectory scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.plot(df["PMI_Smoothed"], df["PMI_Change"], color="gray", alpha=0.4, lw=1.5, zorder=1)

    # Arrow showing current direction
    if len(df) > 1:
        ax.annotate(
            "", xy=(df["PMI_Smoothed"].iloc[-1], df["PMI_Change"].iloc[-1]),
            xytext=(df["PMI_Smoothed"].iloc[-2], df["PMI_Change"].iloc[-2]),
            arrowprops=dict(arrowstyle="->", color="black", lw=2), zorder=4,
        )

    palette = {
        "1_Expansion": "green", "2_Slowdown": "orange",
        "3_Recession": "red", "4_Recovery": "blue", "Unknown": "gray",
    }
    sns.scatterplot(
        data=df, x="PMI_Smoothed", y="PMI_Change", hue="Cycle_Phase",
        palette=palette, s=80, alpha=0.8, edgecolor="w", zorder=3, ax=ax,
    )

    ax.axvline(x=50, color="black", ls="--", lw=1.2, zorder=2)
    ax.axhline(y=0, color="black", ls="--", lw=1.2, zorder=2)

    # Quadrant labels at fixed offsets from the reference lines
    label_cfg = [
        (52, 0.5, "EXPANSION", "green"),
        (52, -0.5, "SLOWDOWN", "orange"),
        (48, -0.5, "RECESSION", "red"),
        (48, 0.5, "RECOVERY", "blue"),
    ]
    for x, y, label, color in label_cfg:
        ax.text(x, y, label, fontsize=12, fontweight="bold", color=color,
                ha="center", va="center", alpha=0.6)

    ax.set_title(f"Economic Cycle Trajectory (EWMA span={span})", fontsize=14)
    ax.set_xlabel("PMI Level (Smoothed)")
    ax.set_ylabel("PMI Momentum (MoM Change)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(title="Cycle Phase", loc="upper right")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


def plot_scored_heatmap(
    scores: pd.DataFrame, horizon: int, save_path: Path | None = None,
):
    """Heatmap of opportunity scores (sectors x phases)."""
    # Rename columns: EXCESS_SXNP_3M_FWD -> "Industrial G&S"
    rename = {}
    for col in scores.columns:
        ticker = col.replace("EXCESS_", "").replace(f"_{horizon}M_FWD", "")
        rename[col] = SECTOR_NAMES.get(ticker, ticker)
    scores = scores.rename(columns=rename)

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(scores.T, annot=True, cmap="RdYlGn", center=0, fmt=".0f", linewidths=0.5, ax=ax)
    ax.set_title(
        f"Sector Opportunity Score by Phase ({horizon}M Forward)\n"
        "(Avg Fwd Excess Return % x Win Rate %)", fontsize=14,
    )
    ax.set_xlabel("Economic Cycle Phase")
    ax.set_ylabel("Sector")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    HORIZON = 3  # months forward
    SPAN = 3     # EWMA smoothing span for PMI

    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found. Run 'python src/data_pipeline/data_factory.py' first.")
        return

    df = pd.read_parquet(DATA_PATH)
    excess_cols = [c for c in df.columns if c.startswith("EXCESS_")]
    if not excess_cols:
        print("No EXCESS_ columns found in data.")
        return

    # 1. Classify phases
    df = classify_phases(df, span=SPAN)

    # 2. Forward returns
    df, fwd_cols = add_forward_returns(df, excess_cols, horizon=HORIZON)

    # 3. Plots
    plot_phase_diagram(df, span=SPAN, save_path=OUTPUT_DIR / "pmi_phase_trajectory.pdf")

    scores = compute_scores(df, fwd_cols)
    plot_scored_heatmap(scores, horizon=HORIZON, save_path=OUTPUT_DIR / "pmi_sector_heatmap.pdf")

    # 4. Strategy summary
    print(f"\n--- Best sector per phase ({HORIZON}M forward) ---")
    for phase in scores.index:
        row = scores.loc[phase]
        best_fwd = row.idxmax()
        ticker = best_fwd.replace("EXCESS_", "").replace(f"_{HORIZON}M_FWD", "")
        name = SECTOR_NAMES.get(ticker, ticker)
        print(f"  {phase:15} | {name:20} | Score: {row.max():.0f}")


if __name__ == "__main__":
    main()

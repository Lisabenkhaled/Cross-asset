"""
Cross-Asset ETL Pipeline.

Extract sector prices (Bloomberg Excel), PMI indicators (CSV),
and benchmark (SXXP from valeur Excel) into a single Parquet file
aligned on a daily frequency.

Output schema (master_data_daily.parquet):
    Index : DatetimeIndex ("Date"), business-day frequency
    PMI   : float – composite Manufacturing+Services PMI (forward-filled)
    <TICKER> : float – daily simple return for each STOXX 600 sector
    BENCHMARK : float – daily simple return for SXXP Index
    EXCESS_<TICKER> : float – sector return minus benchmark return
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")

SOURCES = {
    "sectors": DATA_DIR / "cross_asset_stoxx600_sectors.xlsx",
    "valeur": DATA_DIR / "cross_asset_valeur.xlsx",
    "pmi_manu": DATA_DIR / "pmmneu_m_d.csv",
    "pmi_serv": DATA_DIR / "pmsreu_m_d.csv",
}

OUTPUT_DAILY = DATA_DIR / "master_data_daily.parquet"
OUTPUT_MONTHLY = DATA_DIR / "master_data_monthly.parquet"

# Sheets to skip when reading sector prices
_SKIP_SHEETS = {"DESCRIPTION", "description", "Desc", "DESC"}

# Bloomberg Excel layout: 6 header rows before data
_BLOOMBERG_SKIPROWS = 6

# SXXP benchmark location inside cross_asset_valeur.xlsx
_BENCH_DATE_COL = 54
_BENCH_PRICE_COL = 55
_BENCH_DATA_START_ROW = 7


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _load_pmi() -> pd.Series:
    """Load Manufacturing + Services PMI CSVs and return a composite series."""
    dfs = []
    for key in ("pmi_manu", "pmi_serv"):
        path = SOURCES[key]
        if not path.exists():
            raise FileNotFoundError(f"PMI file missing: {path}")
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        dfs.append(df["Close"])

    composite = pd.concat(dfs, axis=1).mean(axis=1)
    return composite.rename("PMI")


def _load_benchmark() -> pd.Series:
    """Extract SXXP price series from cross_asset_valeur.xlsx and return daily returns."""
    path = SOURCES["valeur"]
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file missing: {path}")

    raw = pd.read_excel(path, sheet_name="Feuil1", header=None)
    chunk = raw.iloc[_BENCH_DATA_START_ROW:, [_BENCH_DATE_COL, _BENCH_PRICE_COL]].copy()
    chunk.columns = ["Date", "Price"]
    chunk = chunk.dropna(subset=["Date"])
    chunk["Date"] = pd.to_datetime(chunk["Date"])
    chunk["Price"] = pd.to_numeric(chunk["Price"], errors="coerce")

    bench = chunk.set_index("Date")["Price"].sort_index().dropna().rename("BENCHMARK")
    return bench.pct_change().rename("BENCHMARK")


def _load_sector_prices(
    excluded_sheets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read all sector sheets from the Bloomberg Excel workbook into a wide DataFrame of prices."""
    path = SOURCES["sectors"]
    if not path.exists():
        raise FileNotFoundError(f"Sector file missing: {path}")

    skip = set(excluded_sheets) if excluded_sheets else _SKIP_SHEETS
    xl = pd.ExcelFile(path)
    sheets = [s for s in xl.sheet_names if s not in skip]

    sectors: list[pd.Series] = []
    for sheet in sheets:
        df = pd.read_excel(path, sheet_name=sheet, skiprows=_BLOOMBERG_SKIPROWS)
        if "Date" not in df.columns or "PX_LAST" not in df.columns:
            print(f"  [skip] {sheet}: missing Date/PX_LAST columns")
            continue

        df["Date"] = pd.to_datetime(df["Date"])
        series = (
            df.set_index("Date")["PX_LAST"]
            .pipe(pd.to_numeric, errors="coerce")
            .rename(sheet)
            .sort_index()
        )
        sectors.append(series)

    if not sectors:
        raise ValueError("No valid sector data found in Excel workbook.")

    return pd.concat(sectors, axis=1, sort=True).sort_index()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run(*, save: bool = True) -> pd.DataFrame:
    """
    Execute the full ETL pipeline.

    Returns the daily master DataFrame with columns:
        PMI, BENCHMARK, <sector tickers>, EXCESS_<sector tickers>
    """
    print("--- Extract ---")
    pmi = _load_pmi()
    print(
        f"  PMI: {len(pmi)} observations ({pmi.index.min():%Y-%m} to {pmi.index.max():%Y-%m})"
    )

    bench = _load_benchmark()
    print(f"  Benchmark (SXXP): {bench.dropna().shape[0]} daily returns")

    prices = _load_sector_prices()
    print(f"  Sectors: {prices.shape[1]} series, {prices.shape[0]} rows")

    # -- Transform --
    print("--- Transform ---")

    sector_rets = prices.pct_change()

    # Align all series on the same daily index
    master = pd.concat([pmi, bench, sector_rets], axis=1, sort=True)
    master["PMI"] = master["PMI"].ffill()
    master = master.dropna(subset=["PMI", "BENCHMARK"])

    # Excess returns
    sector_cols = sector_rets.columns.tolist()
    excess = master[sector_cols].sub(master["BENCHMARK"], axis=0)
    excess.columns = [f"EXCESS_{c}" for c in sector_cols]
    master = pd.concat([master, excess], axis=1, sort=False)

    master.index.name = "Date"

    print(
        f"  Final shape: {master.shape}  ({master.index.min():%Y-%m-%d} to {master.index.max():%Y-%m-%d})"
    )

    # -- Load --
    if save:
        print("--- Load ---")
        master.to_parquet(OUTPUT_DAILY, engine="pyarrow", compression="snappy")
        print(f"  Saved daily  -> {OUTPUT_DAILY}")

        # Monthly resample for pmi_analysis_parquet.py
        ret_cols = ["BENCHMARK"] + sector_cols + [f"EXCESS_{c}" for c in sector_cols]
        monthly_rets = (
            master[ret_cols].resample("ME").apply(lambda x: (1 + x).prod() - 1)
        )
        monthly_pmi = master["PMI"].resample("ME").last()
        monthly = pd.concat([monthly_pmi, monthly_rets], axis=1, sort=False)
        monthly = monthly.dropna(subset=["PMI"])

        monthly.to_parquet(OUTPUT_MONTHLY, engine="pyarrow", compression="snappy")
        print(f"  Saved monthly -> {OUTPUT_MONTHLY}")

    return master


if __name__ == "__main__":
    df = run()
    print(f"\nDone. {df.shape[0]} rows × {df.shape[1]} columns.")

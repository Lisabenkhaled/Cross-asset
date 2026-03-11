from pathlib import Path

import numpy as np
import pandas as pd


def load_sectors_from_excel(
    xlsx_path: str | Path, excluded_sheets: set = None
) -> pd.DataFrame:
    """Charge les feuilles sectorielles et les concatène."""
    if excluded_sheets is None:
        excluded_sheets = {"DESCRIPTION", "description", "Desc", "DESC"}

    xl = pd.ExcelFile(xlsx_path)
    sector_sheets = [s for s in xl.sheet_names if s not in excluded_sheets]

    series_list = []
    for sheet in sector_sheets:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, skiprows=6)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        s = pd.to_numeric(df["PX_LAST"], errors="coerce").rename(sheet)
        series_list.append(s)

    sectors = pd.concat(series_list, axis=1).sort_index()
    return sectors


def load_weights(xlsx_path: str | Path, excluded_tickers: set = None) -> dict:
    """Extrait le mapping Ticker -> Poids de la feuille DESCRIPTION."""
    desc = pd.read_excel(
        xlsx_path,
        sheet_name="DESCRIPTION",
        header=None,
        usecols="B:E",
        skiprows=1,
        names=["Ticker", "Weight", "Sector", "StartDate"],
    )
    desc["Weight"] = pd.to_numeric(desc["Weight"], errors="coerce")
    desc = desc.dropna(subset=["Ticker", "Weight"])
    if excluded_tickers:
        desc = desc[~desc["Ticker"].isin(excluded_tickers)]
    return desc.set_index("Ticker")["Weight"].to_dict()


def extract_series_under_label(
    df: pd.DataFrame, label_candidates: list[str], min_date: str = "2000-01-01"
) -> pd.Series:
    """Cherche un label dans un DataFrame 'bazar' et extrait la série temporelle en dessous."""
    mask = df.astype(str).apply(
        lambda col: col.str.contains("|".join(label_candidates), case=False, na=False)
    )
    if not mask.any().any():
        raise ValueError(f"Labels introuvables parmi: {label_candidates}")

    r, c = np.argwhere(mask.values)[0]
    dates = pd.to_datetime(df.iloc[int(r) + 1 :, int(c)], errors="coerce")
    values = pd.to_numeric(df.iloc[int(r) + 1 :, int(c) + 1], errors="coerce")

    s = pd.Series(values.values, index=dates).dropna()
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s[s.index >= pd.Timestamp(min_date)]

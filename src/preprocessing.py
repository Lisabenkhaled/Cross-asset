import numpy as np
import pandas as pd


def calculate_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Rééchantillonne en fin de mois et calcule les rendements."""
    return prices.resample("ME").last().dropna(how="all").pct_change()


def calculate_future_returns(returns: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    """Calcule les rendements composés futurs sur un horizon donné."""
    return (1 + returns).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1


def build_macro_blocks(
    returns: pd.DataFrame, weight_map: dict, macro_blocks: dict
) -> pd.DataFrame:
    """Agrège les rendements sectoriels en blocs macro-économiques pondérés."""
    block_ret_m = pd.DataFrame(index=returns.index)
    for block_name, tickers in macro_blocks.items():
        valid_tickers = [t for t in tickers if t in returns.columns]
        if not valid_tickers:
            continue

        w = pd.Series({t: weight_map.get(t, np.nan) for t in valid_tickers}).dropna()
        if w.empty:
            w = pd.Series(1.0, index=valid_tickers)
        w = w / w.sum()

        block_ret_m[block_name] = returns[w.index].mul(w, axis=1).sum(axis=1)
    return block_ret_m


def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=0)
    sigma = df.std(axis=0, ddof=0).replace(0, np.nan)
    return (df - mu) / sigma

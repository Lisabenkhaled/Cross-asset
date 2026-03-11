import numpy as np
import pandas as pd


def generate_pmi_macro_signal(pmi_m: pd.Series, lag: int = 0) -> pd.DataFrame:
    """Génère un signal basé sur le niveau et le momentum du PMI."""
    df = pd.DataFrame(index=pmi_m.index)
    df["PMI_lagged"] = pmi_m.shift(lag)
    df["PMI_Delta_lagged"] = df["PMI_lagged"].diff()

    df["macro_state"] = np.where(
        (df["PMI_lagged"] > 50) & (df["PMI_Delta_lagged"] > 0), "PMI_fort", "PMI_faible"
    )
    df["preferred_block"] = np.where(
        df["macro_state"] == "PMI_fort", "Defensive", "Financials"
    )

    # Décalage t+1 pour éviter le look-ahead bias
    df["preferred_block_trade"] = df["preferred_block"].shift(1)
    df["macro_state_trade"] = df["macro_state"].shift(1)
    return df


def generate_momentum_selection(
    returns: pd.DataFrame, window: int = 6, top_n: int = 2
) -> pd.DataFrame:
    """Calcule le momentum sur fenêtre glissante pour la sélection intra-bloc."""
    return (1 + returns).rolling(window).apply(np.prod, raw=True) - 1

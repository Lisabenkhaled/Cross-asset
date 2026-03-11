# %%
import os
from pathlib import Path

# Si le dossier courant est 'notebooks', on remonte d'un cran à la racine
if Path.cwd().name == "notebooks":
    os.chdir("..")

# %% - importations

import numpy as np
import pandas as pd

from src.data_loader import *

# %% local modules
from src.pmi_clustering import *
from src.preprocessing import *
from src.signals import *
from src.stats_metrics import *

# %%
# CONFIGURATION
DATA_PATH = Path("data/cross_asset_sxxp600.xlsx")
BENCH_PATH = Path("data/stoxx600_sectors_history.csv")
EXCLUDED_TICKERS = {"S600PDP"}
MACRO_BLOCKS = {
    "Cyclical": ["SXNP", "SXOP", "SXPP", "SX4P", "SXAP", "SXTP", "SXRP", "S600ENP"],
    "Defensive": ["SXDP", "SX6P", "S600FOP", "S600CPP", "SXKP", "SXMP"],
    "Financials": ["SX7P", "SXIP", "SXFP"],
    "Tech": ["SX8P"],
    "Real_Estate": ["SX86P"],
}
SECTORS = {
    "SXNP": "Industrial Goods & Services",
    "SX7P": "Banks",
    "SXDP": "Health Care",
    "SXIP": "Insurance",
    "S600ENP": "Energy",
    "SX8P": "Technology",
    "S600FOP": "Food, Beverage & Tobacco",
    "S600CPP": "Consumer Products & Services",
    "SX6P": "Utilities",
    "SXFP": "Financial Services",
    "SXOP": "Construction & Materials",
    "SXKP": "Telecommunications",
    "SXPP": "Basic Resources",
    "SX4P": "Chemicals",
    "SXAP": "Automobiles & Parts",
    "SX86P": "Real Estate",
    "SXTP": "Travel & Leisure",
    "SXRP": "Retail",
    "SXMP": "Media",
}

# %%
# PHASE 1 - Nettoyage et préparation mensuelle
sectors = load_sectors_from_excel(DATA_PATH)
sectors = sectors.drop(
    columns=[c for c in EXCLUDED_TICKERS if c in sectors.columns], errors="ignore"
)

ret_m = calculate_monthly_returns(sectors)

# Gestion du Benchmark
bench = (
    pd.read_csv(BENCH_PATH, parse_dates=["Date"]).sort_values("Date").set_index("Date")
)
bench_ret_m = (
    bench[["STOXX_600_Global"]]
    .rename(columns={"STOXX_600_Global": "BENCH"})
    .resample("ME")
    .last()
    .pct_change()
)

ret_all_m = ret_m.join(
    bench_ret_m, how="inner"
)  # dataframe des rendements mensuels pour les secteurs et le benchmark
sectors_cols = [c for c in ret_all_m.columns if c != "BENCH"]

# soustraction par le bench poru avoir les excess_returns
excess_returns = ret_all_m[sectors_cols].sub(ret_all_m["BENCH"], axis=0)
excess_returns.dropna(how="all", inplace=True)
print(f"Rendements mensuels: {ret_all_m.shape}")

# %%
# PHASE 3 - Analyse PMI
pmi_manu = (
    pd.read_csv("data/pmmneu_m_d.csv", parse_dates=["Date"])
    .sort_values("Date")
    .set_index("Date")[["Close"]]
    .rename(columns={"Close": "PMI_M"})
)
pmi_serv = (
    pd.read_csv("data/pmsreu_m_d.csv", parse_dates=["Date"])
    .sort_values("Date")
    .set_index("Date")[["Close"]]
    .rename(columns={"Close": "PMI_S"})
)

# %%
# Moyenne composite PMI
pmi = pmi_manu.join(pmi_serv, how="inner")
pmi["PMI"] = pmi.mean(axis=1)
pmi_m = pmi["PMI"].resample("ME").last()

# %% constitution des quadrants
df_clock = prepare_pmi_coordinates(pmi, pmi_col="PMI")
plot_investment_clock(df_clock, "PMI")
# %%
df_probability_outperform = calcul_probabilites_surperformance(
    excess_returns,
    df_clock,
)

df_probability_outperform.head()
# %%
df_probability_outperform = df_probability_outperform.rename(columns=SECTORS)

df_probability_outperform
# %% Classification

regrouper_meilleurs_secteurs(df_probability_outperform, SECTORS)  # pics de probabilité
# %% - comparaison avec les hypothèses
macro_blocks_clairs = {
    categorie: [SECTORS[ticker] for ticker in liste_tickers]
    for categorie, liste_tickers in MACRO_BLOCKS.items()
}

for categorie, secteurs in macro_blocks_clairs.items():
    print(f"\n[{categorie.upper()}]")
    for secteur in secteurs:
        print(f"  + {secteur}")

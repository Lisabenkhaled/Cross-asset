# %%
from datetime import date
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
data_stoxx600 = pd.read_csv(
    "stoxx600_sectors_history.csv", index_col="Date", parse_dates=True
)
data_stoxx600_etf = pd.read_csv(
    "stoxx600_sectors_etf_history.csv", index_col="Date", parse_dates=True
)

data_stoxx600_etf = data_stoxx600_etf.rename(
    columns={"STOXX_600_Global": "STOXX_600_Global_etf"}
)
data_stoxx600_etf.head()

# %% on ne garde que le stoxx600 dans le premier
data_stoxx600 = data_stoxx600[["STOXX_600_Global"]]
data_stoxx600.head()

# %% comparaison des returns pour s'assurer que c'est comparable
df = pd.concat([data_stoxx600, data_stoxx600_etf], axis=1, join="outer")
df.dropna(axis=0, inplace=True)
df.head()

# %% log returns
log_returns = np.log(df / df.shift(1)).dropna()

log_returns[["STOXX_600_Global", "STOXX_600_Global_etf"]].head()


# %% - ploting des log returns entre etf stoxx600 et le stoxx600
plt.plot(figsize=(10, 6))
log_returns["STOXX_600_Global"].plot()
log_returns["STOXX_600_Global_etf"].plot()

# %% -
performance = 100 * np.exp(log_returns.cumsum())

performance[["STOXX_600_Global", "STOXX_600_Global_etf"]].plot(figsize=(10, 6))

plt.title("Comparaison de performance (Base 100 au départ)")
plt.ylabel("Niveau de prix (Base 100)")
plt.xlabel("Date")
plt.grid(True)
plt.show()

# %% -
perfig, ax = plt.subplots(figsize=(8, 8))

performance.plot(ax=ax)

ax.axvspan("2020-02-20", "2020-05-01", color="gray", alpha=0.2, label="Choc COVID")

ax.axvspan("2021-01-01", "2022-01-01", color="green", alpha=0.1, label="Reprise")

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=5,
    fontsize="small",
    frameon=True,
)

plt.title("Comparaison des secteurs STOXX 600", fontsize=14)
plt.ylabel("Niveau de prix (Base 100)")
plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# %% - module pour faire des backtests
# on doit repasser en rendement arithmetique
returns = df.pct_change().dropna()
returns[["STOXX_600_Global", "STOXX_600_Global_etf"]].head()
# %%


def backtest_portfolio(
    returns_df: pd.DataFrame, weights: Dict[str, float], start_date: str
):
    """
    Returns pd.series() of the portfolio backtested.
    """

    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        print("Somme des poids doit etre 1.")
        return 1

    subset = returns_df.loc[start_date:, list(weights.keys())].copy()

    if subset.empty:
        return "Erreur : Aucune donnée pour cette plage de dates."

    weights_series = pd.Series(weights)

    daily_portfolio_returns = subset.dot(weights_series)

    portfolio_value = 100 * (1 + daily_portfolio_returns).cumprod()

    plt.figure(figsize=(8, 6))

    portfolio_value.plot(label="Portefeuille", linewidth=2, color="black")

    (100 * (1 + subset).cumprod()).plot(
        ax=plt.gca(), alpha=0.3, linewidth=1, linestyle="--"
    )

    plt.title(f"Backtest Portefeuille (Début : {start_date})", fontsize=14)
    plt.ylabel("Prix")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return portfolio_value


# %% - exemple

start_date = "2022-01-01"
dict_weights = {"Banks": 0.4, "Construction_Materials": 0.3, "Health_Care": 0.3}
result = backtest_portfolio(log_returns, dict_weights, start_date)


# %% - metrics backtest
def get_metrics(
    portfolio_value: pd.Series, risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """
    Calcule les métriques de performance à partir d'une série de prix (Base 100).

    Args:
        portfolio_value: pd.Series avec DateTimeIndex.
        risk_free_rate: Taux sans risque annuel (ex: 0.03 pour 3%). Par défaut 0.
    """

    daily_returns = portfolio_value.pct_change().dropna()

    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1

    start_date = portfolio_value.index[0]
    end_date = portfolio_value.index[-1]
    years = (end_date - start_date).days / 365.25

    cagr = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (1 / years) - 1

    volatility = daily_returns.std() * np.sqrt(252)

    sharpe_ratio = (cagr - risk_free_rate) / volatility

    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value / running_max) - 1
    max_drawdown = drawdown.min()

    metrics = pd.DataFrame(
        {
            "Métrique": [
                "Total Return",
                "CAGR (Annuel)",
                "Volatilité (Annuelle)",
                "Ratio de Sharpe",
                "Max Drawdown",
            ],
            "Valeur": [
                f"{total_return:.2%}",
                f"{cagr:.2%}",
                f"{volatility:.2%}",
                f"{sharpe_ratio:.2f}",
                f"{max_drawdown:.2%}",
            ],
        }
    )

    return metrics


# %% -
metrics_df = get_metrics(result)
print(metrics_df)


# %% - Underwater plot
def plot_performance_and_drawdown(portfolio_value):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [2, 1]}
    )

    portfolio_value.plot(ax=ax1, color="#1f77b4", linewidth=2)
    ax1.set_title("Performance du Portefeuille", fontsize=12)
    ax1.set_ylabel("Prix (Base 100)")
    ax1.grid(True, linestyle="--")

    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value / running_max) - 1

    drawdown.plot(ax=ax2, color="red", alpha=0.6, linewidth=1)
    ax2.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)

    ax2.set_title("Drawdown (Perte depuis le sommet)", fontsize=12)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, linestyle="--")

    plt.tight_layout()
    plt.show()


# %% -
plot_performance_and_drawdown(result)

# %% - Regroupement des secteurs par cycle macroeconomique (investment clock method)
# Importation des PMI (services + manufacturiers)

pmi_manufacturing = pd.read_csv(
    "pmmneu_m_d.csv", index_col="Date", parse_dates=True, usecols=["Date", "Close"]
)
pmi_services = pd.read_csv(
    "pmsreu_m_d.csv", index_col="Date", parse_dates=True, usecols=["Date", "Close"]
)

pmi = pd.concat([pmi_manufacturing, pmi_services], join="inner", axis=1)
pmi.dropna(inplace=True)
pmi["mean"] = pmi.mean(axis=1)
pmi.head()

# %% - Identification des periodes


def get_cycle(value, change):
    if pd.isna(value) or pd.isna(change):
        return "Inconnu"

    if value > 50:
        return "Expansion" if change >= 0 else "Ralentissement"
    else:
        return "Récupération" if change >= 0 else "Récession"


df_pmi = pd.DataFrame({"PMI": pmi["mean"]})

# Calcul de la variation sur la donnée lissée
df_pmi["Change"] = df_pmi["PMI"].diff()

df_pmi["Cycle"] = df_pmi.apply(lambda row: get_cycle(row["PMI"], row["Change"]), axis=1)

cycle_colors = {
    "Expansion": "#d4edda",  # Vert clair
    "Ralentissement": "#fff3cd",  # Jaune/Orange clair
    "Récession": "#f8d7da",  # Rouge clair
    "Récupération": "#cce5ff",  # Bleu clair
    "Inconnu": "white",
}

fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(df_pmi.index, df_pmi["PMI"], color="black", linewidth=2, label="PMI")

# Ligne de seuil 50
ax.axhline(50, color="red", linestyle=":", linewidth=1)

# On détecte les changements de cycle pour ne pas dessiner 1000 petits rectangles,
# mais des gros blocs continus
df_pmi["group"] = (df_pmi["Cycle"] != df_pmi["Cycle"].shift()).cumsum()

for group_id, data in df_pmi.groupby("group"):
    cycle_name = data["Cycle"].iloc[0]
    start_date = data.index[0]
    end_date = data.index[-1] + pd.Timedelta(days=30)

    if cycle_name in cycle_colors:
        ax.axvspan(
            start_date,
            end_date,
            facecolor=cycle_colors[cycle_name],
            alpha=0.5,
            label=cycle_name,
        )

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="best")

ax.set_title("Cycles Economiques basés sur le PMI (Niveau & Tendance)", fontsize=14)
ax.set_ylabel("Indice PMI")
ax.set_xlim(df_pmi.index[0], df_pmi.index[-1])
ax.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

# %% secteurs cycliques vs contrecycliques
excess_returns = returns.sub(returns["STOXX_600_Global_etf"], axis=0)
excess_returns.head()

# %% secteurs
# Upsampling
excess_returns["cycle"] = df_pmi["Cycle"].resample("D").ffill()

excess_returns.dropna(inplace=True)
excess_returns.info()
# %%
list_sectors = excess_returns.columns
print(list_sectors)
list_sectors = list_sectors[2:-1]
print(list_sectors)
len(list_sectors.tolist())

# %%
mean_excess = excess_returns.groupby("cycle")[list_sectors].mean()

# %% visualisation avec une heatmap
# Transposition pour avoir les cycles en colonne et les secteurs en ligne
mean_excess = mean_excess.T

plt.figure(figsize=(8, 5))
sns.heatmap(mean_excess, annot=True, cmap="RdYlGn", center=0, fmt=".2%")
plt.title(
    "Excess return par secteur par rapport au benchmark (STOXX 600) "
    + "par Cycle Economique"
)
plt.savefig("excess_returns_by_cycle_forwardfill.pdf")
plt.show()

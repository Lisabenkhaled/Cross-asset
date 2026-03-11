import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def prepare_pmi_coordinates(df, pmi_col="PMI", threshold=50.0, momentum_periods=6):
    """
    Prend un DataFrame avec le PMI et renvoie les coordonnées et les quadrants.
    """
    df_clock = df.copy()
    df_clock["PMI_Smoothed"] = (
        df_clock[pmi_col].ewm(span=momentum_periods, adjust=False).mean()
    )

    df_clock["PMI_Variation"] = df_clock["PMI_Smoothed"].diff(periods=1)

    conditions = [
        (df_clock[pmi_col] < threshold)
        & (df_clock["PMI_Variation"] > 0),  # Q1: Reprise
        (df_clock[pmi_col] > threshold)
        & (df_clock["PMI_Variation"] > 0),  # Q2: Expansion
        (df_clock[pmi_col] > threshold)
        & (df_clock["PMI_Variation"] < 0),  # Q3: Ralentissement
        (df_clock[pmi_col] < threshold)
        & (df_clock["PMI_Variation"] < 0),  # Q4: Contraction
    ]

    choix = ["1_Reprise", "2_Expansion", "3_Ralentissement", "4_Contraction"]
    df_clock["Quadrant"] = np.select(conditions, choix, default="Indéfini")

    return df_clock


def plot_investment_clock(df, pmi_col="PMI", var_col="PMI_Variation", threshold=50.0):
    plt.figure(figsize=(10, 8))

    plt.scatter(
        df[pmi_col], df[var_col], color="royalblue", alpha=0.6, edgecolors="black"
    )
    plt.axvline(x=threshold, color="red", linestyle="--", linewidth=1.5)
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    plt.text(
        threshold + 0.5,
        df[var_col].max() * 0.8,
        "Q2: EXPANSION",
        fontsize=12,
        fontweight="bold",
        color="green",
    )
    plt.text(
        threshold - 0.5,
        df[var_col].max() * 0.8,
        "Q1: REPRISE",
        fontsize=12,
        fontweight="bold",
        color="orange",
        horizontalalignment="right",
    )
    plt.text(
        threshold - 0.5,
        df[var_col].min() * 0.8,
        "Q4: CONTRACTION",
        fontsize=12,
        fontweight="bold",
        color="red",
        horizontalalignment="right",
    )
    plt.text(
        threshold + 0.5,
        df[var_col].min() * 0.8,
        "Q3: RALENTISSEMENT",
        fontsize=12,
        fontweight="bold",
        color="purple",
    )
    plt.title("Horloge de l'Investissement (PMI Phase Plot)", fontsize=14)
    plt.xlabel(f"Niveau du {pmi_col} (Seuil à {threshold})", fontsize=12)
    plt.ylabel("Variation du PMI (Momentum)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.plot(df[pmi_col], df[var_col], color="gray", alpha=0.3, linewidth=1)

    plt.show()


def calcul_probabilites_surperformance(
    df_excess_returns, df_coordonnees, col_quadrant="Quadrant"
):
    """
    Calcule la probabilité historique de surperformance par secteur et par quadrant.

    Arguments:
    - df_excess_returns : DataFrame des rendements excédentaires (Index = Dates, Colonnes = Secteurs)
    - df_coordonnees : DataFrame contenant les quadrants (Index = Dates, doit contenir col_quadrant)

    Retourne:
    - Un DataFrame avec les quadrants en index et les probabilités (de 0 à 1) par secteur.
    """

    df_merged = pd.merge(
        df_excess_returns,
        df_coordonnees[[col_quadrant]],
        left_index=True,
        right_index=True,
        how="inner",
    )
    secteurs = df_excess_returns.columns
    df_signaux = (df_merged[secteurs] > 0).astype(int)
    df_signaux[col_quadrant] = df_merged[col_quadrant]
    df_probabilites = df_signaux.groupby(col_quadrant).mean()
    df_probabilites_pct = df_probabilites * 100

    return df_probabilites_pct


def regrouper_meilleurs_secteurs(df, dictionnaire_secteurs):
    df_noms = df.rename(columns=dictionnaire_secteurs)

    if "Indéfini" in df_noms.index:
        df_noms = df_noms.drop("Indéfini")

    meilleur_quadrant = df_noms.idxmax()  # Trouve l'index de la valeur max
    proba_max = df_noms.max()  # Récupère la valeur max

    print("Classification des secteurs :")

    # Pour chaque quadrant, on cherche quels secteurs y ont leur pic de performance
    for quadrant in df_noms.index:
        print(f"\nPhase : {quadrant[2:].upper()}")

        secteurs_du_quadrant = meilleur_quadrant[meilleur_quadrant == quadrant].index

        if len(secteurs_du_quadrant) == 0:
            print("  Aucun secteur n'a son pic absolu dans cette phase.")
            continue

        # liste de tuples (Secteur, Proba) pour pouvoir trier du meilleur au moins bon
        liste_triee = [
            (secteur, proba_max[secteur]) for secteur in secteurs_du_quadrant
        ]
        liste_triee.sort(key=lambda x: x[1], reverse=True)  # Tri décroissant

        for secteur, proba in liste_triee:
            if proba > 50:
                print(f"  + {secteur:<30} (Proba : {proba:.1f}%)")
            else:
                print(
                    f"  - {secteur:<30} (Proba : {proba:.1f}% - Max atteint, mais < 50%)"
                )

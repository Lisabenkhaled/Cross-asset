import numpy as np
import pandas as pd


def performance_stats(
    ret: pd.Series, benchmark_ret: pd.Series | None = None, periods_per_year: int = 12
) -> pd.Series:
    """Calcule les métriques de performance classiques (CAGR, Sharpe, MaxDD, IR)."""
    r = ret.dropna().copy()
    if len(r) == 0:
        return pd.Series(
            {"CAGR": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "IR": np.nan}
        )

    nav = (1 + r).cumprod()
    years = len(r) / periods_per_year
    cagr = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ((r.mean() * periods_per_year) / vol) if vol else np.nan

    dd = nav / nav.cummax() - 1
    max_dd = dd.min()

    ir = np.nan
    if benchmark_ret is not None:
        br = benchmark_ret.reindex(r.index)
        active = (r - br).dropna()
        if len(active) > 1:
            te = active.std(ddof=1) * np.sqrt(periods_per_year)
            ir = ((active.mean() * periods_per_year) / te) if te else np.nan

    return pd.Series({"CAGR": cagr, "Sharpe": sharpe, "MaxDD": max_dd, "IR": ir})


def kmeans_numpy(
    X: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300,
):
    """Implémentation KMeans pure numpy."""
    rng = np.random.default_rng(random_state)
    best_inertia, best_labels, best_centers = np.inf, None, None

    for _ in range(n_init):
        init_idx = rng.choice(len(X), size=n_clusters, replace=False)
        centers = X[init_idx].copy()

        for _ in range(max_iter):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d2.argmin(axis=1)

            new_centers = centers.copy()
            for k in range(n_clusters):
                members = X[labels == k]
                if len(members) > 0:
                    new_centers[k] = members.mean(axis=0)
            if np.allclose(new_centers, centers, atol=1e-8):
                break
            centers = new_centers

        inertia = ((X - centers[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia, best_labels, best_centers = (
                inertia,
                labels.copy(),
                centers.copy(),
            )

    return best_labels, best_centers, best_inertia

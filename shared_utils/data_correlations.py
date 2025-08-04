import pandas as pd
import numpy as np
from scipy import stats

def cramers_v(cat_series_a: pd.Series, cat_series_b: pd.Series) -> float:
    """Metric that calculates the corrected Cramer's V statistic for categorical-categorical
    correlations, used in heatmap generation.

    Args:
        cat_series_a (pd.Series):
            First categorical series to analyze.
        cat_series_b (pd.Series):
            Second categorical series to analyze.

    Returns:
        float: Value of the statistic.
    """

    valid = cat_series_a.notna() & cat_series_b.notna()
    if valid.sum() < 2:
        return float("nan")

    confusion_matrix = pd.crosstab(cat_series_a[valid], cat_series_b[valid])

    if confusion_matrix.size == 0 or confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        return float("nan")

    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


def pearson(num_series_a: pd.Series, num_series_b: pd.Series) -> float:
    """Metric that calculates Pearson's correlation coefficent for numerical-numerical
    pairs of series, used in heatmap generation.

    Args:
        sr_a (pd.Series): First numerical series to analyze.
        sr_b (pd.Series): Second numerical series to analyze.

    Returns:
        float: Value of the coefficient.
    """
    valid = num_series_a.notna() & num_series_b.notna()
    if valid.sum() < 2:
        return float("nan")

    std_a = num_series_a[valid].std()
    std_b = num_series_b[valid].std()

    if std_a == 0 or std_b == 0:
        return float("nan")

    return abs(num_series_a[valid].corr(num_series_b[valid]))


def kruskal_wallis_boolean(cat_series: pd.Series, num_series: pd.Series, p_cutoff: float = 0.1) -> bool:
    """Metric that uses the Kruskal-Wallis H Test to obtain a p-value that is used to determine
    whether the possibility that the columns obtained by grouping the continuous series
    by the categorical series come from the same distribution. Used for proxy detection.

    Args:
        cat_series (pd.Series):
            The categorical series to analyze, used for grouping the numerical one.
        num_series (pd.Series):
            The numerical series to analyze.
        p_cutoff (float):
            The maximum admitted p-value for the distributions to be considered independent.

    Returns:
        bool: Bool value representing whether or not the two series are correlated.
    """

    valid = cat_series.notna() & num_series.notna()
    cat_series = cat_series[valid]
    num_series = num_series[valid]

    if len(cat_series.unique()) < 2 or len(num_series.unique()) < 2:
        return False

    try:
        groups = [num_series[cat_series == cat] for cat in cat_series.unique()]
        if any(len(g) < 2 for g in groups):
            return False
        stat, p = stats.kruskal(*groups)
        return p < p_cutoff # True
    except Exception:
        return False
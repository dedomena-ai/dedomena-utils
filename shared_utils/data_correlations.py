import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any
import json

def detect_column_types(df):
    """
    Detect the data type of each column in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: A dictionary mapping column names to their data types.
    """
    column_types = {}

    for col in df.columns:
        dtype = df[col].dtype

        # String (nuevo tipo de pandas)
        if isinstance(dtype, pd.StringDtype):
            column_types[col] = "string"
        # Integer
        elif pd.api.types.is_integer_dtype(dtype):
            column_types[col] = "integer"
        # Float
        elif pd.api.types.is_float_dtype(dtype):
            column_types[col] = "float"
        # Boolean
        elif pd.api.types.is_bool_dtype(dtype):
            column_types[col] = "boolean"
        # Datetime
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            column_types[col] = "datetime"
        # Category
        elif isinstance(dtype, pd.CategoricalDtype):
            column_types[col] = "category"
        # Object fallback
        elif dtype == "object":
            column_types[col] = "object"
        else:
            column_types[col] = str(dtype)# fallback to raw dtype name

    return column_types


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


def generate_distribution_summary(serie, bins_amount=15):
    """
    Generates a summarized distribution for a pandas series:
    - If numeric: returns a histogram by bins.
    - If datetime: returns a histogram by date intervals.
    - If categorical: returns counts of the 20 most frequent categories.
    """
    
    serie = serie.dropna()
    if serie.empty:
        return json.dumps({
            "histogram": [],
            "labels": []
        })

    if pd.api.types.is_numeric_dtype(serie):
        min_val, max_val = serie.min(), serie.max()
        if min_val == max_val:
            return json.dumps({
                "histogram": [len(serie)],
                "labels": [f"{min_val}"]
            })
        if pd.api.types.is_integer_dtype(serie) and max_val - min_val <= bins_amount:
            bins = range(min_val, max_val + 2)
            labels = [str(v) for v in range(min_val, max_val + 1)]
            hist = serie.value_counts().reindex(range(min_val, max_val + 1), fill_value=0).tolist()
        else:
            bins = np.linspace(min_val, max_val, bins_amount + 1)
            cat = pd.cut(serie, bins=bins, include_lowest=True)
            hist = cat.value_counts(sort=False).tolist()
            labels = [f"{round(i.left, 2)} - {round(i.right, 2)}" for i in cat.cat.categories]

    elif pd.api.types.is_datetime64_any_dtype(serie):
        timestamps = serie.dropna().astype('int64') // 10**9  # convert to seconds
        if timestamps.nunique() == 1:
            val = pd.to_datetime(timestamps.iloc[0], unit='s').date()
            return {"histogram": [len(timestamps)], "labels": [f"{val}"]}
        hist, bin_edges = np.histogram(timestamps, bins=bins_amount)
        labels = [
            f"{pd.to_datetime(bin_edges[i], unit='s').date()} - {pd.to_datetime(bin_edges[i+1], unit='s').date()}"
            for i in range(len(bin_edges) - 1)
        ]


    else:  # Categorical or any other type
        str_vals = serie.astype(str)
        top_values = str_vals.value_counts().head(20)
        hist = top_values.tolist()
        labels = top_values.index.tolist()
    
    # Filter empty bins
    hist = np.array(hist)
    nonzero_mask = hist > 0
    hist = hist[nonzero_mask].tolist()
    labels = [label for i, label in enumerate(labels) if nonzero_mask[i]]
    return json.dumps({
        "histogram": hist,
        "labels": labels
        }, default=str)
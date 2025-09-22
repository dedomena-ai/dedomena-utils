import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, median_abs_deviation, gaussian_kde
from collections import Counter
from scipy.special import expit
import logging
logger = logging.getLogger(__name__)   

pd.set_option('future.no_silent_downcasting', True)

class AdaptiveOutlier:
    def __init__(self, data, outlier_process="adaptive", outlier_threshold=0.5, extended_explain=False):
        # Normalización del input
        if isinstance(data, pd.Series):
            self.df = data.to_frame()
        elif isinstance(data, list):
            self.df = pd.DataFrame({"feature": data})
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            raise TypeError("Se espera un DataFrame, Series o lista.")
        self.clean_df = self.df.copy()
        self.outlier_process = outlier_process
        self.outlier_threshold = outlier_threshold
        self.extended_explain = extended_explain
        self.commons_idx = []
        self.results = {}

    def get_common_outlier_indices(self, threshold=3):
        """
        Returns the set of indices that appear as outliers in at least `threshold` columns.
        """
        all_indices = []
        for col_data in self.results.values():
            if col_data["summary"].get("n_outliers") > 0:
                all_indices.extend(col_data["outliers_detected"]["idx"])

        index_counts = Counter(all_indices)
        common_indices = [idx for idx, count in index_counts.items() if count >= threshold]
        return common_indices
    
    def run_pipeline(self):
        for col in self.df.columns:
            series = self.df[col]
            series = series[~series.isna()]
            ao = AdaptiveOutlierSingle(series, self.outlier_process, self.outlier_threshold)
            self.results[col] = ao.detect_outliers()

        self.commons_idx = self.get_common_outlier_indices()
        if self.commons_idx:
            self.clean_df = self.clean_df.drop(index=self.commons_idx)
        return self.results


class AdaptiveOutlierSingle:
    def __init__(self, series, outlier_process = "adaptive", outlier_threshold=0.5):
        self.series = pd.Series(series).dropna()
        self.outlier_process = outlier_process
        self.outlier_threshold = outlier_threshold
        self.info = "Outlier analysis has not been performed."
        self.col_type = self._infer_col_type()   
        self._valid = self._is_valid()
        self.skewness, self.kurtosis = self._compute_distribution_features()
        self.distribution_classification = self._classify_distribution()
        self.outlier_methods = self._select_outlier_methods()
        self.thresholds = self._get_thresholds()

    def _infer_col_type(self):
        s = self.series
        if s.isna().all():
            return "empty"
        
        if pd.api.types.is_bool_dtype(s):
            return "boolean"

        if pd.api.types.is_numeric_dtype(s):
            unique_ratio = s.nunique() / len(s)
            if s.nunique() == 1:
                return "constant"
            
            if pd.api.types.is_integer_dtype(s):
                if unique_ratio < 0.05 or s.nunique() <= 20:
                    return "category"   
                else:
                    return "integer"
            elif pd.api.types.is_float_dtype(s):
                if unique_ratio < 0.01 or s.nunique() <= 30:
                    return "category"
                else:
                    return "float"
            else:
                return "numeric"

        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"

        try:
            converted = pd.to_datetime(s, errors="coerce", utc=True)
            if converted.notna().sum() / len(s) > 0.9:  # at least 90% of values should be valid dates
                self.series = converted.dropna()
                return "datetime"
        except Exception:
            pass

        if pd.api.types.is_categorical_dtype(s):
            return "category"

        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            unique_ratio = s.nunique() / len(s)
            avg_len = s.astype(str).map(len).mean()

            if s.nunique() <= 20 and avg_len < 20:
                return "category"
            else:
                return "text"

        return "other"

    def _is_valid(self):
        """Evaluates if it makes sense to apply shape analysis."""
        if self.series.empty:
            self.info = "The series is empty."
            self.col_type = None
            return False
        if pd.api.types.is_bool_dtype(self.series):
            self.info = "The series is boolean, outlier analysis is not applicable."
            return True
        if pd.api.types.is_integer_dtype(self.series) and self.series.nunique() <= 3:
            self.info = "The series is numeric with few unique values, outlier analysis is not applicable."
            return True
        self.info = "Series is valid for outlier analysis."
        return True

    def _compute_distribution_features(self):
        if not self._valid or self.col_type not in ["integer", "float", "numeric"]:
            return None, None
        return np.round(skew(self.series), 2), np.round(kurtosis(self.series, fisher=True), 2)

    def _classify_distribution(self):
        skewness, kurtosis = self.skewness, self.kurtosis
        if skewness is None or kurtosis is None or not self._valid:
            return {
            'symmetry': 'unknown',
            'tails': 'unknown'
            }
        
        if skewness < -0.5:
            symmetry = 'left-skewed'
        elif skewness > 0.5:
            symmetry = 'right-skewed'
        else:
            symmetry = 'symmetric'
        
        if kurtosis > 1:
            tails = 'leptokurtic'       
        elif kurtosis < -1:
            tails = 'platykurtic'        
        else:
            tails = 'mesokurtic'         

        return {
            'symmetry': symmetry,
            'tails': tails
        }

    def _select_outlier_methods(self) -> dict:
        methods = []
        weights = {}

        symmetry = self.distribution_classification['symmetry']
        tails = self.distribution_classification['tails']
        kurt = self.kurtosis if self.kurtosis is not None else 0
        skew = self.skewness if self.skewness is not None else 0

        if not self._valid:
            return {'methods': [], 'weights': {}}

        if symmetry == 'symmetric':
            methods += ['IQR', 'MAD', 'ECOD']
            weights.update({'IQR': 0.4, 'MAD': 0.3, 'ECOD': 0.3})
        else:
            methods += ['IQR_asymmetric', 'MAD_asymmetric', 'ECOD_peak']
            weights.update({'IQR_asymmetric': 0.3, 'MAD_asymmetric': 0.3, 'ECOD_peak': 0.4})

        # --- Adjustment for kurtosis ---
        neutral_kurt = 3.0
        delta_kurt = np.clip(abs(kurt - neutral_kurt) / 5, 0, 1) # Selected by hand
        correction_kurt = 0.05 + delta_kurt * (0.15 - 0.05)

        if tails == 'leptokurtic':
            for m in weights:
                if 'ECOD' in m:
                    weights[m] += correction_kurt
        elif tails == 'platykurtic':
            for m in weights:
                if 'ECOD' in m:
                    weights[m] -= correction_kurt
                if 'IQR' in m or 'IQR_asymmetric' in m:
                    weights[m] += correction_kurt

        # --- Adjustment for skewness ---
        neutral_skew = 0.0
        delta_skew = np.clip(abs(skew - neutral_skew) / 3, 0, 1) # Selected by hand
        correction_skew = 0.05 + delta_skew * (0.15 - 0.05)

        if symmetry in ['right-skewed', 'left-skewed']:
            for m in weights:
                if 'IQR' in m or 'IQR_asymmetric' in m:
                    weights[m] += correction_skew
                if 'MAD' in m or 'MAD_asymmetric' in m:
                    weights[m] += correction_skew / 2
                if 'ECOD' in m or 'ECOD_peak' in m:
                    weights[m] -= correction_skew / 2

        # --- Normalize weights ---
        total_weight = sum(weights.values())
        for m in weights:
            weights[m] = round(weights[m] / total_weight, 3)

        return {'methods': methods, 'weights': weights}

    def _get_thresholds(self):

        skewness = self.skewness
        if not self._valid or skewness is None:
            return {
                'q_low': 25,
                'q_high': 75,
                'mad_low': 3.5,
                'mad_high': 3.5,
                'iqr_low_multiplier': 1.5,
                'iqr_high_multiplier': 1.5
            }
        if skewness < -0.5:
            # Bias to the left → longer left tail → more sensitive to left outliers
            return {
                'q_low': 20,
                'q_high': 75,
                'mad_low': 3.0,
                'mad_high': 4.0,
                'iqr_low_multiplier': 1,
                'iqr_high_multiplier': 2.0
            }
        elif skewness > 0.5:
            # Bias to the right → longer right tail → more sensitive to right outliers
            return {
                'q_low': 25,
                'q_high': 80,
                'mad_low': 4.0,
                'mad_high': 3.0,
                'iqr_low_multiplier': 2.0,
                'iqr_high_multiplier': 1.0
            }
        else:
            # Symmetric distribution → balanced sensitivity
            return {
                'q_low': 25,
                'q_high': 75,
                'mad_low': 3.5,
                'mad_high': 3.5,
                'iqr_low_multiplier': 1.5,
                'iqr_high_multiplier': 1.5
            }
            
    def iqr_score_outliers(self):

        q_low = np.percentile(self.series, self.thresholds['q_low'])
        q_high = np.percentile(self.series, self.thresholds['q_high'])
        iqr = q_high - q_low

        lower_bound = q_low - self.thresholds['iqr_low_multiplier'] * iqr
        upper_bound = q_high + self.thresholds['iqr_high_multiplier'] * iqr

        dist_lower = np.maximum(0, lower_bound - self.series)
        dist_upper = np.maximum(0, self.series - upper_bound)
        dist = dist_lower + dist_upper

        scores = dist / iqr

        max_margin = max(self.thresholds['iqr_low_multiplier'], self.thresholds['iqr_high_multiplier'])
        scores = np.clip(scores / max_margin, 0, 1)
        outlier_mask = scores > 0

        outliers = self.series[outlier_mask]
        outlier_scores = scores[outlier_mask]

        df_outliers = pd.DataFrame({
            'score': outlier_scores
        })

        return df_outliers

    def mad_score_outliers(self):
        median = np.median(self.series)
        mad = np.median(np.abs(self.series - median))

        if mad == 0:
            return pd.DataFrame(columns=['score'])

        lower_bound = median - self.thresholds['mad_low'] * mad
        upper_bound = median + self.thresholds['mad_high'] * mad

        scores = np.zeros_like(self.series, dtype=float)

        mask_low = self.series < lower_bound
        mask_high = self.series > upper_bound

        scores[mask_low] = np.clip(np.abs(self.series[mask_low] - lower_bound) / (self.thresholds['mad_low'] * mad), 0, 1)
        scores[mask_high] = np.clip(np.abs(self.series[mask_high] - upper_bound) / (self.thresholds['mad_high'] * mad), 0, 1)

        outlier_mask = mask_low | mask_high
        df_outliers = pd.DataFrame({
            'score': pd.Series(scores, index=self.series.index)[outlier_mask]
        })

        return df_outliers
    
    def ecod_symmetric_score_outliers(self, threshold=0.0):
        x = self.series.dropna().values
        n = len(x)
        if n < 3 or len(np.unique(x)) < 2:
            return pd.DataFrame(columns=['score'])
        
        sorted_x = np.sort(x)

        def ecdf(val):
            return np.searchsorted(sorted_x, val, side='right') / n

        ecdf_vals = np.vectorize(ecdf)(x)
        rarity = np.minimum(ecdf_vals, 1 - ecdf_vals)

        with np.errstate(divide='ignore'):
            scores = -np.log(rarity)

        max_score = np.max(scores[np.isfinite(scores)])
        scores[np.isinf(scores)] = max_score * 1.5

        scores = scores / np.max(scores)

        outlier_mask = scores > threshold

        outliers = self.series.dropna().iloc[outlier_mask]
        outlier_scores = scores[outlier_mask]

        df_outliers = pd.DataFrame({
            'score': outlier_scores
        })

        return df_outliers

    def ecod_asymmetric_score_outliers(self, threshold=0.5):
        """
        Detects outliers using ECOD with KDE as the center and ECDF as the score.
        Falls back to symmetric ECOD if KDE/ECDF fails.
        """
        series = self.series.dropna()
        if series.empty or len(np.unique(series.values)) < 3:
            return pd.DataFrame(columns=['score'])

        x = series.values

        try:
            kde = gaussian_kde(x)
            xs = np.linspace(np.min(x), np.max(x), 1000)
            peak = xs[np.argmax(kde(xs))]

            sorted_data = np.sort(x)
            n = len(x)
            def ecdf_func(v):
                return np.searchsorted(sorted_data, v, side='right') / n

            raw_scores = 2 * np.abs(np.array([ecdf_func(v) for v in x]) - ecdf_func(peak))
            norm_scores = np.clip(raw_scores / threshold, 0, 1)

            outlier_mask = norm_scores > 0
            outliers = series[outlier_mask]
            scores = norm_scores[outlier_mask]

            return pd.DataFrame({'score': scores}, index=outliers.index)
        except Exception as e:
            logger.warning(
                f"ECOD asymmetric scoring failed. Falling back to symmetric ECOD."
            )
            return self.ecod_symmetric_score_outliers()
        
    def detect_numeric_outliers(self):
        """
        Applies IQR, MAD, and ECOD, and combines the weighted scores.
        """
        if not self._valid:
            return {
                "idx": [],
                "outliers_score": [],
                "total": 0,
                "info": self.info
                }
        weights = self.outlier_methods.get('weights')
        method_keys = list(weights.keys())

        iqr_df = self.iqr_score_outliers()
        mad_df = self.mad_score_outliers()
        if self.distribution_classification['symmetry'] == 'symmetric':
            ecod_df = self.ecod_symmetric_score_outliers()
        else:
            ecod_df = self.ecod_asymmetric_score_outliers()

        df_scores = pd.DataFrame({'value': self.series})
        
        for key in method_keys:
            if 'IQR' in key:
                iqr_df_renamed = iqr_df.rename(columns={'score': key})
                df_scores = df_scores.join(iqr_df_renamed, how='left')
            elif 'MAD' in key:
                mad_df_renamed = mad_df.rename(columns={'score': key})
                df_scores = df_scores.join(mad_df_renamed, how='left')
            elif 'ECOD' in key:
                ecod_df_renamed = ecod_df.rename(columns={'score': key})
                df_scores = df_scores.join(ecod_df_renamed, how='left')
        
        # Weights are normalized to sum to 1
        gamma = 1.5
        N = len(method_keys)
        numerador = 0
        denominador = 0
        detected_count = 0
        for k in method_keys:
            w = weights.get(k, 0)
            col = df_scores[k]
            col_filled = col.fillna(0).infer_objects(copy=False)
            numerador += col_filled * w
            denominador += w * col.notna().astype(int)
            detected_count += ((col > 0).fillna(False)).astype(int)
            

        consensus = detected_count / N
        penalty = consensus ** gamma
        
        # Score final
        avg_prob = numerador / (denominador.replace(0, np.nan))
        df_scores['combined_score'] = avg_prob * penalty
        df_scores['combined_score'] = df_scores['combined_score'].round(3)

        df_scores = df_scores.fillna(0).infer_objects(copy=False)
        df_filtrado = df_scores[df_scores['combined_score'] > self.outlier_threshold]
        valores_unicos = df_filtrado.drop_duplicates(subset=['value'])[['value', 'combined_score']]
        valores_dict = dict(zip(valores_unicos['value'], valores_unicos['combined_score']))
        
        final_info = {
            "idx": df_filtrado.index.tolist(),
            "outliers_score": valores_dict,
            "total": len(df_filtrado),
            "info": self.info
        }
        return final_info
        
    def detect_rare_categories(self, rare_threshold=0.01):
        freqs = self.series.value_counts(normalize=True, dropna=True)
        rare = freqs[freqs < rare_threshold]
        outliers = rare.index.tolist()
        idx = self.series[self.series.isin(outliers)].index.tolist()

        min_freq = rare.min()
        max_freq = rare.max()
        if pd.isna(min_freq) or pd.isna(max_freq):
            return {
                "idx": [],
                "outliers_score": [],
                "total": 0,
                "info": "No outliers detected."
            }
            
        if max_freq > min_freq:
            score = [1 - (freqs[v] - min_freq) / (max_freq - min_freq) for v in outliers]
        else:
            score = [1.0 for _ in outliers]
            
        score = [round(s, 3) for s in score]
        unique_scores = dict(zip(outliers, score))


        final_info = {
            "idx": idx,
            "outliers_score": unique_scores,
            "total": len(idx),
            "info": self.info
        }
        return final_info
        
    def detect_datetime_outliers(
        self,
        iqr_multiplier: float = 1.5,
        mad_threshold: float = 3.5
        ) -> dict:
        """
        Detects outliers in a datetime series by combining IQR, MAD.

        Returns:
            A dictionary with:
            - values: Series with the values and the number of methods that flagged them as outliers
            - total: Total number of unique outliers
            - info: Contextual information
        """
        if not pd.api.types.is_datetime64_any_dtype(self.series):
            return {
                "idx": [],
                "outliers_score": [],
                "total": 0,
                "info": "The series is not of datetime type."
            }

        timestamps = self.series.dropna().astype("int64")  # nanoseconds
        index_clean = timestamps.index

        # -----------------------------
        # IQR
        q1, q3 = np.percentile(timestamps, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr

        iqr_dist = np.zeros_like(timestamps, dtype=float)
        iqr_dist[timestamps < lower] = (lower - timestamps[timestamps < lower]) / iqr
        iqr_dist[timestamps > upper] = (timestamps[timestamps > upper] - upper) / iqr
        iqr_score = expit(iqr_dist)  # map to [0, 1]

        # -----------------------------
        # MAD
        median = np.median(timestamps)
        mad = np.median(np.abs(timestamps - median))
        mad_score = np.zeros_like(timestamps, dtype=float)

        if mad > 0:
            modified_z = 0.6745 * (timestamps - median) / mad
            mad_score = expit(np.abs(modified_z))  # map to [0, 1]


        combined_score = 0.5 * iqr_score + 0.5 * mad_score
        combined_score = combined_score.round(3)
        combined_series = pd.Series(combined_score, index=index_clean)

    
        threshold = 0.7
        final_outliers = combined_series[combined_series > threshold]
        values_scores = {idx_val: float(score) for idx_val, score in final_outliers.items()}

        

        return {
            "idx": final_outliers.index.tolist(),
            "outliers_score": values_scores,
            "total": len(final_outliers),
            "info": self.info
        }
        
    def detectar_outliers_texto_por_longitud(self):
        longitud = self.series.astype(str).str.len()
        self.series = longitud
        self.skew, self.kurtosis = self._compute_distribution_features()
        self.distribution_classification = self._classify_distribution()
        self.outlier_methods = self._select_outlier_methods()
        self.thresholds = self._get_thresholds()
        final_info = self.detect_numeric_outliers()   
        return final_info
            
    def detect_outliers(self):
        
        result = {
            "summary": {
                "n_outliers": 0,
                "n_rare": 0,
                "n_possible": 0
            },
            "outliers_detected": {},
            "rare_categories": {},
            "possible_outliers": {}
        }
        if not self._valid:
            return result
            
        if self.col_type in ["integer", "float", "numeric"]:
            final_info = self.detect_numeric_outliers()
            result["summary"]["n_outliers"] = final_info["total"]
            result["outliers_detected"] = final_info
        elif self.col_type =="category":
            final_info = self.detect_rare_categories()
            result["summary"]["n_rare"] = final_info["total"]
            result["rare_categories"] = final_info
        elif self.col_type == "datetime":
            final_info = self.detect_datetime_outliers()
            result["summary"]["n_possible"] = final_info["total"]
            result["possible_outliers"] = final_info
        elif self.col_type in ["boolean", "constant", "other"]:
            final_info = {"values": [], "idx": [], "total": 0, "info": "Outlier analysis is not applicable."}
        else:
            final_info = self.detectar_outliers_texto_por_longitud()
            result["summary"]["n_possible"] = final_info["total"]
            result["possible_outliers"] = final_info
        return result
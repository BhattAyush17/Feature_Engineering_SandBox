"""
Feature engineering and analysis layer.

Handles feature creation, statistical transformations, correlation analysis,
and feature importance extraction. Model-agnostic functions.
No Streamlit imports.
"""

import pandas as pd
import numpy as np
from src.data.data import get_column_types


MODEL_GUIDANCE = {
    'Logistic Regression': "Sensitive to scale and multicollinearity. Scale features and check VIF.",
    'KNN': "Distance-based; unscaled features dominate. Scaling is critical.",
    'SVM': "Sensitive to scale. Always scale.",
    'Decision Tree': "Scale-invariant but can overfit on noisy features."
}


def get_model_guidance(model_name):
    """Get guidance for the selected model."""
    return MODEL_GUIDANCE.get(model_name, "No specific guidance available.")


def compute_signal_strength(df, feature, target_col):
    """
    Compute signal strength between feature and target.
    
    Returns:
        str: 'High', 'Medium', 'Low', or 'N/A'
    """
    if feature == target_col:
        return 'N/A'

    target = df[target_col]
    feature_data = df[feature]

    def _label(strength: float) -> str:
        if pd.isna(strength):
            return 'Low'
        if strength >= 0.5:
            return 'High'
        if strength >= 0.2:
            return 'Medium'
        return 'Low'

    try:
        feature_is_num = pd.api.types.is_numeric_dtype(feature_data)
        target_is_num = pd.api.types.is_numeric_dtype(target)

        if feature_is_num and target_is_num:
            strength = abs(feature_data.corr(target))
            return _label(float(strength))

        def _eta_squared(values: pd.Series, groups: pd.Series) -> float:
            valid = pd.DataFrame({'v': values, 'g': groups}).dropna()
            if valid.empty:
                return 0.0
            overall_mean = valid['v'].mean()
            grouped = valid.groupby('g')['v']
            ss_between = ((grouped.mean() - overall_mean) ** 2 * grouped.size()).sum()
            ss_total = ((valid['v'] - overall_mean) ** 2).sum()
            if ss_total == 0:
                return 0.0
            return float(ss_between / ss_total)

        if feature_is_num and not target_is_num:
            return _label(_eta_squared(feature_data, target))

        if (not feature_is_num) and target_is_num:
            return _label(_eta_squared(target, feature_data))

        # Categorical vs categorical - use Cramer's V
        contingency = pd.crosstab(feature_data, target)
        if contingency.size == 0:
            return 'Low'
        observed = contingency.to_numpy(dtype=float)
        n = observed.sum()
        if n == 0:
            return 'Low'
        row_sums = observed.sum(axis=1, keepdims=True)
        col_sums = observed.sum(axis=0, keepdims=True)
        expected = (row_sums @ col_sums) / n
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2 = np.nansum((observed - expected) ** 2 / expected)
        r, k = observed.shape
        denom = min(k - 1, r - 1)
        if denom <= 0:
            return 'Low'
        strength = float(np.sqrt((chi2 / n) / denom))
        return _label(strength)
    except Exception:
        return 'Low'


def compute_scaling_sensitivity(df, feature):
    """
    Estimate how sensitive a feature is to scaling.
    
    Returns:
        str: 'High', 'Medium', 'Low', or 'N/A'
    """
    feature_data = df[feature]
    
    if not pd.api.types.is_numeric_dtype(feature_data):
        return 'N/A'
    
    try:
        std = feature_data.std()
        mean = abs(feature_data.mean()) + 1e-10
        cv = std / mean
        data_range = feature_data.max() - feature_data.min()
        
        if data_range > 100 or cv > 1:
            return 'High'
        elif data_range > 10 or cv > 0.3:
            return 'Medium'
        else:
            return 'Low'
    except:
        return 'Low'


def get_model_compatibility(feature, signal, scaling_sens, model_name, is_numeric):
    """
    Determine model compatibility for a feature.
    
    Returns:
        str: 'Good', 'Risky', or 'Weak'
    """
    if signal == 'Low':
        return 'Weak'
    
    if model_name in ['Logistic Regression', 'KNN', 'SVM']:
        if scaling_sens == 'High' and signal in ['High', 'Medium']:
            return 'Risky'
        elif signal == 'High':
            return 'Good'
        else:
            return 'Good'
    
    elif model_name == 'Decision Tree':
        if signal == 'High':
            return 'Good'
        else:
            return 'Good'
    
    return 'Good'


def get_recommendation(signal, scaling_sens, model_compat, model_name):
    """Generate feature recommendation."""
    if signal == 'Low':
        return 'Optional (drop if noisy)'
    
    if model_compat == 'Risky':
        if model_name in ['Logistic Regression']:
            return 'Scale + Regularize'
        else:
            return 'Scale first'
    
    if scaling_sens == 'High' and model_name in ['KNN', 'SVM']:
        return 'Must scale'
    
    if signal == 'High':
        return 'Keep (strong)'
    
    return 'Keep'


def get_feature_flags(signal, scaling_sens, model_name, is_numeric):
    """
    Return emoji flags for quick scanning.
    
    Returns:
        str: Emoji flags or '—' if none
    """
    flags = []
    
    if signal == 'High':
        flags.append('⭐')
    
    if scaling_sens == 'High' and model_name in ['Logistic Regression', 'KNN', 'SVM']:
        flags.append('⚖️')
    
    if model_name == 'Decision Tree' and signal in ['High', 'Medium']:
        flags.append('🌳')
    
    if model_name == 'KNN' and is_numeric:
        flags.append('📏')
    
    return ' '.join(flags) if flags else '—'


def analyze_features(df, target_col, model_name):
    """
    Generate feature suitability analysis table.
    
    Args:
        df: DataFrame to analyze
        target_col: Target column name
        model_name: Model to analyze for
        
    Returns:
        DataFrame: Feature analysis results
    """
    numeric_cols, categorical_cols = get_column_types(df)
    all_features = [c for c in df.columns if c != target_col]
    
    results = []
    
    for feature in all_features:
        is_numeric = feature in numeric_cols
        
        signal = compute_signal_strength(df, feature, target_col)
        scaling_sens = compute_scaling_sensitivity(df, feature)
        model_compat = get_model_compatibility(
            feature, signal, scaling_sens, model_name, is_numeric
        )
        recommendation = get_recommendation(signal, scaling_sens, model_compat, model_name)
        flags = get_feature_flags(signal, scaling_sens, model_name, is_numeric)
        
        results.append({
            'Feature': feature,
            'Flags': flags,
            'Signal': signal,
            'Scale Sensitivity': scaling_sens,
            'Compatibility': model_compat,
            'Recommendation': recommendation
        })
    
    return pd.DataFrame(results)


def get_feature_impact_ranking(model, feature_names, model_name, top_n=3):
    """
    Extract top N important features from trained model.
    
    Args:
        model: Trained model instance
        feature_names: List of feature names
        model_name: Name of the model type
        top_n: Number of top features to return
        
    Returns:
        list: List of (feature_name, importance) tuples
    """
    try:
        if model_name in ['Logistic Regression', 'SVM']:
            if hasattr(model, 'coef_'):
                coefs = np.abs(model.coef_)
                if coefs.ndim > 1:
                    coefs = coefs.mean(axis=0)
                importance = coefs
            else:
                return []
        
        elif model_name == 'Decision Tree':
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                return []
        
        elif model_name == 'KNN':
            return []
        
        else:
            return []
        
        indices = np.argsort(importance)[::-1][:top_n]
        
        result = []
        for idx in indices:
            if idx < len(feature_names):
                result.append((feature_names[idx], float(importance[idx])))
        
        return result
    
    except Exception:
        return []


def compute_correlation_matrix(df, target_col=None):
    """
    Compute correlation matrix for numeric features.
    
    Returns:
        DataFrame: Correlation matrix
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col and target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])
    
    if numeric_df.empty:
        return None
    
    return numeric_df.corr()


def get_highly_correlated_pairs(df, target_col=None, threshold=0.8):
    """
    Find highly correlated feature pairs.
    
    Returns:
        list: List of dicts with 'pair' and 'corr' keys
    """
    corr_matrix = compute_correlation_matrix(df, target_col)
    
    if corr_matrix is None:
        return []
    
    pairs = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:
                corr_val = abs(corr_matrix.loc[col1, col2])
                if corr_val >= threshold:
                    pairs.append({
                        'pair': f"{col1} ↔ {col2}",
                        'corr': corr_val
                    })
    
    return pairs


def create_interaction_features(df, feature_pairs):
    """
    Create interaction features from pairs.
    
    Args:
        df: DataFrame
        feature_pairs: List of (feature1, feature2) tuples
        
    Returns:
        DataFrame: DataFrame with new interaction columns
    """
    df_new = df.copy()
    
    for f1, f2 in feature_pairs:
        if f1 in df.columns and f2 in df.columns:
            # Multiplicative interaction
            if pd.api.types.is_numeric_dtype(df[f1]) and pd.api.types.is_numeric_dtype(df[f2]):
                df_new[f"{f1}_x_{f2}"] = df[f1] * df[f2]
    
    return df_new


def create_polynomial_features(df, numeric_cols, degree=2):
    """
    Create polynomial features.
    
    Args:
        df: DataFrame
        numeric_cols: List of numeric columns
        degree: Polynomial degree
        
    Returns:
        DataFrame: DataFrame with polynomial features
    """
    df_new = df.copy()
    
    for col in numeric_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            for d in range(2, degree + 1):
                df_new[f"{col}^{d}"] = df[col] ** d
    
    return df_new

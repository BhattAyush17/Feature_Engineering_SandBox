"""
Model verification and validation layer.

Handles cross-validation, ablation studies, sensitivity analysis,
and performance stability checks.
No Streamlit imports.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.preprocessing.preprocessing import impute_data, encode_data, scale_data
from src.data.data import get_column_types


def _preprocess(df, target_col, params):
    """Preprocess data and return X, y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    numeric_cols, categorical_cols = get_column_types(X)
    strat_imp = params.get('imputation', 'Drop Rows')
    strat_enc = params.get('encoding', 'Label Encoding')
    strat_scale = params.get('scaling', 'No Scaling')
    
    X_p, imputer = impute_data(X.copy(), strat_imp, numeric_cols)
    y_p = y.loc[X_p.index]
    X_p, encoder = encode_data(X_p, strat_enc, categorical_cols)
    numeric_after, _ = get_column_types(X_p)
    X_p, scaler = scale_data(X_p, strat_scale, numeric_after)
    
    return X_p, y_p


def _get_model(model_type):
    """Get model instance by type."""
    from src.models.model_logic import get_model
    return get_model(model_type)


def _model_type_from_instance(model):
    """Infer model type from instance."""
    name = type(model).__name__
    if 'Logistic' in name:
        return 'Logistic Regression'
    elif 'KNeighbors' in name:
        return 'KNN'
    elif 'SVC' in name:
        return 'SVM'
    return 'Decision Tree'


def run_feature_ablation(df, target_col, model, preprocessing_params, feature_names=None, max_features=8):
    """
    Measure accuracy drop when each feature is removed.
    
    Args:
        df: DataFrame with features and target
        target_col: Target column name
        model: Trained model instance
        preprocessing_params: Preprocessing configuration
        feature_names: List of feature names (optional)
        max_features: Maximum features to test
        
    Returns:
        dict: Baseline accuracy and ablation results
    """
    X_p, y_p = _preprocess(df, target_col, preprocessing_params)
    X_train, X_test, y_train, y_test = train_test_split(
        X_p, y_p, test_size=0.2, random_state=42
    )
    
    model_type = _model_type_from_instance(model)
    
    # Train baseline
    baseline_model = _get_model(model_type)
    baseline_model.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test))
    
    features = list(X_train.columns)[:max_features]
    results = []
    
    for feat in features:
        X_tr = X_train.drop(columns=[feat])
        X_te = X_test.drop(columns=[feat])
        try:
            m = _get_model(model_type)
            m.fit(X_tr, y_train)
            acc = accuracy_score(y_test, m.predict(X_te))
            drop = baseline_acc - acc
        except:
            drop = 0
        results.append({'feature': feat, 'drop': drop})
    
    results.sort(key=lambda x: x['drop'], reverse=True)
    return {'baseline_accuracy': baseline_acc, 'ablation': results}


def plot_ablation_chart(ablation_data):
    """
    Create bar chart of accuracy drop per feature.
    
    Args:
        ablation_data: Output from run_feature_ablation
        
    Returns:
        plotly Figure
    """
    results = ablation_data['ablation']
    baseline = ablation_data['baseline_accuracy']
    
    features = [r['feature'] for r in results]
    drops = [r['drop'] * 100 for r in results]
    colors = [
        '#27ae60' if d > 0.5 else '#e74c3c' if d < -0.5 else '#95a5a6' 
        for d in drops
    ]
    
    fig = go.Figure(go.Bar(
        x=drops, y=features, orientation='h',
        marker_color=colors,
        text=[f'{d:+.1f}%' for d in drops],
        textposition='outside'
    ))
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(
        title=f"Feature Ablation (Baseline: {baseline*100:.1f}%)",
        xaxis_title="Accuracy Drop (%)",
        height=max(250, len(features) * 35),
        margin=dict(l=10, r=10, t=50, b=30),
        yaxis=dict(categoryorder='total ascending')
    )
    return fig


def run_sensitivity_analysis(df, target_col, model, preprocessing_params, feature_to_vary, n_points=20):
    """
    Analyze how predictions change when varying one feature.
    
    Args:
        df: DataFrame with features and target
        target_col: Target column name
        model: Trained model instance
        preprocessing_params: Preprocessing configuration
        feature_to_vary: Feature to vary
        n_points: Number of points to test
        
    Returns:
        dict: Sensitivity analysis results or None
    """
    X_p, _ = _preprocess(df, target_col, preprocessing_params)
    
    feat = feature_to_vary if feature_to_vary in X_p.columns else None
    if not feat:
        match = [c for c in X_p.columns if feature_to_vary in c]
        feat = match[0] if match else None
    if not feat:
        return None
    
    median_row = X_p.median()
    f_min, f_max = X_p[feat].min(), X_p[feat].max()
    values = np.linspace(f_min, f_max, n_points)
    
    synthetic = pd.DataFrame([median_row] * n_points)
    synthetic[feat] = values
    
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(synthetic)
            preds = probs[:, 1] if probs.shape[1] == 2 else probs.max(axis=1)
        else:
            preds = model.predict(synthetic).astype(float)
    except:
        return None
    
    pred_range = preds.max() - preds.min()
    noise = np.std(np.diff(preds))
    
    if pred_range < 0.05:
        response_type = 'flat'
    elif noise > 0.1:
        response_type = 'noisy'
    else:
        response_type = 'smooth'
    
    return {
        'feature': feat, 
        'values': values, 
        'predictions': preds, 
        'response_type': response_type
    }


def plot_sensitivity_chart(sensitivity_data, target_col):
    """
    Plot prediction vs feature value.
    
    Args:
        sensitivity_data: Output from run_sensitivity_analysis
        target_col: Target column name
        
    Returns:
        plotly Figure or None
    """
    if not sensitivity_data:
        return None
    
    feat = sensitivity_data['feature']
    x = sensitivity_data['values']
    y = sensitivity_data['predictions']
    resp = sensitivity_data['response_type']
    
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode='lines+markers',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6)
    ))
    
    labels = {'flat': 'Flat', 'noisy': 'Noisy', 'smooth': 'Smooth'}
    fig.add_annotation(
        text=labels.get(resp, ''), xref="paper", yref="paper",
        x=0.02, y=0.98, showarrow=False,
        font=dict(size=12, color='orange' if resp != 'smooth' else 'green')
    )
    
    fig.update_layout(
        title=f"Sensitivity: {feat}",
        xaxis_title=feat,
        yaxis_title=f"P({target_col}=1)" if max(y) <= 1 else "Prediction",
        height=300, margin=dict(l=10, r=10, t=50, b=30)
    )
    return fig


def run_multi_model_response(df, target_col, preprocessing_params, feature_to_vary, n_points=25):
    """
    Compare how different models respond to one feature.
    
    Args:
        df: DataFrame with features and target
        target_col: Target column name
        preprocessing_params: Preprocessing configuration
        feature_to_vary: Feature to vary
        n_points: Number of points to test
        
    Returns:
        dict: Model response data or None
    """
    X_p, y_p = _preprocess(df, target_col, preprocessing_params)
    
    feat = feature_to_vary if feature_to_vary in X_p.columns else None
    if not feat:
        match = [c for c in X_p.columns if feature_to_vary in c]
        feat = match[0] if match else None
    if not feat:
        return None
    
    X_train, _, y_train, _ = train_test_split(X_p, y_p, test_size=0.2, random_state=42)
    
    models = {}
    for name in ['Logistic Regression', 'KNN', 'Decision Tree']:
        try:
            m = _get_model(name)
            m.fit(X_train, y_train)
            models[name] = m
        except:
            pass
    
    if not models:
        return None
    
    median_row = X_p.median()
    values = np.linspace(X_p[feat].quantile(0.05), X_p[feat].quantile(0.95), n_points)
    synthetic = pd.DataFrame([median_row] * n_points)
    synthetic[feat] = values
    
    preds_dict = {}
    for name, m in models.items():
        try:
            if hasattr(m, 'predict_proba'):
                probs = m.predict_proba(synthetic)
                p = probs[:, 1] if probs.shape[1] == 2 else probs.max(axis=1)
            else:
                p = m.predict(synthetic).astype(float)
            preds_dict[name] = p.tolist()
        except:
            pass
    
    return {'feature': feat, 'values': values.tolist(), 'model_predictions': preds_dict}


def plot_multi_model_response(response_data, target_col):
    """
    Create line plot comparing model predictions vs feature.
    
    Args:
        response_data: Output from run_multi_model_response
        target_col: Target column name
        
    Returns:
        plotly Figure or None
    """
    if not response_data or not response_data.get('model_predictions'):
        return None
    
    feat = response_data['feature']
    x = response_data['values']
    preds = response_data['model_predictions']
    
    colors = {
        'Logistic Regression': '#3498db', 
        'KNN': '#e74c3c', 
        'Decision Tree': '#27ae60'
    }
    
    fig = go.Figure()
    for name, y in preds.items():
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines+markers', name=name,
            line=dict(color=colors.get(name, '#95a5a6'), width=2),
            marker=dict(size=5)
        ))
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=f"Model Comparison: {feat}",
        xaxis_title=feat,
        yaxis_title=f"P({target_col}=1)",
        height=350, margin=dict(l=10, r=10, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        yaxis=dict(range=[-0.05, 1.05])
    )
    return fig


def get_verification_summary(ablation_data, sensitivity_data=None):
    """
    Generate brief summary of verification results.
    
    Args:
        ablation_data: Output from run_feature_ablation
        sensitivity_data: Output from run_sensitivity_analysis (optional)
        
    Returns:
        str: Summary text
    """
    results = ablation_data['ablation']
    top = [r for r in results if r['drop'] > 0.01][:3]
    
    if not top:
        return "No single feature dominates accuracy."
    
    names = ', '.join([r['feature'] for r in top])
    drops = ', '.join([f"{r['drop']*100:+.1f}%" for r in top])
    summary = f"Key features: {names} ({drops})"
    
    if sensitivity_data and sensitivity_data.get('response_type') == 'flat':
        summary += f" | {sensitivity_data['feature']} has flat response"
    
    return summary

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_target_distribution(df, target_col):
    """Plot target distribution."""
    target = df[target_col]
    n_unique = target.nunique()
    
    if n_unique <= 20:
        value_counts = target.value_counts().reset_index()
        value_counts.columns = [target_col, 'Count']
        fig = px.bar(value_counts, x=target_col, y='Count', 
                     title=f"Target Distribution: {target_col}",
                     color=target_col)
    else:
        fig = px.histogram(df, x=target_col, nbins=30,
                          title=f"Target Distribution: {target_col} (Continuous)",
                          color_discrete_sequence=['#3498db'])
        fig.add_vline(x=target.mean(), line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {target.mean():.2f}")
    
    fig.update_layout(showlegend=False)
    return fig


def plot_numeric_distributions(df, target_col=None, max_features=6):
    """Plot histograms of numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col and target_col in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != target_col]
    
    if not numeric_cols:
        return None
    
    plot_cols = numeric_cols[:max_features]
    n_cols = min(3, len(plot_cols))
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=plot_cols)
    
    for i, col in enumerate(plot_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1
        fig.add_trace(
            go.Histogram(x=df[col], name=col, marker_color='#2ecc71', showlegend=False),
            row=row, col=col_idx
        )
    
    fig.update_layout(title_text="Numeric Feature Distributions", height=300 * n_rows)
    return fig


def plot_feature_vs_target(df, target_col, max_features=4):
    """Plot numeric features against the target."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    feature_cols = [c for c in numeric_cols if c != target_col]
    
    if not feature_cols:
        return None
    
    plot_cols = feature_cols[:max_features]
    target_is_numeric = df[target_col].dtype in [np.float64, np.int64, np.float32, np.int32]
    target_unique = df[target_col].nunique()
    
    n_cols = min(2, len(plot_cols))
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=plot_cols)
    
    for i, col in enumerate(plot_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1
        
        if target_is_numeric and target_unique > 20:
            fig.add_trace(
                go.Scatter(x=df[col], y=df[target_col], mode='markers', 
                          marker=dict(opacity=0.5, size=5), name=col, showlegend=False),
                row=row, col=col_idx
            )
        else:
            for target_val in df[target_col].unique()[:10]:
                mask = df[target_col] == target_val
                fig.add_trace(
                    go.Box(y=df.loc[mask, col], name=str(target_val), showlegend=False),
                    row=row, col=col_idx
                )
    
    fig.update_layout(title_text=f"Features vs {target_col}", height=350 * n_rows)
    return fig


def plot_missing_heatmap(df):
    missing_matrix = df.isnull()
    if missing_matrix.sum().sum() == 0:
        return None
        
    fig = px.imshow(missing_matrix, 
                    labels=dict(x="Features", y="Rows", color="Missing"),
                    color_continuous_scale=['#eee', '#e74c3c'],
                    title="Missing Value Heatmap")
    return fig

def plot_distributions(df_before, df_after, column):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_before[column], name='Original', opacity=0.5, marker_color='blue'))
    fig.add_trace(go.Histogram(x=df_after[column], name='Processed', opacity=0.5, marker_color='orange'))
    fig.update_layout(barmode='overlay', title=f"Distribution Shift: {column}", xaxis_title=column, yaxis_title="Count")
    return fig

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None
        
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlation Matrix")
    return fig


# Decision-oriented plots for feature selection

def plot_feature_target_separation(df, target_col, max_features=6):
    """Boxplots showing feature separation by target class."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]
    
    if not feature_cols:
        return None
    
    target = df[target_col]
    target_unique = target.nunique()
    
    if target_unique > 20:
        return None
    
    plot_cols = feature_cols[:max_features]
    n_cols = min(3, len(plot_cols))
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=plot_cols)
    
    colors = px.colors.qualitative.Set2
    
    for i, col in enumerate(plot_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1
        
        for j, target_val in enumerate(sorted(target.unique())[:5]):
            mask = target == target_val
            fig.add_trace(
                go.Box(y=df.loc[mask, col], name=str(target_val), 
                       marker_color=colors[j % len(colors)], showlegend=(i == 0)),
                row=row, col=col_idx
            )
    
    fig.update_layout(
        title_text="Feature vs Target Separation (Clear separation = useful feature)",
        height=300 * n_rows,
        boxmode='group'
    )
    return fig


def plot_correlation_to_target(df, target_col):
    """Sorted bar plot of absolute correlation with target."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col not in numeric_df.columns:
        target_encoded = pd.factorize(df[target_col])[0]
    else:
        target_encoded = numeric_df[target_col]
    
    feature_cols = [c for c in numeric_df.columns if c != target_col]
    
    if not feature_cols:
        return None
    
    correlations = []
    for col in feature_cols:
        try:
            corr = abs(numeric_df[col].corr(pd.Series(target_encoded, index=numeric_df.index)))
            if not pd.isna(corr):
                correlations.append({'Feature': col, 'Correlation': corr})
        except:
            pass
    
    if not correlations:
        return None
    
    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
    
    max_corr = corr_df['Correlation'].max() if corr_df['Correlation'].max() > 0 else 1
    colors = [f'rgba(52, 152, 219, {0.3 + 0.7 * (c / max_corr)})' for c in corr_df['Correlation']]
    colors[:3] = ['#2980b9', '#3498db', '#5dade2'][:len(colors[:3])]
    
    fig = go.Figure(go.Bar(
        x=corr_df['Correlation'],
        y=corr_df['Feature'],
        orientation='h',
        marker_color=colors,
        text=[f'{c:.2f}' for c in corr_df['Correlation']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Feature Importance (Correlation with Target)",
        xaxis_title="|Correlation|",
        yaxis_title="",
        height=max(250, len(feature_cols) * 30),
        xaxis=dict(range=[0, 1.05]),
        margin=dict(l=10, r=10, t=40, b=30),
        yaxis=dict(categoryorder='total ascending')
    )
    
    return fig, corr_df.head(3)['Feature'].tolist()


def plot_feature_scale_comparison(df, target_col=None):
    """Bar chart showing scale differences across features."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col and target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])
    
    if numeric_df.empty:
        return None, []
    
    scale_data = []
    for col in numeric_df.columns:
        data = numeric_df[col].dropna()
        if len(data) > 0:
            scale_data.append({
                'Feature': col,
                'Range': data.max() - data.min()
            })
    
    if not scale_data:
        return None, []
    
    scale_df = pd.DataFrame(scale_data).sort_values('Range', ascending=False)
    
    max_range = scale_df['Range'].max()
    scale_df['Normalized'] = scale_df['Range'] / max_range
    colors = ['#e74c3c' if r > 0.5 else '#bdc3c7' for r in scale_df['Normalized']]
    dominant_features = scale_df[scale_df['Normalized'] > 0.5]['Feature'].tolist()
    
    fig = go.Figure(go.Bar(
        x=scale_df['Range'],
        y=scale_df['Feature'],
        orientation='h',
        marker_color=colors,
        text=[f'{r:.0f}' for r in scale_df['Range']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Red = large range, scale for LR/KNN/SVM",
        xaxis_title="Range",
        yaxis_title="",
        height=max(250, len(scale_df) * 30),
        margin=dict(l=10, r=10, t=40, b=30),
        yaxis=dict(categoryorder='total ascending')
    )
    
    return fig, dominant_features


def get_multicollinearity_pairs(df, target_col=None, threshold=0.8):
    """Return highly correlated feature pairs."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col and target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])
    
    if numeric_df.shape[1] < 2:
        return []
    
    corr_matrix = numeric_df.corr()
    
    high_corr_pairs = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:
                corr_val = abs(corr_matrix.loc[col1, col2])
                if corr_val >= threshold:
                    high_corr_pairs.append({
                        'pair': f"{col1} ↔ {col2}",
                        'corr': corr_val
                    })
    
    return high_corr_pairs


def generate_model_guidance(model_name, dominant_features, high_corr_pairs):
    """Generate actionable guidance."""
    guidance_parts = []
    
    if model_name in ['Logistic Regression', 'KNN', 'SVM']:
        if dominant_features:
            guidance_parts.append(f"Scale {', '.join(dominant_features[:2])}")
        if high_corr_pairs and model_name == 'Logistic Regression':
            pair = high_corr_pairs[0]
            guidance_parts.append(f"Consider dropping one of {pair['pair']} (r={pair['corr']:.2f})")
    
    elif model_name == 'Decision Tree':
        guidance_parts.append("Scaling not required")
        if high_corr_pairs:
            guidance_parts.append("Multicollinearity not critical for trees")
    
    if not guidance_parts:
        return "No immediate action needed."
    
    return ". ".join(guidance_parts) + "."


def plot_multicollinearity_warning(df, target_col=None, threshold=0.8):
    """Backwards compatible wrapper."""
    pairs = get_multicollinearity_pairs(df, target_col, threshold)
    return None, pairs


def plot_feature_importance_bar(feature_names, importances, model_name, top_n=5):
    """Bar plot of top N feature importances."""
    if not feature_names or importances is None or len(importances) == 0:
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(top_n)
    
    colors = px.colors.sequential.Viridis
    
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='#3498db',
        text=[f'{imp:.3f}' for imp in importance_df['Importance']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Features ({model_name})",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=max(250, top_n * 40)
    )
    
    return fig


def plot_performance_comparison(results):
    df_res = pd.DataFrame(results)
    if df_res.empty:
        return None
        
    df_melt = df_res.melt(id_vars=['Model', 'Experiment'], value_vars=['Accuracy', 'F1 Score'], 
                          var_name='Metric', value_name='Score')
    
    fig = px.bar(df_melt, x='Experiment', y='Score', color='Metric', barmode='group',
                 facet_col='Model', title="Performance Comparison")
    return fig

def plot_feature_space(df, target_col, feature_x, feature_y):
    fig = px.scatter(df, x=feature_x, y=feature_y, color=target_col, 
                     title=f"Feature Space: {feature_x} vs {feature_y}")
    return fig

"""
ML Feature Engineering Sandbox - Entry Point (UI Layer)

This is the Streamlit application entry point.
Acts as a controller between UI and core logic.

MUST stay here:
- Streamlit UI layout (st.title, st.sidebar, st.tabs, st.expander)
- User input controls (sliders, selects, file upload)
- High-level orchestration calls
- Rendering plots and metrics

MUST NOT be here:
- Data cleaning logic
- Feature engineering logic  
- Model training internals
- Evaluation math
- File system logic
"""

import os

# Prevent loky CPU detection crash
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import streamlit as st
import pandas as pd

# Absolute imports from src package
from src.data.data import (
    load_data, 
    generate_messy_data, 
    get_column_types,
    load_data_with_metadata, 
    get_schema_summary, 
    get_visualization_sample, 
    get_training_sample,
    MAX_ROWS_VISUALIZATION, 
    MAX_ROWS_TRAINING
)
from src.models.model_logic import (
    train_and_evaluate, 
    detect_target_type, 
    check_training_compatibility
)
from src.visualization.plots import (
    plot_target_distribution, 
    plot_numeric_distributions, 
    plot_correlation_heatmap, 
    plot_feature_vs_target, 
    plot_missing_heatmap,
    plot_feature_target_separation, 
    plot_correlation_to_target,
    plot_feature_scale_comparison, 
    get_multicollinearity_pairs,
    plot_feature_importance_bar, 
    generate_model_guidance
)
from src.features.feature_analysis import (
    analyze_features, 
    get_model_guidance, 
    get_feature_impact_ranking
)
from src.models.model_verification import (
    run_feature_ablation, 
    plot_ablation_chart, 
    run_sensitivity_analysis, 
    plot_sensitivity_chart,
    get_verification_summary,
    run_multi_model_response, 
    plot_multi_model_response
)

st.set_page_config(
    page_title="Feature_Engineering_SandBox",
    page_icon="🧪",
    layout="wide"
)

# Inject CSS
st.markdown("""
<style>
.stApp {
    background-color: #0b0b0b;
    color: #f5f5f5;
}
.main-header {
    background: linear-gradient(-45deg, #0a0a0a, #1a1a1a, #2d2d2d, #1a1a1a);
    background-size: 400% 400%;
    animation: gradientFlow 15s ease infinite;
    padding: 2.5rem 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}
@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.main-header h1 {
    color: #ffffff;
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 2rem;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin-top: 1.5rem;
}
.feature-item {
    background: transparent;
    padding: 1rem;
    border-radius: 8px;
    border-left: 3px solid rgba(180, 180, 180, 0.4);
    transition: all 0.3s ease;
    cursor: default;
}
.feature-item:hover {
    background: rgba(255, 255, 255, 0.08);
    border-left: 3px solid rgba(255, 255, 255, 0.7);
}
.feature-item:hover strong {
    color: #ffffff;
}
.feature-item strong {
    color: #c0c0c0;
    font-size: 1.1rem;
    transition: color 0.3s ease;
}
.feature-item span {
    color: #888888;
    font-size: 0.95rem;
}
.tagline {
    text-align: center;
    color: #707070;
    font-style: italic;
    margin-top: 1.5rem;
    font-size: 1rem;
}

/* Bold main section headers */
.stApp h2 {
    font-weight: 800 !important;
    font-size: 1.8rem !important;
    color: #ffffff !important;
    border-bottom: 2px solid rgba(255, 255, 255, 0.2);
    padding-bottom: 0.5rem;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stApp h3 {
    font-weight: 700 !important;
    font-size: 1.3rem !important;
    color: #e0e0e0 !important;
}
</style>
""", unsafe_allow_html=True)

# Header HTML
st.markdown("""
<div class="main-header">
<h1>Feature_Engineering_SandBox</h1>
<div class="feature-grid">
<div class="feature-item"><strong>Features</strong> <span>- transform, encode, select</span></div>
<div class="feature-item"><strong>Assumptions</strong> <span>- distributions, relationships</span></div>
<div class="feature-item"><strong>Models</strong> <span>- compare with stable metrics</span></div>
<div class="feature-item"><strong>Iteration</strong> <span>- experiment without risk</span></div>
</div>
<p class="tagline">A focused sandbox for ML experimentation</p>
</div>
""", unsafe_allow_html=True)

if 'df_full' not in st.session_state:
    st.session_state.df_full = None
    st.session_state.metadata = None
    st.session_state.schema = None

# Sidebar
with st.sidebar:
    st.header("Data Configuration")
    data_source = st.radio("Data Source", ["Generate Messy Data", "Upload CSV"])
    
    if data_source == "Generate Messy Data":
        n_rows = st.slider("Number of rows", 100, 10000, 1000, step=100)
        df_full = generate_messy_data(n_rows=n_rows)
        metadata = {
            'total_rows': n_rows,
            'loaded_rows': n_rows,
            'columns': len(df_full.columns),
            'file_size_mb': 0,
            'is_sampled': False
        }
        st.success(f"Generated {n_rows:,} rows with missing values and outliers.")
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            st.caption("Large CSVs are loaded as a capped sample for performance.")
            load_limit = st.number_input(
                "Max rows to load from CSV",
                min_value=5000,
                max_value=200000,
                value=MAX_ROWS_TRAINING,
                step=5000,
                help="For very large files (e.g., hundreds of MB), loading everything can be slow or run out of memory.",
            )
            df_full, metadata = load_data_with_metadata(uploaded_file, max_rows=int(load_limit))
            if df_full is None:
                st.error(f"Failed to load CSV: {metadata.get('error', 'Unknown error')}")
        else:
            df_full = None
            metadata = None
            st.info("Please upload a CSV file.")
    
    # Sampling controls
    if df_full is not None:
        st.divider()
        st.subheader("Sampling Controls")
        viz_limit = st.number_input("Max rows for plots", 1000, 50000, MAX_ROWS_VISUALIZATION, step=1000)
        train_limit = st.number_input("Max rows for training", 1000, 100000, MAX_ROWS_TRAINING, step=5000)

if df_full is not None:
    st.subheader("Dataset Overview")
    
    meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
    with meta_col1:
        total_rows = metadata.get('total_rows')
        st.metric("Total Rows", f"{total_rows:,}" if isinstance(total_rows, int) else "Unknown")
    with meta_col2:
        st.metric("Columns", metadata['columns'])
    with meta_col3:
        size_str = f"{metadata['file_size_mb']:.1f} MB" if metadata['file_size_mb'] > 0 else "Generated"
        st.metric("Size", size_str)
    with meta_col4:
        status = "Sampled" if metadata.get('is_sampled') else "Full"
        st.metric("Status", status)
    
    schema = get_schema_summary(df_full)
    schema_col1, schema_col2, schema_col3 = st.columns(3)
    with schema_col1:
        st.caption(f"**Numeric:** {schema['n_numeric']} columns")
    with schema_col2:
        st.caption(f"**Categorical:** {schema['n_categorical']} columns")
    with schema_col3:
        if schema['high_cardinality']:
            hc_info = ", ".join([f"{h['column']} ({h['unique']})" for h in schema['high_cardinality'][:2]])
            st.caption(f"△ **High-cardinality:** {hc_info}")
        else:
            st.caption("· No high-cardinality categoricals")
    
    if schema['missing_summary']:
        missing_str = ", ".join([f"{col} ({pct}%)" for col, pct in list(schema['missing_summary'].items())[:3]])
        st.warning(f"△ Missing values: {missing_str}")
    
    st.dataframe(df_full.head())
    
    target_col = st.selectbox("Select Target Column", df_full.columns, index=len(df_full.columns)-1)
    
    df_viz = get_visualization_sample(df_full, target_col) if len(df_full) > viz_limit else df_full
    df_train = get_training_sample(df_full, target_col) if len(df_full) > train_limit else df_full
    
    if len(df_full) > viz_limit or len(df_full) > train_limit:
        sample_col1, sample_col2 = st.columns(2)
        with sample_col1:
            st.caption(f"◇ Plots using: {len(df_viz):,} rows (stratified sample)")
        with sample_col2:
            st.caption(f"◆ Training using: {len(df_train):,} rows (stratified sample)")
    
    df = df_viz
    if target_col in df.columns:
        before_rows = len(df)
        df = df.dropna(subset=[target_col])
        dropped = before_rows - len(df)
        if dropped > 0:
            st.caption(f"· Dropped {dropped:,} row(s) with missing target for analysis.")
        if len(df) == 0:
            st.error("Target column is entirely missing in the selected sample. Choose another target or increase the sample size.")
            st.stop()
    
    target_info = detect_target_type(df[target_col])
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Target Type", target_info['task_type'].capitalize())
    with col_info2:
        st.metric("Unique Values", target_info['n_unique'])
    with col_info3:
        status = "Ready" if target_info['is_valid_for_training'] else "Classification N/A"
        st.metric("Training Status", status)
    
    st.header("Preprocessing Logic")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Imputation")
        imp_strat = st.selectbox("Imputation Strategy", 
                                ["Drop Rows", "Mean", "Median", "Most Frequent", "Constant"])
        
    with col2:
        st.subheader("Encoding")
        enc_strat = st.selectbox("Categorical Encoding", 
                                ["Label Encoding", "One-Hot Encoding", "Frequency Encoding"])
        
    with col3:
        st.subheader("Scaling")
        scale_strat = st.selectbox("Scaling Strategy", 
                                  ["No Scaling", "StandardScaler", "MinMaxScaler", "RobustScaler"])

    preprocessing_params = {
        'imputation': imp_strat,
        'encoding': enc_strat,
        'scaling': scale_strat
    }

    st.header("Model Selection")
    model_name = st.selectbox("Choose Model", 
                             ["Logistic Regression", "KNN", "SVM", "Decision Tree"])
    
    # Model-specific guidance (one sentence)
    st.info(f"{model_name}: {get_model_guidance(model_name)}")
    
    can_train, warning_msg = check_training_compatibility(target_info, model_name)
    
    run_btn = st.button("Train & Evaluate")
    
    trained_model_result = None
    if run_btn:
        if can_train:
            with st.spinner(f"Training on {len(df_train):,} rows..."):
                try:
                    metrics = train_and_evaluate(df_train, target_col, model_name, preprocessing_params)
                    trained_model_result = metrics
                    
                    st.subheader("Evaluation Results")
                    st.caption(f"Trained on {len(df_train):,} rows")
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    m_col2.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                    
                    if 'trained_model' in metrics and 'feature_names' in metrics:
                        top_features = get_feature_impact_ranking(
                            metrics['trained_model'], 
                            metrics['feature_names'], 
                            model_name, 
                            top_n=3
                        )
                        if top_features:
                            st.subheader(f"Top Influential Features ({model_name})")
                            for rank, (feat_name, importance) in enumerate(top_features, 1):
                                st.write(f"**{rank}.** {feat_name} — {importance:.4f}")
                        elif model_name == 'KNN':
                            st.caption("KNN does not provide feature importance.")
                    
                    st.header("Model Verification")
                    st.caption("Check that predictions depend on meaningful features.")
                    
                    # 1. Feature Ablation Analysis
                    with st.spinner("Running feature ablation analysis..."):
                        ablation_data = run_feature_ablation(
                            df_train, target_col, 
                            metrics['trained_model'], 
                            preprocessing_params,
                            metrics['feature_names'],
                            max_features=min(8, len(metrics['feature_names']))
                        )
                    
                    st.subheader("[A] Feature Ablation Test")
                    st.caption("Accuracy drop when each feature is removed. Larger drop means more important.")
                    
                    fig_ablation = plot_ablation_chart(ablation_data)
                    st.plotly_chart(fig_ablation, use_container_width=True)
                    
                    # Verification summary
                    summary = get_verification_summary(ablation_data)
                    st.markdown(summary)
                    
                    # 2. Feature Sensitivity Analysis
                    st.subheader("[B] Feature Sensitivity Test")
                    st.caption("How predictions change when varying one feature. Flat means no effect.")
                    
                    numeric_features = [f for f in metrics['feature_names'] 
                                       if f in df_train.select_dtypes(include=['number']).columns or 
                                       any(f in c for c in df_train.select_dtypes(include=['number']).columns)]
                    
                    if numeric_features:
                        default_feature = ablation_data['ablation'][0]['feature'] if ablation_data['ablation'] else numeric_features[0]
                        if default_feature not in numeric_features:
                            default_feature = numeric_features[0]
                        
                        sensitivity_feature = st.selectbox(
                            "Select feature to analyze",
                            options=numeric_features[:10],
                            index=0
                        )
                        
                        sensitivity_data = run_sensitivity_analysis(
                            df_train, target_col,
                            metrics['trained_model'],
                            metrics.get('preprocessors', {}),
                            sensitivity_feature
                        )
                        
                        if sensitivity_data:
                            fig_sensitivity = plot_sensitivity_chart(sensitivity_data, target_col)
                            if fig_sensitivity:
                                st.plotly_chart(fig_sensitivity, use_container_width=True)
                                
                                y_vals = sensitivity_data['predictions']
                                y_range = max(y_vals) - min(y_vals)
                                if y_range > 0.2:
                                    st.success(f"Good response: {sensitivity_feature} affects predictions.")
                                elif y_range > 0.05:
                                    st.info(f"Moderate response: {sensitivity_feature} has some effect.")
                                else:
                                    st.warning(f"Flat response: {sensitivity_feature} has little effect.")
                        else:
                            st.info("Could not analyze this feature (may be categorical or encoded).")
                    else:
                        st.info("No numeric features available for sensitivity analysis.")
                    
                except Exception as e:
                    st.error(f"Pipeline Failed: {str(e)}")
                    import traceback
                    st.text(traceback.format_exc())
        else:
            st.warning(warning_msg)
            st.info("Scroll down to explore your data visually.")
    
    st.header("Model Comparison")
    st.caption("How different models respond to feature changes")
    
    # Get numeric features for comparison
    numeric_features_for_compare = df_train.select_dtypes(include=['number']).columns.tolist()
    numeric_features_for_compare = [f for f in numeric_features_for_compare if f != target_col]
    
    if numeric_features_for_compare and can_train:
        primary_feature = st.selectbox(
            "Select feature to analyze",
            options=numeric_features_for_compare,
            key="primary_feature"
        )
        
        # Run comparison
        with st.spinner("Comparing models..."):
            response_data = run_multi_model_response(
                df_train, target_col, preprocessing_params, primary_feature
            )
        
        if response_data:
            st.subheader("[A] Feature Response Comparison")
            st.caption("How each model reacts as the feature changes")
            
            fig_compare = plot_multi_model_response(response_data, target_col)
            if fig_compare:
                st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.info("Could not run model comparison for this feature.")
    elif not can_train:
        st.info("Model comparison requires a valid classification target.")
    else:
        st.info("No numeric features available for comparison.")
    
    st.header("Feature Suitability Summary")
    st.caption(f"Analysis for **{model_name}**")
    
    feature_summary = analyze_features(df, target_col, model_name)
    
    def style_recommendation(val):
        text = str(val)
        lower = text.lower()

        if 'keep' in lower:
            return 'background-color: rgba(70, 130, 90, 0.22); color: #eaeaea; font-weight: 600;'
        if 'scale' in lower or 'regularize' in lower or 'must' in lower:
            return 'background-color: rgba(150, 130, 70, 0.22); color: #eaeaea; font-weight: 600;'
        if 'optional' in lower or 'drop' in lower:
            return 'background-color: rgba(150, 70, 70, 0.22); color: #eaeaea; font-weight: 600;'

        return 'color: #e6e6e6;'

    styled_df = feature_summary.style.map(
        style_recommendation, subset=['Recommendation']
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    with st.expander("Flag Legend"):
        st.markdown(
            """
<style>
.flag-legend { margin-top: 0.25rem; }
.flag-row { display: flex; gap: 0.6rem; align-items: baseline; margin: 0.25rem 0; }
.flag-badge {
  display: inline-block;
    width: 1.6rem;
  text-align: center;
  color: #d8d8d8;
    font-weight: 600;
}
.flag-text { color: #d8d8d8; }
.flag-note { color: #9a9a9a; margin-top: 0.5rem; }
</style>

<div class="flag-legend">
    <div class="flag-row"><span class="flag-badge">⭐</span><span class="flag-text"><b>Strong signal</b> - high correlation/importance</span></div>
    <div class="flag-row"><span class="flag-badge">⚖️</span><span class="flag-text"><b>Scale-sensitive</b> - needs scaling</span></div>
    <div class="flag-row"><span class="flag-badge">🌳</span><span class="flag-text"><b>Tree-friendly</b> - works well with trees</span></div>
    <div class="flag-row"><span class="flag-badge">📏</span><span class="flag-text"><b>Distance-sensitive</b> - affects KNN distance</span></div>
</div>
            """,
            unsafe_allow_html=True,
        )
    
    st.header("Feature Selection Summary")
    
    corr_result = plot_correlation_to_target(df, target_col)
    if corr_result:
        fig_corr, top_features = corr_result
    else:
        fig_corr, top_features = None, []
    
    scale_result = plot_feature_scale_comparison(df, target_col)
    if scale_result:
        fig_scale, dominant_features = scale_result
    else:
        fig_scale, dominant_features = None, []
    
    high_corr_pairs = get_multicollinearity_pairs(df, target_col, threshold=0.8)
    
    st.subheader("[A] Feature Importance")
    st.caption("Top = keep first, bottom = optional")
    if fig_corr:
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("No numeric features for correlation analysis.")
    
    st.subheader("[B] Scaling Requirement")
    if fig_scale:
        st.plotly_chart(fig_scale, use_container_width=True)
    else:
        st.info("No numeric features to compare scales.")
    
    st.subheader("[C] Model Fit Guidance")
    guidance = generate_model_guidance(model_name, dominant_features, high_corr_pairs)
    
    if model_name == 'Decision Tree':
        st.success(f"{model_name}: {guidance}")
    else:
        st.warning(f"{model_name}: {guidance}")
    
    if high_corr_pairs:
        st.error(f"High correlation: {high_corr_pairs[0]['pair']} (r={high_corr_pairs[0]['corr']:.2f})")
    
    with st.expander("Additional Analysis (Target, Missing Values, etc.)"):
        st.subheader("Target Distribution")
        fig_target = plot_target_distribution(df, target_col)
        if fig_target:
            st.plotly_chart(fig_target, use_container_width=True)
        
        st.subheader("Feature vs Target Separation")
        fig_separation = plot_feature_target_separation(df, target_col)
        if fig_separation:
            st.plotly_chart(fig_separation, use_container_width=True)
        else:
            fig_vs_target = plot_feature_vs_target(df, target_col)
            if fig_vs_target:
                st.plotly_chart(fig_vs_target, use_container_width=True)
        
        fig_missing = plot_missing_heatmap(df)
        if fig_missing:
            st.subheader("Missing Values")
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("No missing values in this dataset.")
        
        st.subheader("Feature Distributions")
        fig_features = plot_numeric_distributions(df, target_col)
        if fig_features:
            st.plotly_chart(fig_features, use_container_width=True)

else:
    st.warning("No data loaded.")

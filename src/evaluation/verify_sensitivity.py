"""
Sensitivity analysis verification.

Tests how model predictions change when varying individual features.
"""

import pandas as pd
import numpy as np
import sys

from src.data.data import generate_messy_data
from src.models.model_logic import get_model, train_and_evaluate
from src.models.model_verification import run_sensitivity_analysis, plot_sensitivity_chart


def verify_sensitivity():
    """Run sensitivity analysis verification tests."""
    print("Generating data...", flush=True)
    df = generate_messy_data(n_rows=200)
    target_col = 'Purchased'
    
    preprocessing_params = {
        'imputation': 'Mean',
        'encoding': 'Label Encoding', 
        'scaling': 'StandardScaler'
    }
    
    print("Training model...", flush=True)
    metrics = train_and_evaluate(df, target_col, 'Logistic Regression', preprocessing_params)
    
    print("Checking for preprocessors...", flush=True)
    if 'preprocessors' in metrics:
        print("OK: Preprocessors returned.", flush=True)
    else:
        print("FAIL: Preprocessors missing.", flush=True)
        return

    feature_to_vary = 'Age'
    
    print(f"Running sensitivity analysis on {feature_to_vary}...", flush=True)
    try:
        res = run_sensitivity_analysis(
            df, target_col, 
            metrics['trained_model'],
            metrics['preprocessors'],
            feature_to_vary
        )
        
        if res:
            print(f"Sensitivity Result keys: {list(res.keys())}", flush=True)
            print(f"Plotting for {res['feature']}...", flush=True)
            fig = plot_sensitivity_chart(res, target_col)
            print("Plot object type:", type(fig), flush=True)
            print("PASSED: Sensitivity analysis and plotting success.", flush=True)
        else:
            print("FAILED: Sensitivity analysis returned None.", flush=True)
            
    except Exception as e:
        print(f"FAILED: Exception during sensitivity analysis: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    verify_sensitivity()

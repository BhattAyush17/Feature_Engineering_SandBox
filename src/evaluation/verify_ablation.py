"""
Ablation study verification.

Tests feature importance by removing features and measuring accuracy drop.
"""

import pandas as pd
import numpy as np
import sys

from src.data.data import generate_messy_data
from src.models.model_logic import get_model, train_and_evaluate
from src.models.model_verification import run_feature_ablation, plot_ablation_chart


def verify_ablation():
    """Run ablation verification tests."""
    print("Generating data...", flush=True)
    df = generate_messy_data(n_rows=200)
    target_col = 'Purchased'
    
    preprocessing_params = {
        'imputation': 'Mean',
        'encoding': 'Label Encoding', 
        'scaling': 'StandardScaler'
    }
    
    print("Training baseline...", flush=True)
    model = get_model('Logistic Regression')
    
    feature_names = [c for c in df.columns if c != target_col]
    
    print(f"Running ablation on {len(feature_names)} features...", flush=True)
    try:
        results = run_feature_ablation(
            df, target_col, 
            model, 
            preprocessing_params,
            feature_names
        )
        
        print(f"Baseline Accuracy: {results['baseline_accuracy']:.4f}", flush=True)
        print("Ablation Results:", flush=True)
        for res in results['ablation']:
            print(f"  {res['feature']}: Drop={res['drop']:.4f}", flush=True)
            
        print("Creating plot...", flush=True)
        fig = plot_ablation_chart(results)
        print("Plot created object:", type(fig), flush=True)
        
        if results['ablation']:
            print("PASSED: Ablation ran successfully.", flush=True)
        else:
            print("FAILED: No ablation results returned.", flush=True)
            
    except Exception as e:
        print(f"FAILED: Error running ablation: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    verify_ablation()

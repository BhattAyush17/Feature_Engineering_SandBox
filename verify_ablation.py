import pandas as pd
import numpy as np
from model_verification import run_feature_ablation, plot_ablation_chart
from data import generate_messy_data
from model_logic import get_model, train_and_evaluate
import sys

def verify_ablation():
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
    
    # To match app.py flow, we pass df and let the function split/process
    # But run_feature_ablation takes (df, target_col, model_instance, params, feature_names)
    # We need to get feature names first.
    # Usually these come from a trained result in app.py.
    
    # Let's verify what run_feature_ablation expects.
    # def run_feature_ablation(df, target_col, model, preprocessing_params, feature_names, max_features=8):
    
    # We need to get valid feature names (columns of df minus target)
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
            print(f"  {res['feature']}: Drop={res['drop']:.4f} (Acc={res['accuracy']:.4f})", flush=True)
            
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

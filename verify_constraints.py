import pandas as pd
import numpy as np
from data import generate_messy_data
from model_logic import train_and_evaluate
import sys
import traceback

def verify_pipeline():
    with open("verification_log.txt", "w") as f:
        def log(msg):
            print(msg, flush=True)
            f.write(msg + "\n")
            f.flush()
            
        log("Generating data...")
        try:
            df = generate_messy_data(n_rows=100)
            target_col = 'Purchased'
            
            log("\n=== Test 1: Scaling Impact ===")
            models_affected = ['Logistic Regression', 'KNN']
            models_unaffected = ['Decision Tree']
            
            params_no_scale = {'imputation': 'Mean', 'encoding': 'Label Encoding', 'scaling': 'No Scaling'}
            params_scale = {'imputation': 'Mean', 'encoding': 'Label Encoding', 'scaling': 'StandardScaler'}
            
            for m in models_affected:
                log(f"Testing {m}...")
                res1 = train_and_evaluate(df, target_col, m, params_no_scale)
                res2 = train_and_evaluate(df, target_col, m, params_scale)
                diff = abs(res1['Accuracy'] - res2['Accuracy'])
                log(f"  {m}: Acc NoScale={res1['Accuracy']:.3f}, Acc Scale={res2['Accuracy']:.3f}, Diff={diff:.3f}")
                
            for m in models_unaffected:
                log(f"Testing {m}...")
                res1 = train_and_evaluate(df, target_col, m, params_no_scale)
                res2 = train_and_evaluate(df, target_col, m, params_scale)
                diff = abs(res1['Accuracy'] - res2['Accuracy'])
                log(f"  {m}: Acc NoScale={res1['Accuracy']:.3f}, Acc Scale={res2['Accuracy']:.3f}, Diff={diff:.3f}")
                if diff > 1e-9:
                     log(f"  WARNING: {m} should be invariant to scaling but changed!")
                else:
                     log(f"  PASSED: {m} is invariant.")

            log("\n=== Test 2: Encoding Dimensionality ===")
            params_ohe = {'imputation': 'Mean', 'encoding': 'One-Hot Encoding', 'scaling': 'StandardScaler'}
            log("Running Logistic Regression with One-Hot Encoding...")
            res_ohe = train_and_evaluate(df, target_col, 'Logistic Regression', params_ohe)
            log(f"One-Hot Encoding run successfully. Result: {res_ohe}")
            
            log("\n=== Test 3: Target Validation ===")
            
            df_cont = df.copy()
            df_cont['ContinuousTarget'] = np.random.normal(0, 1, size=len(df))
            params_basic = {'imputation': 'Mean', 'encoding': 'Label Encoding', 'scaling': 'No Scaling'}
            
            try:
                log("Testing Continuous Target (Should Fail)...")
                train_and_evaluate(df_cont, 'ContinuousTarget', 'Logistic Regression', params_basic)
                log("FAILED: Continuous target was accepted! (Should have raised ValueError)")
            except ValueError as e:
                if "continuous values" in str(e):
                    log(f"PASSED: Correctly rejected continuous target. Error: {e}")
                else:
                    log(f"WARNING: Rejected but with unexpected message: {e}")
            except Exception as e:
                log(f"FAILED: Raised wrong exception type: {type(e)}")

            df_const = df.copy()
            df_const['ConstantTarget'] = 1
            
            try:
                log("Testing Constant Target (Should Fail)...")
                train_and_evaluate(df_const, 'ConstantTarget', 'Logistic Regression', params_basic)
                log("FAILED: Constant target was accepted! (Should have raised ValueError)")
            except ValueError as e:
                if "two unique classes" in str(e):
                    log(f"PASSED: Correctly rejected constant target. Error: {e}")
                else:
                    log(f"WARNING: Rejected but with unexpected message: {e}")
            except Exception as e:
                log(f"FAILED: Raised wrong exception type: {type(e)}")

            log("\nVerification Complete.")
            
        except Exception:
            log("An error occurred:")
            log(traceback.format_exc())

if __name__ == "__main__":
    verify_pipeline()

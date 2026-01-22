import pandas as pd
from data import generate_messy_data
from model_logic import train_and_evaluate
import sys

print("Start debug", flush=True)
try:
    df = generate_messy_data(n_rows=50)
    print("Data ready", flush=True)
    params = {'imputation': 'Mean', 'encoding': 'Label Encoding', 'scaling': 'No Scaling'}
    print("Training...", flush=True)
    res = train_and_evaluate(df, 'Purchased', 'Logistic Regression', params)
    print("Done:", res, flush=True)
except Exception as e:
    print("Error:", e, flush=True)

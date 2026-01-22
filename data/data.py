import pandas as pd
import numpy as np
from io import StringIO

MAX_ROWS_VISUALIZATION = 5000
MAX_ROWS_TRAINING = 20000
ROW_COUNT_THRESHOLD_MB = 50
HIGH_CARDINALITY_THRESHOLD = 50


def get_file_size_mb(file):
    """File size in MB."""
    file.seek(0, 2)
    size_bytes = file.tell()
    file.seek(0)
    return size_bytes / (1024 * 1024)


def count_rows_fast(file):
    """Count rows without full load."""
    file.seek(0)
    row_count = sum(1 for _ in file) - 1
    file.seek(0)
    return row_count


def load_data_with_metadata(file, max_rows=None):
    """Load CSV with metadata. Returns (df, metadata) or (None, error_dict)."""
    if file is None:
        return None, {'error': 'No file provided'}
    
    try:
        # Get file size
        file_size_mb = get_file_size_mb(file)

        # Skip row count on huge files
        total_rows = None
        if file_size_mb <= ROW_COUNT_THRESHOLD_MB:
            total_rows = count_rows_fast(file)
        
        file.seek(0)

        if max_rows is not None and (total_rows is None or total_rows > max_rows):
            df = pd.read_csv(file, nrows=max_rows)
        else:
            df = pd.read_csv(file)
        
        if total_rows is None:
            is_sampled = max_rows is not None and len(df) == max_rows
        else:
            is_sampled = len(df) < total_rows

        metadata = {
            'total_rows': total_rows,
            'loaded_rows': len(df),
            'columns': len(df.columns),
            'file_size_mb': file_size_mb,
            'is_sampled': is_sampled
        }
        
        return df, metadata
        
    except Exception as e:
        return None, {'error': str(e)}


def get_schema_summary(df):
    """Get schema info: numeric, categorical, high-cardinality columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    high_cardinality = []
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique > HIGH_CARDINALITY_THRESHOLD:
            high_cardinality.append({'column': col, 'unique': n_unique})
    
    missing_summary = {}
    for col in df.columns:
        missing_pct = df[col].isna().mean() * 100
        if missing_pct > 0:
            missing_summary[col] = round(missing_pct, 1)
    
    return {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'high_cardinality': high_cardinality,
        'missing_summary': missing_summary,
        'n_numeric': len(numeric_cols),
        'n_categorical': len(categorical_cols)
    }


def sample_data(df, max_rows, target_col=None, strategy='stratified'):
    """Sample data, optionally stratified."""
    if len(df) <= max_rows:
        return df
    
    if strategy == 'stratified' and target_col and target_col in df.columns:
        # Check if target is suitable for stratification
        target = df[target_col]
        n_unique = target.nunique()
        
        if n_unique <= 20 and n_unique >= 2:
            try:
                from sklearn.model_selection import train_test_split
                frac = max_rows / len(df)
                _, sampled = train_test_split(
                    df, 
                    test_size=frac, 
                    stratify=target,
                    random_state=42
                )
                return sampled.reset_index(drop=True)
            except:
                pass
    
    return df.sample(n=max_rows, random_state=42).reset_index(drop=True)


def get_visualization_sample(df, target_col=None):
    """Smaller sample for plots."""
    return sample_data(df, MAX_ROWS_VISUALIZATION, target_col, strategy='stratified')


def get_training_sample(df, target_col=None):
    """Larger sample for training."""
    return sample_data(df, MAX_ROWS_TRAINING, target_col, strategy='stratified')


def load_data(file):
    """Simple loader for backward compatibility."""
    if file is not None:
        try:
            return pd.read_csv(file)
        except Exception:
            return None
    return None


def get_column_types(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_cols, categorical_cols


def generate_messy_data(n_rows=500):
    np.random.seed(42)
    
    age = np.random.normal(loc=35, scale=10, size=n_rows).astype(int)
    income = np.random.normal(loc=50000, scale=15000, size=n_rows)
    credit_score = np.random.randint(300, 850, size=n_rows)
    years_employed = np.random.exponential(scale=5, size=n_rows)
    
    education = np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], size=n_rows, p=[0.3, 0.4, 0.2, 0.1])
    city = np.random.choice(['New York', 'London', 'Paris', 'Tokyo', 'Mumbai'], size=n_rows)
    gender = np.random.choice(['Male', 'Female', 'Non-Binary'], size=n_rows)
    
    # Balance target classes roughly 50/50
    prob = (income / 100000) + (age / 100) + np.random.normal(0, 0.1, n_rows)
    target = (prob > prob.mean()).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'CreditScore': credit_score,
        'YearsEmployed': years_employed,
        'Education': education,
        'City': city,
        'Gender': gender,
        'Purchased': target
    })
    
    mask_income = np.random.choice([True, False], size=n_rows, p=[0.1, 0.9])
    df.loc[mask_income, 'Income'] = np.nan
    
    mask_edu = np.random.choice([True, False], size=n_rows, p=[0.05, 0.95])
    df.loc[mask_edu, 'Education'] = np.nan
    
    return df

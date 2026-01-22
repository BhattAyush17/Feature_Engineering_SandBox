from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.multiclass import type_of_target
import pandas as pd
from preprocessing import impute_data, encode_data, scale_data
from data import get_column_types


def detect_target_type(y):
    """Detect if target is classification or regression."""
    try:
        y_series = pd.Series(y)
    except Exception:
        y_series = y

    y_clean = y_series.dropna() if hasattr(y_series, 'dropna') else y_series
    n_unique = int(y_clean.nunique()) if hasattr(y_clean, 'nunique') else 0

    if y_clean is None or (hasattr(y_clean, '__len__') and len(y_clean) == 0):
        return {
            'sklearn_type': 'unknown',
            'task_type': 'unknown',
            'n_unique': 0,
            'compatible_models': [],
            'is_valid_for_training': False,
        }

    try:
        target_type = type_of_target(y_clean)
    except Exception:
        y_as_str = y_clean.astype(str)
        target_type = type_of_target(y_as_str)
    
    classification_types = ['binary', 'multiclass', 'multiclass-multioutput']
    regression_types = ['continuous', 'continuous-multioutput']
    
    if target_type in classification_types:
        task_type = 'classification'
        compatible_models = ['Logistic Regression', 'KNN', 'SVM', 'Decision Tree']
    elif target_type in regression_types:
        task_type = 'regression'
        compatible_models = []  # No regression models available yet
    else:
        task_type = 'unknown'
        compatible_models = []
    
    return {
        'sklearn_type': target_type,
        'task_type': task_type,
        'n_unique': n_unique,
        'compatible_models': compatible_models,
        'is_valid_for_training': task_type == 'classification' and n_unique >= 2
    }


def check_training_compatibility(target_info, model_name):
    """Check if model is compatible with target."""
    if target_info['task_type'] == 'regression':
        return False, (
            f"Skipped: Target is continuous ({target_info['n_unique']} unique values). "
            f"'{model_name}' needs discrete class labels."
        )
    
    if target_info['n_unique'] < 2:
        return False, (
            f"Skipped: Target has only {target_info['n_unique']} unique value(s). "
            f"Cannot train on constant data."
        )
    
    if model_name not in target_info['compatible_models']:
        return False, (
            f"Skipped: '{model_name}' is not compatible with this target type."
        )
    
    return True, None


def get_model(model_name):
    if model_name == 'Logistic Regression':
        return LogisticRegression(max_iter=1000)
    elif model_name == 'KNN':
        return KNeighborsClassifier()
    elif model_name == 'SVM':
        return SVC()
    elif model_name == 'Decision Tree':
        return DecisionTreeClassifier()
    return LogisticRegression()

def train_and_evaluate(df, target_col, model_name, preprocessing_params, test_size=0.2):
    """Train and evaluate a classification model."""
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Coerce mixed types to string
    try:
        y_types = pd.Series(y).dropna().map(type).nunique()
        if y_types and y_types > 1:
            y = y.astype(str)
    except Exception:
        pass
    
    target_type = type_of_target(y)
    if target_type in ['continuous', 'continuous-multioutput']:
        raise ValueError("Target is continuous. Run check_training_compatibility() first.")
    
    if y.nunique() < 2:
        raise ValueError("Target needs at least 2 classes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    numeric_cols, categorical_cols = get_column_types(X_train)
    
    strat_imp = preprocessing_params.get('imputation', 'Drop Rows')
    
    X_train, imputer = impute_data(X_train, strat_imp, numeric_cols)
    # Align y_train in case rows were dropped
    y_train = y_train.loc[X_train.index]
    
    # Transform test
    X_test, _ = impute_data(X_test, strat_imp, numeric_cols, imputer=imputer)
    # Align y_test
    y_test = y_test.loc[X_test.index]
    
    # 2. Encoding
    strat_enc = preprocessing_params.get('encoding', 'Label Encoding')
    X_train, encoder = encode_data(X_train, strat_enc, categorical_cols)
    X_test, _ = encode_data(X_test, strat_enc, categorical_cols, encoder=encoder)
    
    numeric_cols, _ = get_column_types(X_train)
    strat_scale = preprocessing_params.get('scaling', 'No Scaling')
    
    X_train, scaler = scale_data(X_train, strat_scale, numeric_cols)
    X_test, _ = scale_data(X_test, strat_scale, numeric_cols, scaler=scaler)
    
    model = get_model(model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    feature_names = list(X_train.columns)
    
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'Model': model_name,
        'trained_model': model,
        'feature_names': feature_names,
        'preprocessors': {
            'imputer': imputer,
            'encoder': encoder,
            'scaler': scaler
        }
    }

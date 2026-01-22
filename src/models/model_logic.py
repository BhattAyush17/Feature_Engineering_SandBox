"""
Model logic layer.

Handles model initialization, training, prediction, and evaluation.
No Streamlit imports.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.multiclass import type_of_target
import pandas as pd

from src.preprocessing.preprocessing import impute_data, encode_data, scale_data
from src.data.data import get_column_types


def detect_target_type(y):
    """
    Detect if target is classification or regression.
    
    Args:
        y: Target series or array
        
    Returns:
        dict: Target type information
    """
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
        compatible_models = []
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
    """
    Check if model is compatible with target.
    
    Args:
        target_info: Dict from detect_target_type
        model_name: Name of model to check
        
    Returns:
        tuple: (is_compatible, warning_message)
    """
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
    """
    Get model instance by name.
    
    Args:
        model_name: 'Logistic Regression', 'KNN', 'SVM', or 'Decision Tree'
        
    Returns:
        sklearn model instance
    """
    if model_name == 'Logistic Regression':
        return LogisticRegression(max_iter=1000)
    elif model_name == 'KNN':
        return KNeighborsClassifier()
    elif model_name == 'SVM':
        return SVC()
    elif model_name == 'Decision Tree':
        return DecisionTreeClassifier()
    return LogisticRegression()


def build_model(model_name, **kwargs):
    """
    Build model with custom parameters.
    
    Args:
        model_name: Model type name
        **kwargs: Model-specific parameters
        
    Returns:
        sklearn model instance
    """
    if model_name == 'Logistic Regression':
        return LogisticRegression(max_iter=kwargs.get('max_iter', 1000), **kwargs)
    elif model_name == 'KNN':
        return KNeighborsClassifier(
            n_neighbors=kwargs.get('n_neighbors', 5),
            **{k: v for k, v in kwargs.items() if k != 'n_neighbors'}
        )
    elif model_name == 'SVM':
        return SVC(**kwargs)
    elif model_name == 'Decision Tree':
        return DecisionTreeClassifier(**kwargs)
    return LogisticRegression(max_iter=1000)


def train_model(X, y, model):
    """
    Train a model on data.
    
    Args:
        X: Feature matrix
        y: Target vector
        model: sklearn model instance
        
    Returns:
        Trained model
    """
    model.fit(X, y)
    return model


def predict(model, X):
    """
    Make predictions with a trained model.
    
    Args:
        model: Trained sklearn model
        X: Feature matrix
        
    Returns:
        array: Predictions
    """
    return model.predict(X)


def train_and_evaluate(df, target_col, model_name, preprocessing_params, test_size=0.2):
    """
    Train and evaluate a classification model.
    
    Args:
        df: DataFrame with features and target
        target_col: Target column name
        model_name: Model to train
        preprocessing_params: Dict with 'imputation', 'encoding', 'scaling' keys
        test_size: Test split ratio
        
    Returns:
        dict: Metrics and trained model info
        
    Raises:
        ValueError: If target is invalid for classification
    """
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
        raise ValueError(
            "The selected target column contains continuous values. "
            "Classification models require discrete class labels."
        )
    
    if y.nunique() < 2:
        raise ValueError(
            "Target column must have at least two unique classes. "
            "Cannot train a model on constant or single-class data."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    numeric_cols, categorical_cols = get_column_types(X_train)
    
    # 1. Imputation
    strat_imp = preprocessing_params.get('imputation', 'Drop Rows')
    X_train, imputer = impute_data(X_train, strat_imp, numeric_cols)
    y_train = y_train.loc[X_train.index]
    X_test, _ = impute_data(X_test, strat_imp, numeric_cols, imputer=imputer)
    y_test = y_test.loc[X_test.index]
    
    # 2. Encoding
    strat_enc = preprocessing_params.get('encoding', 'Label Encoding')
    X_train, encoder = encode_data(X_train, strat_enc, categorical_cols)
    X_test, _ = encode_data(X_test, strat_enc, categorical_cols, encoder=encoder)
    
    # 3. Scaling
    numeric_cols, _ = get_column_types(X_train)
    strat_scale = preprocessing_params.get('scaling', 'No Scaling')
    X_train, scaler = scale_data(X_train, strat_scale, numeric_cols)
    X_test, _ = scale_data(X_test, strat_scale, numeric_cols, scaler=scaler)
    
    # Train
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

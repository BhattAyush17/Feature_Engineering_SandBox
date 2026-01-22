"""
Preprocessing layer.

Handles missing value imputation, encoding, scaling, and type casting.
Accepts raw DataFrames and returns transformed DataFrames.
No model logic or Streamlit imports.
"""

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder, 
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler, 
    OneHotEncoder
)
import pandas as pd
import numpy as np


def impute_data(df, strategy, numeric_cols, constant_value=0, imputer=None):
    """
    Impute missing values in numeric columns.
    
    Args:
        df: DataFrame to impute
        strategy: 'Drop Rows', 'Mean', 'Median', 'Most Frequent', or 'Constant'
        numeric_cols: List of numeric columns to impute
        constant_value: Value to use for 'Constant' strategy
        imputer: Pre-fitted imputer (for test data)
        
    Returns:
        tuple: (imputed_df, fitted_imputer)
    """
    df_imputed = df.copy()
    
    if strategy == 'Drop Rows':
        return df_imputed.dropna(subset=numeric_cols), None
    
    if not numeric_cols:
        return df_imputed, None
    
    if imputer is None:
        if strategy == 'Constant':
            imputer = SimpleImputer(strategy='constant', fill_value=constant_value)
        elif strategy == 'Mean':
            imputer = SimpleImputer(strategy='mean')
        elif strategy == 'Median':
            imputer = SimpleImputer(strategy='median')
        elif strategy == 'Most Frequent':
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            return df, None
        
        imputer.fit(df_imputed[numeric_cols])
        
    if imputer:
        df_imputed[numeric_cols] = imputer.transform(df_imputed[numeric_cols])
        
    return df_imputed, imputer


def encode_data(df, strategy, categorical_cols, encoder=None):
    """
    Encode categorical columns.
    
    Args:
        df: DataFrame to encode
        strategy: 'Label Encoding', 'One-Hot Encoding', or 'Frequency Encoding'
        categorical_cols: List of categorical columns to encode
        encoder: Pre-fitted encoder (for test data)
        
    Returns:
        tuple: (encoded_df, fitted_encoder)
    """
    df_encoded = df.copy()
    
    if not categorical_cols:
        return df_encoded, None
        
    if strategy == 'Label Encoding':
        if encoder is None:
            encoder = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = df_encoded[col].astype(str)
                le.fit(df_encoded[col])
                encoder[col] = le
        
        for col in categorical_cols:
            df_encoded[col] = df_encoded[col].astype(str)
            df_encoded[col] = encoder[col].transform(df_encoded[col])
            
    elif strategy == 'One-Hot Encoding':
        if encoder is None:
            try:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            except TypeError:
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
            encoder.fit(df_encoded[categorical_cols])
            
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        encoded_data = encoder.transform(df_encoded[categorical_cols])
        
        df_encoded_ohe = pd.DataFrame(
            encoded_data, 
            columns=encoded_cols, 
            index=df_encoded.index
        )
        df_encoded = df_encoded.drop(columns=categorical_cols)
        df_encoded = pd.concat([df_encoded, df_encoded_ohe], axis=1)
        
    elif strategy == 'Frequency Encoding':
        if encoder is None:
            encoder = {}
            for col in categorical_cols:
                encoder[col] = df_encoded[col].value_counts(normalize=True).to_dict()
                
        for col in categorical_cols:
            df_encoded[col] = df_encoded[col].map(encoder[col]).fillna(0)
            
    return df_encoded, encoder


def scale_data(df, strategy, numeric_cols, scaler=None):
    """
    Scale numeric columns.
    
    Args:
        df: DataFrame to scale
        strategy: 'No Scaling', 'StandardScaler', 'MinMaxScaler', or 'RobustScaler'
        numeric_cols: List of numeric columns to scale
        scaler: Pre-fitted scaler (for test data)
        
    Returns:
        tuple: (scaled_df, fitted_scaler)
    """
    df_scaled = df.copy()
    
    if not numeric_cols:
        return df_scaled, None
        
    if strategy == 'No Scaling':
        return df_scaled, None
    
    if scaler is None:
        if strategy == 'StandardScaler':
            scaler = StandardScaler()
        elif strategy == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif strategy == 'RobustScaler':
            scaler = RobustScaler()
        else:
            return df_scaled, None
        
        scaler.fit(df_scaled[numeric_cols])
        
    if scaler:
        df_scaled[numeric_cols] = scaler.transform(df_scaled[numeric_cols])
        
    return df_scaled, scaler


def handle_outliers(df, numeric_cols, method='clip', threshold=3):
    """
    Handle outliers in numeric columns.
    
    Args:
        df: DataFrame to process
        numeric_cols: List of numeric columns
        method: 'clip' (cap at threshold), 'remove' (drop rows), or 'none'
        threshold: Number of standard deviations for outlier detection
        
    Returns:
        DataFrame: Processed data
    """
    df_processed = df.copy()
    
    if method == 'none' or not numeric_cols:
        return df_processed
    
    for col in numeric_cols:
        mean = df_processed[col].mean()
        std = df_processed[col].std()
        lower = mean - threshold * std
        upper = mean + threshold * std
        
        if method == 'clip':
            df_processed[col] = df_processed[col].clip(lower=lower, upper=upper)
        elif method == 'remove':
            mask = (df_processed[col] >= lower) & (df_processed[col] <= upper)
            df_processed = df_processed[mask]
    
    return df_processed


def cast_types(df, type_mapping):
    """
    Cast columns to specified types.
    
    Args:
        df: DataFrame to process
        type_mapping: Dict of column_name -> dtype
        
    Returns:
        DataFrame: Processed data
    """
    df_casted = df.copy()
    
    for col, dtype in type_mapping.items():
        if col in df_casted.columns:
            try:
                df_casted[col] = df_casted[col].astype(dtype)
            except (ValueError, TypeError):
                pass
    
    return df_casted

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
import pandas as pd
import numpy as np

def impute_data(df, strategy, numeric_cols, constant_value=0, imputer=None):
    df_imputed = df.copy()
    if strategy == 'Drop Rows':
        return df_imputed.dropna(subset=numeric_cols), None
    
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
        
        df_encoded_ohe = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_encoded.index)
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

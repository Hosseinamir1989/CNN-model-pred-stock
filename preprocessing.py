import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

def normalize_features(df, target_col="Close"):
    feature_cols = [c for c in df.columns if c != target_col]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df[feature_cols])
    y_scaled = scaler_y.fit_transform(df[[target_col]])
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    df_scaled[target_col] = y_scaled
    return df_scaled, scaler_X, scaler_y

def df_to_windowed_df(df, window_size=5, target_col="Close"):
    feature_cols = [col for col in df.columns if col != target_col]
    windowed_data, target_dates = [], []
    for i in range(window_size, len(df)):
        row = []
        for col in feature_cols:
            row.extend(df[col].iloc[i - window_size:i].values)
        row.append(df[target_col].iloc[i])
        target_dates.append(df.index[i].strftime('%Y-%m-%d'))
        windowed_data.append(row)
    columns = ["Target Date"] + [f"{col}_t-{j}" for col in feature_cols for j in range(window_size, 0, -1)] + ["Target"]
    windowed_df = pd.DataFrame(windowed_data, columns=columns[1:])
    windowed_df.insert(0, "Target Date", target_dates)
    logging.info(f"Windowed DataFrame shape: {windowed_df.shape}")
    return windowed_df

def windowed_df_to_date_X_y(windowed_df, window_size):
    dates = windowed_df["Target Date"].values
    X = windowed_df.drop(columns=["Target Date", "Target"]).values
    y = windowed_df["Target"].values
    n_features = int(X.shape[1] / window_size)
    X = X.reshape((len(dates), window_size, n_features))
    return dates, X.astype(np.float32), y.astype(np.float32)

import pandas as pd
import logging

def add_indicators(df):
    logging.info("Adding technical indicators")
    df = df.copy()

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # RSI (with small epsilon to avoid div/0)
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    ma20 = df["Close"].rolling(window=20).mean()
    std20 = df["Close"].rolling(window=20).std()
    df["BB_upper"] = ma20 + 2 * std20
    df["BB_lower"] = ma20 - 2 * std20

    # Drop initial NaNs
    df.dropna(inplace=True)

    return df

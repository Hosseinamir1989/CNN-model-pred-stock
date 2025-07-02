import yfinance as yf
import pandas as pd
import pytz
from datetime import datetime
import time

def download_stock_data(ticker: str, start_date: str, end_date: str, timezone: str = "UTC", retries: int = 3, wait_time: int = 5):
    """
    Downloads historical stock data for a given ticker and time range,
    retries on failure, and converts to the specified timezone.

    Parameters:
    - ticker (str): The stock ticker (e.g., 'SAP.DE', 'AAPL')
    - start_date (str): Start date in 'YYYY-MM-DD' format
    - end_date (str): End date in 'YYYY-MM-DD' format
    - timezone (str): Timezone to convert datetime index (e.g., 'Europe/Vienna')
    - retries (int): Number of retry attempts on failure
    - wait_time (int): Wait time between retries in seconds

    Returns:
    - pd.DataFrame or None
    """
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt+1}: Downloading {ticker} from {start_date} to {end_date}...")
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False
            )

            if df.empty:
                print(f"No data returned for ticker: {ticker}. Please verify the symbol and dates.")
                return None

            # Convert index to datetime with timezone
            df.index = pd.to_datetime(df.index).tz_localize("UTC").tz_convert(timezone)
            print(f"Successfully downloaded data for {ticker} in timezone: {timezone}")
            return df

        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(wait_time)

    print(f"All {retries} attempts failed for {ticker}.")
    return None

# Example usage
if __name__ == "__main__":
    ticker_symbol = "SAP.DE"                # Change to your desired ticker
    timezone = "Europe/Vienna"              # Change to desired timezone
    start = "2020-01-01"
    end = "2025-01-01"

    stock_df = download_stock_data(ticker_symbol, start, end, timezone)

    if stock_df is not None:
        filename = f"{ticker_symbol.replace('.', '_')}_2020_2025_data.csv"
        stock_df.to_csv(filename)
        print(f"✅ Data saved to: {filename}")

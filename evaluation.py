import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import os
def plot_predictions(dates_train, y_train, train_pred,
                     dates_val, y_val, val_pred,
                     dates_test, y_test, test_pred):
    plt.figure(figsize=(14,6))
    plt.plot(dates_train, y_train, label='Train Actual')
    plt.plot(dates_train, train_pred, label='Train Predicted')
    plt.plot(dates_val, y_val, label='Validation Actual')
    plt.plot(dates_val, val_pred, label='Validation Predicted')
    plt.plot(dates_test, y_test, label='Test Actual')
    plt.plot(dates_test, test_pred, label='Test Predicted')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('LSTM Predictions')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.show()







def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    logging.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae

def compute_directional_accuracy(y_true, y_pred):
    """
    Computes the directional accuracy of predictions: 
    how often the model predicts the correct direction of change.
    """
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    correct_direction = np.sign(y_true_diff) == np.sign(y_pred_diff)
    dir_acc = np.mean(correct_direction)
    logging.info(f"Directional Accuracy: {dir_acc:.4f}")
    return dir_acc


def evaluate_multi_horizon_accuracy(y_true, y_pred, horizon_days=[1, 2, 3]):
    """
    Evaluate RMSE and MAE at multiple horizons (T+1, T+2, T+3).
    
    y_true: actual prices (1D numpy array)
    y_pred: predicted prices (1D numpy array) — usually T+1 predictions
    horizon_days: list of horizons in days
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    results = []

    for horizon in horizon_days:
        # Ensure we don't exceed index boundaries
        if len(y_true) <= horizon or len(y_pred) <= horizon:
            continue

        shifted_true = y_true[horizon:]
        valid_pred = y_pred[:-horizon]

        rmse = np.sqrt(mean_squared_error(shifted_true, valid_pred))
        mae = mean_absolute_error(shifted_true, valid_pred)

        results.append({
            "Horizon": f"T+{horizon}",
            "RMSE": rmse,
            "MAE": mae
        })

    return results


def evaluate_directional_accuracy_by_horizon(y_true, y_pred, horizon_days=[1, 2, 3]):
    """s
    Evaluate directional accuracy for multiple horizons (T+1, T+2, T+3).
    """
    results = []

    for horizon in horizon_days:
        if len(y_true) <= horizon or len(y_pred) <= horizon:
            continue

        shifted_true = y_true[horizon:]
        valid_pred = y_pred[:-horizon]

        # True direction: compare horizon-day forward difference with today
        true_direction = np.sign(shifted_true - y_true[:-horizon])
        pred_direction = np.sign(valid_pred - y_true[:-horizon])

        dir_acc = np.mean(true_direction == pred_direction)
        results.append({
            "Horizon": f"T+{horizon}",
            "Directional Accuracy": dir_acc
        })

    return results

import pandas as pd

def simulate_pnl(y_true, y_pred, initial_balance=100.0, threshold=0.005, ticker="Unknown", output_folder="output"):
    """
    Simulate PnL based on predicted signals and true prices using a threshold-based strategy.
    
    threshold: minimum % change in predicted price relative to today to trigger a buy/sell.
    """
    balance = initial_balance
    position = 0  # 0 = no position, 1 = long position
    balance_history = [balance]
    trade_log = []

    for i in range(len(y_true) - 1):
        actual_price_today = y_true[i]
        predicted_price_next = y_pred[i + 1]
        predicted_change_pct = (predicted_price_next - actual_price_today) / actual_price_today

        if predicted_change_pct > threshold and position == 0:
            # Buy
            position = 1
            entry_price = actual_price_today
            trade_log.append({
                "Day": i,
                "Action": "BUY",
                "Price": entry_price,
                "Balance": balance
            })
        elif predicted_change_pct < -threshold and position == 1:
            # Sell
            exit_price = actual_price_today
            profit = exit_price - entry_price
            balance += profit
            position = 0
            trade_log.append({
                "Day": i,
                "Action": "SELL",
                "Price": exit_price,
                "Profit/Loss": profit,
                "Balance": balance
            })
        else:
            # Hold
            trade_log.append({
                "Day": i,
                "Action": "HOLD",
                "Price": actual_price_today,
                "Balance": balance
            })

        balance_history.append(balance)

    # Close any open position at the end
    if position == 1:
        exit_price = y_true[-1]
        profit = exit_price - entry_price
        balance += profit
        trade_log.append({
            "Day": len(y_true)-1,
            "Action": "SELL (End)",
            "Price": exit_price,
            "Profit/Loss": profit,
            "Balance": balance
        })
        balance_history.append(balance)

    final_balance = balance

    # Save trade log as CSV
    trade_log_df = pd.DataFrame(trade_log)
    os.makedirs(output_folder, exist_ok=True)
    trade_log_path = os.path.join(output_folder, f"trade_log_{ticker}.csv")
    trade_log_df.to_csv(trade_log_path, index=False)
    logging.info(f"Trade log saved to {trade_log_path}")

    return balance_history, trade_log, final_balance


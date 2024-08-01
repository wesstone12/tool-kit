import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np
from mlforecast.utils import PredictionIntervals
from tabulate import tabulate
from matplotlib.animation import ArtistAnimation
import traceback

def fetch_stock_data(ticker_symbol, days=3):
    ny_tz = pytz.timezone('America/New_York')
    end_date = datetime.now(ny_tz).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)

    print(f"Fetching data for {ticker_symbol} from {start_date} to {end_date}\n")
    
    stock_data = yf.Ticker(ticker_symbol)
    historical_data = stock_data.history(start=start_date, end=end_date, interval='1m')['Close']
    
    df = pd.DataFrame({
        'unique_id': [ticker_symbol] * len(historical_data),
        'ds': historical_data.index.astype(int) // 10**9,
        'y': historical_data.values
    })
    return df

def create_model():
    return MLForecast(
        models=[LGBMRegressor(min_child_samples=5, min_data_in_leaf=5, verbose=-1)],
        freq="min",
        lags=[1, 2, 3, 4, 5, 15, 30, 60, 120],
        target_transforms=[Differences([1])]
    )

def execute_trade(current_price, lower_bound, upper_bound, next_pred, cash, shares, threshold):
    
    
    if current_price < lower_bound and next_pred > current_price :
        # Price is higher than expected, but predicted to go even higher
        if cash > current_price:
            shares_to_buy = cash // current_price
            cash -= shares_to_buy * current_price
            shares += shares_to_buy
            return cash, shares, "BUY"
    elif current_price > upper_bound and next_pred < current_price :
            # Price is lower than expected, but predicted to go even lower
        if shares > 0:
            cash += current_price * shares
            shares = 0
            return cash, shares, "SELL"
    
    return cash, shares, "HOLD"

def calculate_portfolio_value(cash, shares, current_price):
    return cash + (shares * current_price)

def main():
    ticker_symbol = 'AAPL'
    initial_cash = 10000  
    level = 60
    threshold = 0.1
    
    try:
        cash = initial_cash
        shares = 0
        portfolio_values = []
        trade_signals = []

        df = fetch_stock_data(ticker_symbol, days=3)
        
        df['ds'] = pd.to_datetime(df['ds'], unit='s')
        grouped = df.groupby(df['ds'].dt.date)
        
        days = list(grouped.groups.keys())
        if len(days) < 2:
            raise ValueError("Not enough days of data available")
        
        day_t_1 = grouped.get_group(days[-2])
        # day_t_2 = grouped.get_group(days[-3])
        day_t = grouped.get_group(days[-1])
        
        print(f"Total data points: {len(df)}")
        print(f"Day t-1 data points: {len(day_t_1)}")
        print(f"Day t data points: {len(day_t)}")
        print(day_t_1.head())
        print(day_t.tail())
        
        plt.figure(figsize=(15, 7))
        plt.plot(day_t_1['ds'], day_t_1['y'], label='Day t-1 Data')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(f'{ticker_symbol} Stock Price: Day t-1 Data')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{ticker_symbol}_day_t_1_plot.png')
        plt.close()
        print(f"Day t-1 data plot saved as {ticker_symbol}_day_t_1_plot.png")
        
        mlf = create_model()
        # mlf.fit(day_t_1, prediction_intervals=PredictionIntervals(n_windows=10, h = 1), as_numpy=True)
        mlf.fit(day_t_1, prediction_intervals=PredictionIntervals(n_windows=10, h = 5), as_numpy=True)
        
        predictions = []
        lows = []
        highs = []
        actuals = []
        anomalies = []
        anomaly_times = []
        
        for i in range(len(day_t)):
            actual_value = day_t.iloc[i]['y']
            actuals.append(actual_value)

            if i == 0:
                # For the first iteration, we don't have a previous prediction
                pred = mlf.predict(1, level=[level])
                pred_value = pred[f'LGBMRegressor'].values[0]
                lower_bound = pred[f'LGBMRegressor-lo-{level}'].values[0]
                upper_bound = pred[f'LGBMRegressor-hi-{level}'].values[0]
                previous_price = actual_value
            else:
                # Use the prediction from the previous iteration
                pred_value = next_pred_value
                lower_bound = next_lower_bound
                upper_bound = next_upper_bound
                previous_price = actuals[-2]  # The previous actual price

            predictions.append(pred_value)
            lows.append(lower_bound)
            highs.append(upper_bound)

            is_anomaly = actual_value < lower_bound or actual_value > upper_bound



            # Update the model with the new data point
            new_data = pd.DataFrame({
                'unique_id': [day_t.iloc[i]['unique_id']],
                'ds': [day_t.iloc[i]['ds']],
                'y': [actual_value]
            })
            mlf.update(new_data)

            # Get the prediction for the next time step
            next_pred = mlf.predict(1, level=[level])
            #few steps ahead lol
            new_pred_few = mlf.predict(5, level=[level])
            new_pred_few_value = new_pred_few[f'LGBMRegressor'].values[0]
            new_pred_few_lower_bound = new_pred_few[f'LGBMRegressor-lo-{level}'].values[0]
            new_pred_few_upper_bound = new_pred_few[f'LGBMRegressor-hi-{level}'].values[0]

            next_pred_value = next_pred['LGBMRegressor'].values[0]
            next_lower_bound = next_pred[f'LGBMRegressor-lo-{level}'].values[0]
            next_upper_bound = next_pred[f'LGBMRegressor-hi-{level}'].values[0]

                        # Make trading decision based on current information
            if i > 10:             
                # cash, shares, signal = execute_trade(actual_value,  lower_bound, upper_bound, next_pred_value, cash, shares, threshold)
                cash, shares, signal = execute_trade(actual_value,  new_pred_few_lower_bound, new_pred_few_upper_bound, new_pred_few_value, cash, shares, threshold)
            else:
                signal = "HOLD"
            
            portfolio_value = calculate_portfolio_value(cash, shares, actual_value)
            portfolio_values.append(portfolio_value)
            trade_signals.append(signal)

            row = [
                f"{day_t.iloc[i]['ds'].strftime('%Y-%m-%d %H:%M:%S')}",
                f"{pred_value:.2f}",
                f"{actual_value:.2f}",
                f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                f"{next_pred_value:.2f}",
                signal,
                f"${portfolio_value:.2f}"
            ]


            print(tabulate([row], headers=["Timestamp", "Predicted", "Actual", f"{level}% Interval", "Next Pred", "Signal", "Portfolio Value"], tablefmt="grid"))
        
        initial_portfolio_value = initial_cash
        final_portfolio_value = portfolio_values[-1]
        total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100

        print(f"Initial Portfolio Value: ${initial_portfolio_value:.2f}")
        print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
        print(f"Total Return: {total_return:.2f}%")

        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
        print(f"Mean Absolute Error: {mae}")
        print(f"Root Mean Square Error: {rmse}")

        times = day_t['ds']


        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14), sharex=True)
        ax1.set_xlim(times.min(), times.max())
        ax1.set_ylim(min(min(actuals), min(lows)), max(max(actuals), max(highs)))
        ax1.set_ylabel('Price')
        ax1.set_title(f'{ticker_symbol} Stock Price: Day t Simulation (Actual vs Predicted)')

        ax2.set_xlim(times.min(), times.max())
        ax2.set_ylim(min(portfolio_values), max(portfolio_values))
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.set_title('Portfolio Value Over Time')

        plt.xticks(rotation=45)

        print(f"Total trades: {len([s for s in trade_signals if s != 'HOLD'])}")
        print(f"Buy signals: {trade_signals.count('BUY')}")
        print(f"Sell signals: {trade_signals.count('SELL')}")
        
        print("Creating animation...")
        artists = []
        for i in range(1, len(times) + 1):
            try:
                actual_line, = ax1.plot(times.iloc[:i], actuals[:i], color='blue', label='Actual' if i == 1 else "")
                predicted_line, = ax1.plot(times.iloc[:i], predictions[:i], color='orange', label='Predicted' if i == 1 else "")
                fill = ax1.fill_between(times.iloc[:i], lows[:i], highs[:i], color='blue', alpha=0.3, label=f'{level}% Prediction Interval' if i == 1 else "")

                portfolio_line, = ax2.plot(times.iloc[:i], portfolio_values[:i], color='green', label='Portfolio Value' if i == 1 else "")

                buy_signals = [j for j in range(i) if trade_signals[j] == "BUY"]
                sell_signals = [j for j in range(i) if trade_signals[j] == "SELL"]

                buy_scatter = ax1.scatter([times.iloc[j] for j in buy_signals], 
                                        [actuals[j] for j in buy_signals], 
                                        color='green', s=100, marker='^', label='Buy Signal' if i == len(times) else "")
                
                sell_scatter = ax1.scatter([times.iloc[j] for j in sell_signals], 
                                        [actuals[j] for j in sell_signals], 
                                        color='red', s=100, marker='v', label='Sell Signal' if i == len(times) else "")

                artists.append([actual_line, predicted_line, fill, portfolio_line, buy_scatter, sell_scatter])

            except Exception as e:
                print(f"Error in animation loop at iteration {i}: {str(e)}")
                print(traceback.format_exc())
                raise

        print("Animation frames created. Adding legend...")
        ax1.legend()
        ax2.legend()

        print("Creating ArtistAnimation...")
        anim = ArtistAnimation(fig, artists, interval=50, blit=True, repeat_delay=1000)

        print("Saving animation...")
        anim.save(f'{ticker_symbol}_prediction_animation_with_trades.gif', writer='pillow')
        print(f"Animation saved as {ticker_symbol}_prediction_animation_with_trades.gif")

        plt.close()
            
    except Exception as e:
         print(f"An error occurred in main: {e}")
         print(traceback.format_exc())

if __name__ == "__main__":
    main()
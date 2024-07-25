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
import tabulate
def fetch_stock_data(ticker_symbol, days=2):
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
        lags=[1, 2, 3, 4, 5, 15, 30, 60],
        target_transforms=[Differences([1])]
    )

def main():
    ticker_symbol = 'MSFT'
    
    try:
        df = fetch_stock_data(ticker_symbol, days=2)
        
        # Convert 'ds' to datetime for easier manipulation
        df['ds'] = pd.to_datetime(df['ds'], unit='s')
        
        # Group data by day
        grouped = df.groupby(df['ds'].dt.date)
        
        days = list(grouped.groups.keys())
        if len(days) < 2:
            raise ValueError("Not enough days of data available")
        
        day_t_1 = grouped.get_group(days[-2])
        day_t = grouped.get_group(days[-1])
        
        print(f"Total data points: {len(df)}")
        print(f"Day t-1 data points: {len(day_t_1)}")
        print(f"Day t data points: {len(day_t)}")
        print(day_t_1.head())
        print(day_t.tail())
        
        # Plot day t-1 data
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
        
        # Train model on day t-1
        mlf = create_model()
        mlf.fit(day_t_1, prediction_intervals=PredictionIntervals(n_windows=10, h = 1), as_numpy=True)
        
        # Predict for day t and update with actuals
        predictions = []
        lows = []
        highs = []
        actuals = []
        
        for i in range(len(day_t)):
            # Make prediction
            pred = mlf.predict(1, level = [99])
            pred_value = pred['LGBMRegressor'].values[0]
            lower_bound = pred['LGBMRegressor-lo-99'].values[0]
            upper_bound = pred['LGBMRegressor-hi-99'].values[0]

       
            predictions.append(pred_value)
            lows.append(lower_bound)
            highs.append(upper_bound)
            
            # Get actual value
            actual_value = day_t.iloc[i]['y']
            actuals.append(actual_value)
            # print(f"__________________________________________________")
            # print(f"|Predicted: {pred_value} | Actual: {actual_value}|")
            print(tabulate.tabulate([['Predicted', pred_value], ['Actual', actual_value]]))
            
            # Update the model with the actual value
            new_data = pd.DataFrame({
                'unique_id': [day_t.iloc[i]['unique_id']],
                'ds': [day_t.iloc[i]['ds']],
                'y': [actual_value]
            })
            mlf.update(new_data)
        
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
        print(f"Mean Absolute Error: {mae}")
        print(f"Root Mean Square Error: {rmse}")
        
        # Plot simulation results
        plt.figure(figsize=(15, 7))
        plt.plot(day_t['ds'], actuals, label='Actual')
        plt.plot(day_t['ds'], predictions, label='Predicted')
        plt.fill_between(day_t['ds'], lows, highs, color='gray', alpha=0.5, label='95% Prediction Interval')

        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(f'{ticker_symbol} Stock Price: Day t Simulation (Actual vs Predicted)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{ticker_symbol}_day_t_simulation_plot.png')
        plt.close()
        print(f"Simulation plot saved as {ticker_symbol}_day_t_simulation_plot.png")
        
    except Exception as e:
        print(f"An error occurred in main: {e}")
if __name__ == "__main__":
    main()
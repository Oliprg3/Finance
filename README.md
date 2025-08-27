# Finance
Dividend forecasting

# AI-Driven Dividend Forecasting Project

This project uses PyTorch to build an advanced AI model for forecasting dividend payouts based on historical stock data. It employs an LSTM (Long Short-Term Memory) neural network for time-series forecasting, which is suitable for capturing temporal dependencies in financial data.

## Features
- Data fetching and preprocessing using pandas and yfinance.
- LSTM model implemented in PyTorch for multi-step forecasting.
- Hyperparameter tuning with learning rate scheduling.
- Evaluation metrics: MAE, RMSE, MAPE.
- Visualization of predictions vs. actuals using matplotlib.
- Modular structure for easy extension (e.g., add Transformer models).

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download historical stock data (e.g., via yfinance: `python -c "import yfinance as yf; yf.download('AAPL', start='2010-01-01').to_csv('data/raw/AAPL.csv')"` for Apple stock).
3. Run preprocessing: Update `config.yaml` with stock symbols, then run `python src/data.py`.
4. Train the model: `python src/train.py`
5. Make predictions: `python src/predict.py --stock AAPL --future_steps 12`

## Data
- Raw data: Historical OHLCV + dividends from Yahoo Finance (CSV format).
- Processed: Normalized time-series sequences ready for LSTM.

## Model Architecture
- Input: Sequences of past dividends, prices, volumes (window size configurable).
- LSTM layers: 2-3 stacked with dropout.
- Output: Forecasted dividends for future periods (e.g., quarterly).

## Limitations
- Financial markets are volatile; this is for educational purposes, not financial advice.
- Requires internet for data fetching (yfinance).

## Future Improvements
- Integrate alternative data (e.g., sentiment from news via APIs).
- Use ensemble methods or attention mechanisms.

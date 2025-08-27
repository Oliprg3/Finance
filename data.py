import os
import yaml
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def fetch_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data.to_csv(f"data/raw/{symbol}.csv")
    return data

def preprocess_data(df, features, window_size):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size, features.index('Dividends')])  
    
    return np.array(X), np.array(y), scaler

if __name__ == "__main__":
    config = load_config()
    os.makedirs(config['paths']['processed_data_dir'], exist_ok=True)
    
    for symbol in config['data']['symbols']:
        df = fetch_data(symbol, config['data']['start_date'], config['data']['end_date'])
        df = df.fillna(0) 
        X, y, scaler = preprocess_data(df, config['data']['features'], config['data']['window_size'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['data']['test_size'], shuffle=False)
        
        np.save(f"{config['paths']['processed_data_dir']}/{symbol}_X_train.npy", X_train)
        np.save(f"{config['paths']['processed_data_dir']}/{symbol}_y_train.npy", y_train)
        np.save(f"{config['paths']['processed_data_dir']}/{symbol}_X_test.npy", X_test)
        np.save(f"{config['paths']['processed_data_dir']}/{symbol}_y_test.npy", y_test)
        np.save(f"{config['paths']['processed_data_dir']}/{symbol}_scaler.npy", scaler)

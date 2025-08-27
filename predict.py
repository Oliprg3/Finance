import argparse
import yaml
import numpy as np
import torch
from models import DividendLSTM
import matplotlib.pyplot as plt

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def predict(symbol, future_steps, config):
    scaler = np.load(f"{config['paths']['processed_data_dir']}/{symbol}_scaler.npy", allow_pickle=True).item()
    X_test = np.load(f"{config['paths']['processed_data_dir']}/{symbol}_X_test.npy")
    
    input_size = X_test.shape[2]
    model = DividendLSTM(input_size, config['model']['lstm_hidden_size'], config['model']['lstm_num_layers'], config['model']['dropout'])
    model.load_state_dict(torch.load(config['paths']['model_save_path']))
    model.eval()
    
 
    with torch.no_grad():
        test_inputs = torch.from_numpy(X_test).float()
        predictions = model(test_inputs).numpy()
    
    y_test = np.load(f"{config['paths']['processed_data_dir']}/{symbol}_y_test.npy")
    predictions = scaler.inverse_transform(np.concatenate((np.zeros((len(predictions), len(config['data']['features'])-1)), predictions), axis=1))[:, -1]
    actuals = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), len(config['data']['features'])-1)), y_test.reshape(-1,1)), axis=1))[:, -1]
    
    plt.plot(actuals, label='Actual Dividends')
    plt.plot(predictions, label='Predicted')
    plt.title(f'{symbol} Dividend Forecast')
    plt.legend()
    plt.show()
    

    last_sequence = X_test[-1]
    future_preds = []
    for _ in range(future_steps):
        with torch.no_grad():
            pred = model(torch.from_numpy(last_sequence).float().unsqueeze(0)).item()
        future_preds.append(pred)
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1, -1] = pred  
    
    future_preds = scaler.inverse_transform(np.concatenate((np.zeros((len(future_preds), len(config['data']['features'])-1)), np.array(future_preds).reshape(-1,1)), axis=1))[:, -1]
    print(f"Future {future_steps} steps dividends: {future_preds}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', required=True)
    parser.add_argument('--future_steps', type=int, default=12)
    args = parser.parse_args()
    
    config = load_config()
    predict(args.stock, args.future_steps, config)

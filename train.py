import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import DividendLSTM

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_model(symbol, config):
    X_train = np.load(f"{config['paths']['processed_data_dir']}/{symbol}_X_train.npy")
    y_train = np.load(f"{config['paths']['processed_data_dir']}/{symbol}_y_train.npy")
    
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float().unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], shuffle=True)
    
    input_size = X_train.shape[2]  
    model = DividendLSTM(input_size, config['model']['lstm_hidden_size'], config['model']['lstm_num_layers'], config['model']['dropout'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    for epoch in range(config['model']['epochs']):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{config['model']['epochs']}, Loss: {loss.item()}")
        scheduler.step(loss.item())
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), config['paths']['model_save_path'])

if __name__ == "__main__":
    config = load_config()
    for symbol in config['data']['symbols']: 
        train_model(symbol, config)

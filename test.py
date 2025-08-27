import pytest
import torch
from src.models import DividendLSTM

def test_lstm_model():
    input_size = 3
    hidden_size = 64
    num_layers = 2
    model = DividendLSTM(input_size, hidden_size, num_layers)
    dummy_input = torch.randn(32, 60, input_size)  
    output = model(dummy_input)
    assert output.shape == (32, 1), "Output shape mismatch"

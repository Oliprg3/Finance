import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_predictions(actuals, predictions):
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

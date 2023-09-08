import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class SWRC_Model:
    def __init__(self, model_name, model_function, param_names, param_bounds=None):
        self.model_name = model_name
        self.model_function = model_function
        self.param_names = param_names
        self.params = None
        if param_bounds:
            self.param_bounds = (list(param_bounds[name][0] for name in param_names),
                                 list(param_bounds[name][1] for name in param_names))
        else:
            self.param_bounds = None

    def fit(self, data):
        if self.param_bounds:
            self.params, _ = curve_fit(self.model_function, data.x, data.y, bounds=self.param_bounds)
        else:
            self.params, _ = curve_fit(self.model_function, data.x, data.y)
        return dict(zip(self.param_names, self.params))
    
    def error_metrics(self, data):
        """Calculate and return the error metrics."""
        y_pred = self.predict(data.x)
        y_true = data.y
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        nse = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
        return mse, rmse, mae, r2, nse

    def predict(self, x):
        return self.model_function(x, *self.params)

    def plot(self, data):
        plt.scatter(data.x, data.y, label="Observed")
        plt.plot(data.x, self.predict(data.x), 'r-', label=f"Fitted {self.model_name}")
        plt.legend()
        plt.show()
        
        mse, rmse, mae, r2, nse = self.error_metrics(data)
        print(f"Error Metrics for {self.model_name} Model:")
        print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2: {r2}, NSE: {nse}")

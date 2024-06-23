import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Model_base
class SWRC_base:
    """Base class for soil water retention curve models."""

    def __init__(self, model_name, model_function, param_names, param_bounds=None):
        """Initialize the model with its name, function, parameter names, and bounds.

        Parameters
        ----------
        model_name : str
            Name of the model.
        model_function : callable
            Function that implements the model.
        param_names : list of str
            Names of the model parameters.
        param_bounds : dict, optional
            Parameter bounds for the model fitting.
        """
        self.model_name = model_name
        self.model_function = model_function
        self.param_names = param_names
        self.params = None
        self.param_bounds = self._process_param_bounds(param_bounds, param_names)

    def _process_param_bounds(self, param_bounds, param_names):
        """Process parameter bounds and return a tuple of lower and upper bounds.

        Parameters
        ----------
        param_bounds : dict, optional
            Parameter bounds for the model fitting.
        param_names : list of str
            Names of the model parameters.

        Returns
        -------
        tuple of lists
            A tuple containing two lists: the first list contains the lower bounds for each parameter,
            and the second list contains the upper bounds.
        """
        if param_bounds:
            return (list(param_bounds[name][0] for name in param_names),
                    list(param_bounds[name][1] for name in param_names))
        else:
            return None

    def fit(self, data):
        """Fit the model to the data and return the estimated parameters.

        Parameters
        ----------
        data : object with attributes x and y
            Data to fit the model to.

        Returns
        -------
        dict
            A dictionary containing the estimated parameters and their standard errors.
        """
        self.params, covariance = curve_fit(self.model_function, data.x, data.y, bounds=self.param_bounds)
        std_errors = np.sqrt(np.diag(covariance))
        return dict(zip(self.param_names, zip(self.params, std_errors)))
    
    def predict(self, x):
        """Predict the output value for the given input.

        Parameters
        ----------
        x : array-like
            Input values.

        Returns
        -------
        array-like
            Predicted output values.
        """
        return self.model_function(x, *self.params)

    def error_metrics(self, data):
        """Calculate and return the error metrics.

        Parameters
        ----------
        data : object with attributes x and y
            Data to calculate error metrics on.

        Returns
        -------
        tuple
            A tuple containing MSE, RMSE, MAE, R^2, and NSE.
        """
        y_pred = self.predict(data.x)
        y_true = data.y
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        nse = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
        return mse, rmse, mae, r2, nse

    def plot(self, data):
        """Plot the observed data and the fitted model predictions.

        Parameters
        ----------
        data : object with attributes x and y
            Data to plot.
        """
        self._plot_data_and_predictions(data)
        mse, rmse, mae, r2, nse = self.error_metrics(data)
        print(f"Error Metrics for {self.model_name} Model:")
        print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R^2: {r2}, NSE: {nse}")

    def _plot_data_and_predictions(self, data):
        """Plot the observed data and the fitted model predictions."""
        plt.scatter(data.x, data.y, label="Observed")
        plt.plot(data.x, self.predict(data.x), 'r-', label=f"Fitted {self.model_name}")
        plt.legend()
        plt.show()

    def __str__(self):
        """Return a string representation of the model."""
        return f"{self.model_name} with parameters: {self.params}"

# Model1
class SWRC_Model1(SWRC_base):
    """SWRC Model 1: van Genuchten model."""

    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model1", self.model1_function, ['theta_R', 'theta_S', 'alpha', 'n'], param_bounds)

    @staticmethod
    def model1_function(x, theta_R, theta_S, alpha, n):
        """van Genuchten model function.

        Parameters
        ----------
        x : array-like
            Input values (e.g., soil water content).
        theta_R : float
            Residual water content.
        theta_S : float
            Saturated water content.
        alpha : float
            Shape parameter.
        n : float
            Exponent parameter.

        Returns
        -------
        array-like
            Predicted output values.
        """
        m = 1 - 1/n
        sat_index = (1 + (alpha * abs(x)) ** n) ** (-m)
        return theta_R + (theta_S - theta_R) * sat_index

# Model2
class SWRC_Model2(SWRC_base):
    """SWRC Model 2: Brooks-Corey model."""

    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model2", self.model2_function, ['k0', 'k1', 'n'], param_bounds)

    @staticmethod
    def model2_function(x, k0, k1, n):
        """Brooks-Corey model function.

        Parameters
        ----------
        x : array-like
            Input values (e.g., soil water content).
        k0 : float
            Air entry suction.
        k1 : float
            Slope of the retention curve.
        n : float
            Exponent parameter.

        Returns
        -------
        array-like
            Predicted output values.
        """
        return k1 * (np.exp(-k0 / 6.653**n) - np.exp(-k0 / x**n))

# Model3
class SWRC_Model3(SWRC_base):
    """SWRC Model 3: Three-parameter exponential model."""

    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model3", self.model3_function, ['theta_R', 'a1', 'p1', 'a2', 'p2'], param_bounds)

    @staticmethod
    def model3_function(x, theta_R, a1, p1, a2, p2):
        """Three-parameter exponential model function.

        Parameters
        ----------
        x : array-like
            Input values (e.g., soil water content).
        theta_R : float
            Residual water content.
        a1 : float
            Amplitude of the first exponential term.
        p1 : float
            Time constant of the first exponential term.
        a2 : float
            Amplitude of the second exponential term.
        p2 : float
            Time constant of the second exponential term.

        Returns
        -------
        array-like
            Predicted output values.
        """
        return theta_R + a1 * np.exp(-x/p1) + a2 * np.exp(-x/p2)

# Model4 (Silva)
class SWRC_Model4_Silva(SWRC_base):
    """SWRC Model 4 (Silva): Silva et al.'s model."""

    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model4_Silva", self.model4_silva_function, ['Bd', 'a', 'b', 'c'], param_bounds)

    @staticmethod
    def model4_silva_function(x, Bd, a, b, c):
        """Silva et al.'s model function.

        Parameters
        ----------
        x : array-like
            Input values (e.g., soil water content).
        Bd : float
            Bulk density.
        a : float
            Coefficient for the exponential term.
        b : float
            Coefficient for the linear term.
        c : float
            Exponent for the input term.

        Returns
        -------
        array-like
            Predicted output values.
        """
        return np.exp(a + b * Bd) * x ** c

# Model4 (Ross)
class SWRC_Model4_Ross(SWRC_base):
    """SWRC Model 4 (Ross): Ross's model."""

    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model4_Ross", self.model4_ross_function, ['a', 'c'], param_bounds)

    @staticmethod
    def model4_ross_function(x, a, c):
        """Ross's model function.

        Parameters
        ----------
        x : array-like
            Input values (e.g., soil water content).
        a : float
            Coefficient for the power law term.
        c : float
            Exponent for the input term.

        Returns
        -------
        array-like
            Predicted output values.
        """
        return a * x ** c

# Model5
class SWRC_Model5(SWRC_base):
    """SWRC Model 5: Five-parameter model."""

    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model5", self.model5_function, ['theta_R', 'theta_S', 'sat_index', 'b0', 'b1', 'b2'], param_bounds)

    @staticmethod
    def model5_function(x, theta_R, theta_S, sat_index, b0, b1, b2):
        """Five-parameter model function.

        Parameters
        ----------
        x : array-like
            Input values (e.g., soil water content).
        theta_R : float
            Residual water content.
        theta_S : float
            Saturated water content.
        sat_index : float
            Saturation index.
        b0 : float
            Intercept of the linear term.
        b1 : float
            Coefficient for the linear term.
        b2 : float
            Coefficient for the quadratic term.

        Returns
        -------
        array-like
            Predicted output values.
        """
        return theta_R + (theta_S - theta_R) * sat_index + b0 + b1 * x + b2 * x**2
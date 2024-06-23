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
        """
        Parameters
        ----------
        x :
            the matric potential.
        theta_R :
            the residual water content.
        theta_S :
            the water content at saturation.
        alpha :
            a scale parameter of the van Genuchten’s formula.
        n :
            a shape parameter in van Genuchten’s formula.
        m :
            a shape parameter in van Genuchten’s Formula. Default is 1 − 1/n (Mualem,1976).

        Returns
        -------
        array-like
            Predicted output values.

        References
        -------
            [1]Genuchten, M. T. van. (1980). A closed form equation for predicting the hydraulic conductivity of
            unsaturated soils. Soil Science Society of America Journal, 44:892-898.
            [2]Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of unsaturated porous
            media. Water Resources Research, 12:513-522.
        """
        m = 1 - 1/n
        sat_index = (1 + (alpha * abs(x)) ** n) ** (-m)
         # If saturation.index is True, return the saturation index, otherwise return the soil water content
        return theta_R + (theta_S - theta_R) * sat_index

# Model2
class SWRC_Model2 (SWRC_base):
    """SWRC Model based on Groenevelt & Grant (2004) model."""

    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model2", self.model_function, ['x0', 'k0', 'k1', 'n'], param_bounds)

    @staticmethod
    def model_function(h, x0, k0, k1, n):
        """Groenevelt & Grant (2004) model function.

        Parameters
        ----------
        h : array-like
            Pore water suction (hPa).
        x0 : float
            The value of pF at which the soil water content becomes zero.The default is 6.653.
        k0 : float
            A parameter value.
        k1 : float
            A parameter value.
        n : float
            A parameter value.

        Returns
        -------
        array-like
            Predicted soil water content.

        References
        -------
            Groenevelt & Grant (2004). A newmodel for the soil-water retention curve that solves the problem
            of residualwater contents. European Journal of Soil Science, 55:479-485.
        """
        #  x = logh (pore water suction), and h is in units of hPa
        x = np.log(h)
        # Calculate soil water content based on the Groenevelt & Grant (2004) model
        return k1 * np.exp(-k0 / (x0 ** n)) - k1 * np.exp(-k0 / (x ** n))

# Model3
class SWRC_Model3(SWRC_base):
    """SWRC Model 3: based on the Dexter’s (2008) formula."""

    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model3", self.model3_function, ['theta_R', 'a1', 'p1', 'a2', 'p2'], param_bounds)

    @staticmethod
    def model3_function(x, theta_R, a1, p1, a2, p2):
        """Soil Water Retention, based on the Dexter’s (2008) formula.

        Parameters
        ----------
        x :
            a numeric vector containing the values of applied air pressure.
        theta_R :
            a parameter that represents the residual water content.
        a1 :
            a parameter that represents the drainable part of the textural pore space in units 
            of gravimetric water content at saturation.
        p1 :
            a parameter that represents the applied air pressures characteristic 
            for displacement of water from the textural pore space.
        a2 :
            a parameter that represents the total structural pore space in units of gravimetric
            water content at saturation.
        p2 :
            a parameter that represents the applied air pressure that is characteristic 
            for displacing water from the structural pores.

        Returns
        -------
        array-like
            Predicted output values.

        References
        -------
            [1] Dexter et al. (2008). A user-friendly water retention function that takes account of the textural and
            structural pore spaces in soil. Geoderma, 143:243-253.
        """
        return theta_R + a1 * np.exp(-x/p1) + a2 * np.exp(-x/p2)

# Model4 (Silva)
class SWRC_Model4_Silva(SWRC_base):
    """SWRC Model 4 (Silva): Silva et al.'s model."""

    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model4_Silva", self.model4_silva_function, ['Bd', 'a', 'b', 'c'], param_bounds)

    @staticmethod
    def model4_silva_function(psi, Bd, a, b, c):
        """Silva et al.'s model function.

        Parameters
        ----------
        psi :
            a numeric vector containing values of water potential (Psi).
        Bd :
            a numeric vector containing values of dry bulk density.
        a :
            a model-fitting parameter. See details.
        b :
            a model-fitting parameter. See details.
        c :
            a model-fitting parameter. See details.

        Returns
        -------
        array-like
            Predicted output values.

        References
        -------  
            [1]Silva et al. (1994). Characterization of the least limiting water range of soils. Soil Science Society
            of America Journal, 58:1775-1781.  
        """
        return np.exp(a + b * Bd) * psi ** c

# Model4 (Ross)
class SWRC_Model4_Ross(SWRC_base):
    """SWRC Model 4 (Ross): Ross's model."""

    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model4_Ross", self.model4_ross_function, ['a', 'c'], param_bounds)

    @staticmethod
    def model4_ross_function(psi, a, c):
        """Ross's model function.

        Parameters
        ----------
        psi :
            a numeric vector containing values of water potential (Psi).
        a :
            a model-fitting parameter. See details.
        c :
            a model-fitting parameter. See details.

        Returns
        -------
        array-like
            Predicted output values.

        References
        -------  
            [1]Ross et al. (1991). Equation for extending water-retention curves to dryness. Soil Science Society
            of America Journal, 55:923-927.
        """
        return a * psi ** c

# Model5
class SWRC_Model5(SWRC_base):
    """SWRC Model 5: Soil Water Retention, based on the modified van Genuchten’s formula"""

    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model5", self.model5_function, ['theta_R', 'theta_S', 'alpha', 'n', 'b0', 'b1', 'b2'], param_bounds)

    @staticmethod
    def model5_function(x, theta_R, theta_S, alpha, n, b0, b1, b2):
        """Function to calculate the soil water content based on the modified van Genuchten’s formula, as
            suggested by Pierson and Mulla (1989).

        Parameters
        ----------
        x:
            the matric potential.
        theta_R:
            the residual water content.
        theta_S: 
            the water content at saturation.
        alpha: 
            a scale parameter of the van Genuchten’s formula.
        n:
            a shape parameter in van Genuchten’s formula.
        m: 
            a shape parameter in van Genuchten’s Formula. Default is 1 − 1/n (Mualem,1976).

        Returns
        -------
        array-like
            Predicted output values.
        
        References
        -------  
            [1]Pierson, F.B.; Mulla, D.J. (1989) An Improved Method for Measuring Aggregate Stability of a
            Weakly Aggregated Loessial Soil. Soil Sci. Soc. Am. J., 53:1825–1831.
        """
        m = 1 - 1/n
        sat_index = (1 + (alpha * abs(x)) ** n) ** (-m)
        return theta_R + (theta_S - theta_R) * sat_index + b0 + b1 * x + b2 * x**2
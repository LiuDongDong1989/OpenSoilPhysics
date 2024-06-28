import numpy as np
from scipy.optimize import curve_fit

#土壤水分保持曲线模型的基类
class SWRC_base:
    """
    Soil Water Retention Curve base class.
    """

    def __init__(self, model_name: str, model_function, param_names: list, param_dict: dict):
        """
        Initialize the model with a name, function, and parameter details.
        
        :param model_name: The name of the model.
        :param model_function: The function representing the model.
        :param param_names: List of parameter names.
        :param param_dict: Dictionary with parameter names as keys and tuples of (lower, upper) bounds as values.
        """
        # Check if param_names and param_dict have the same keys
        mismatched_keys = set(param_names) ^ set(param_dict.keys())
        if mismatched_keys:
            raise ValueError(f"Mismatched keys between param_names and param_dict: {mismatched_keys}. Please check your input.")

        for name, bounds in param_dict.items():
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(f"Parameter '{name}' has an invalid bound setting. It should be a tuple of (lower, upper) bounds.")
            
        self.model_name = model_name
        self.model_function = model_function
        self.param_names = param_names
        self.param_dict = param_dict
        self.param_bounds = self._process_param_bounds()
    
    def _process_param_bounds(self) -> tuple:
        """
        Process parameter bounds and return them as lower and upper bounds.
        
        :return: A tuple containing two lists: lower and upper bounds for each parameter.
        """
        return ([bound[0] for bound in self.param_dict.values()], [bound[1] for bound in self.param_dict.values()])
        
    def forwardCalculation(self, x) -> float:
        """
        Predict the output value for a given input.
        
        :param x: Input value(s).
        :return: Predicted output value(s).
        """
        if self.param_bounds:
            raise ValueError("The model has bounded parameters. Please use the 'fit' method to estimate the parameters first.")
        
        return self.model_function(x, *self.param_dict.values())

    def fit(self, data) -> dict:
        """
        Fit the model to data and return estimated parameters and error metrics.
        
        :param data: Data object with 'x' and 'y' attributes.
        :return: A dictionary containing estimated parameters and error metrics.
        """
        self.params, covariance = curve_fit(self.model_function, data.x, data.y, bounds=self.param_bounds)
        
        std_errors = np.sqrt(np.diag(covariance))
        self.param_dict = {name: (param, error) for name, (param, error) in zip(self.param_names, zip(self.params, std_errors))}
        
        y_pred = self.model_function(data.x, *self.params)
        mse = np.mean((y_pred - data.y) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - data.y))
        r2 = 1 - (np.sum((data.y - y_pred) ** 2) / np.sum((data.y - np.mean(data.y)) ** 2))
        error_metrics_dict = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

        # Print each key-value pair on a separate line
        result_dict = {**self.param_dict, **error_metrics_dict}
        for key, value in result_dict.items():
            print(f"{key}: {value}")

        return result_dict
    
# Model vanGenuchten （经典）
class vanGenuchten (SWRC_base):
    """van Genuchten model"""

    def __init__(self, param_dict=None):
        super().__init__("vanGenuchten", self.model_function, ['theta_r', 'theta_s', 'alpha', 'n'], param_dict)

    @staticmethod
    def model_function(x, theta_r, theta_s, alpha, n):
        """
        Parameters
        ----------
        x :
            the matric potential, （hPa or cm).
            备注：1百帕(hPa) = 1.01974厘米水柱(cmH2O)
        theta_r :
            the residual water content (cm3 cm−3).
        theta_s :
            the water content at saturation (cm3 cm−3).
        alpha :
            a scale parameter of the van Genuchten’s formula(hPa−1).
        n :
            a shape parameter in van Genuchten’s formula(dimensionless).
        m :
            a shape parameter in van Genuchten’s Formula. Default is 1 − 1/n (Mualem,1976)(dimensionless).

        Returns
        -------
        array-like
            Predicted output values.

        References
        ----------
            [1]Genuchten, M. T. van. (1980). A closed form equation for predicting the hydraulic conductivity of
            unsaturated soils. Soil Science Society of America Journal, 44:892-898.
            [2]Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of unsaturated porous
            media. Water Resources Research, 12:513-522.
        """
        m = 1 - 1/n
        Se = (1 + (alpha * abs(x)) ** n) ** (-m)
        return theta_r + (theta_s - theta_r) * Se   

# Model BrooksCorey （经典）
class BrooksCorey(SWRC_base):
    """Brooks and Corey model for soil water retention curve."""

    def __init__(self, param_dict=None):
        super().__init__("BrooksCorey", self.model_function, ['theta_r', 'theta_s', 'lambda_', 'h_a'], param_dict)

    @staticmethod
    def model_function(x, theta_r, theta_s, lambda_, h_a):
        """
        Parameters
        ----------
        x : array-like
            Matric potential (hPa or cm).
        theta_R : float
            Residual water content (cm^3/cm^3).
        theta_S : float
            Saturation water content (cm^3/cm^3).
        lambda_ : float
            Brooks-Corey model parameter (dimensionless).
        h_a : float
            Capillary pressure at the air-entry value (hPa or cm).

        Returns
        -------
        array-like
            Predicted soil water content (cm^3/cm^3).

        References
        ----------
            [1]Brooks, R. H., & Corey, A. T. (1964). Hydraulic properties of porous
            media and their relation to drainage design. Transactions of the ASAE,
            7, 26–28. https://doi.org/10.13031/2013.40684
        """
        Se = np.where(x > h_a, abs((x / h_a) ** -lambda_), 1)
        return theta_r + (theta_s - theta_r) * Se

# Model Durner （经典）
class Durner(SWRC_base):
    """Durner model for soil water retention curve."""

    def __init__(self, param_dict=None):
        super().__init__("Durner", self.model_function, ['theta_r', 'theta_s', 'alpha1', 'n1', 'alpha2', 'n2', 'w1'], param_dict)

    @staticmethod
    def model_function(x, theta_r, theta_s, alpha1, n1, alpha2, n2, w1):
        """
        Parameters
        ----------
        x : array-like
            Matric potential (hPa or cm).
        theta_r : float
            Residual water content (cm^3/cm^3).
        theta_s : float
            Saturation water content (cm^3/cm^3).
        alpha1 : float
            Scale parameter for the first pore system (hPa^-1).
        n1 : float
            Shape parameter for the first pore system (dimensionless).
        alpha2 : float
            Scale parameter for the second pore system (hPa^-1).
        n2 : float
            Shape parameter for the second pore system (dimensionless).
        w1 : float
            Weight factor for the first pore system (dimensionless).

        Returns
        -------
        array-like
            Predicted soil water content (cm^3/cm^3).

        References
        ----------
            [1]Durner, W. (1994). Hydraulic conductivity estimation for soils with heterogeneous pore structure. 
            Water Resources Research, 30, 211–223. https://doi.org/10.1029/93WR02676
        """
        m1 = 1 - 1/n1
        m2 = 1 - 1/n2
        Se1 = (1 + (alpha1 * abs(x)) ** n1) ** (-m1)
        Se2 = (1 + (alpha2 * abs(x)) ** n2) ** (-m2)
        Se = w1 * Se1 + (1 - w1) * Se2
        return theta_r + (theta_s - theta_r) * Se

# Model GroeneveltGrant （不常用）
class GroeneveltGrant (SWRC_base):
    """Groenevelt & Grant (2004) model."""

    def __init__(self, param_dict=None):
        super().__init__("GroeneveltGrant", self.model_function, ['x0', 'k0', 'k1', 'n'], param_dict)

    @staticmethod
    def model_function(h, x0, k0, k1, n):
        """Groenevelt & Grant (2004) model function.

        Parameters
        ----------
        h : array-like
            Pore water suction (hPa).
        x0 : float
            The value of pF at which the soil water content becomes zero. The default is 6.653.
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
        x = np.log10(h)
        # Calculate soil water content based on the Groenevelt & Grant (2004) model
        return k1 * np.exp(-k0 / (x0 ** n)) - k1 * np.exp(-k0 / (x ** n))

# Model Dexter （不常用）
class Dexter(SWRC_base):
    """Dexter’s (2008) formula."""

    def __init__(self, param_dict=None):
        super().__init__("Dexter", self.model_function, ['theta_r', 'a1', 'p1', 'a2', 'p2'], param_dict)

    @staticmethod
    def model_function(x, theta_r, a1, p1, a2, p2):
        """Soil Water Retention, based on the Dexter’s (2008) formula.

        Parameters
        ----------
        x :
            a numeric vector containing the values of applied air pressure.
        theta_r :
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
        return theta_r + a1 * np.exp(-x/p1) + a2 * np.exp(-x/p2)

# Model ModifiedvanGenuchten (不常用)
class ModifiedvanGenuchten(SWRC_base):
    """The modified van Genuchten’s formula"""

    def __init__(self, param_dict=None):
        super().__init__("ModifiedvanGenuchten", self.model_function, ['theta_r', 'theta_s', 'alpha', 'n', 'b0', 'b1', 'b2'], param_dict)

    @staticmethod
    def model_function(x, theta_r, theta_s, alpha, n, b0, b1, b2):
        """Function to calculate the soil water content based on the modified van Genuchten’s formula, as
            suggested by Pierson and Mulla (1989).

        Parameters
        ----------
        x:
            the matric potential.
        theta_r:
            the residual water content.
        theta_s: 
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
        return theta_r + (theta_s - theta_r) * sat_index + b0 + b1 * x + b2 * x**2

# Model Silva (不常用)
class Silva(SWRC_base):
    """Silva et al.'s model."""

    def __init__(self, param_dict=None):
        super().__init__("Silva", self.model_function, ['Bd', 'a', 'b', 'c'], param_dict)

    @staticmethod
    def model_function(x, Bd, a, b, c):
        """Silva et al.'s model function.

        Parameters
        ----------
        x :
            a numeric vector containing values of water potential (hPa).
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
        #1 hPa等于0.0102 psi
        psi = 0.0102 * x
        return np.exp(a + b * Bd) * psi ** c

# Model Ross (不常用)
class Ross(SWRC_base):
    """SWRC Model 4 (Ross): Ross's model."""

    def __init__(self, param_dict=None):
        super().__init__("Ross", self.model_function, ['a', 'c'], param_dict)

    @staticmethod
    def model_function(x, a, c):
        """Ross's model function.

        Parameters
        ----------
        x :
            a numeric vector containing values of water potential (hPa).
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
        #1 hPa等于0.0102 psi
        psi = 0.0102 * x
        return a * psi ** c
from PYSOILPHYSICS.base_model import SWRC_Model

class SWRC_Model1(SWRC_Model):
    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model1", self.model1_function, ['theta_R', 'theta_S', 'alpha', 'n'], param_bounds)

    @staticmethod
    def model1_function(x, theta_R, theta_S, alpha, n):
        m = 1 - 1/n
        sat_index = (1 + (alpha * abs(x)) ** n) ** (-m)
        return theta_R + (theta_S - theta_R) * sat_index

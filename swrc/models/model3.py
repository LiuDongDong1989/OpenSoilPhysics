import numpy as np
from base_model import SWRC_Model
# Model3
class SWRC_Model3(SWRC_Model):
    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model3", self.model3_function, ['theta_R', 'a1', 'p1', 'a2', 'p2'], param_bounds)

    @staticmethod
    def model3_function(x, theta_R, a1, p1, a2, p2):
        return theta_R + a1 * np.exp(-x/p1) + a2 * np.exp(-x/p2)
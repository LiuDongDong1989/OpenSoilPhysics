import numpy as np
from PYSOILPHYSICS.base_model import SWRC_Model

# Model2
class SWRC_Model2(SWRC_Model):
    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model2", self.model2_function, ['k0', 'k1', 'n'], param_bounds)

    @staticmethod
    def model2_function(x, k0, k1, n):
        return k1 * (np.exp(-k0 / 6.653**n) - np.exp(-k0 / x**n))
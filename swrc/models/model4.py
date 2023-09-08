import numpy as np
from base_model import SWRC_Model

# Model4 (Silva)
class SWRC_Model4_Silva(SWRC_Model):
    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model4_Silva", self.model4_silva_function, ['Bd', 'a', 'b', 'c'], param_bounds)

    @staticmethod
    def model4_silva_function(x, Bd, a, b, c):
        return np.exp(a + b * Bd) * x ** c
    
# Model4 (Ross)
class SWRC_Model4_Ross(SWRC_Model):
    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model4_Ross", self.model4_ross_function, ['a', 'c'], param_bounds)

    @staticmethod
    def model4_ross_function(x, a, c):
        return a * x ** c
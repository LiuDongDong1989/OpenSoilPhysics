from swrc.base_model import SWRC_Model
# Model5
class SWRC_Model5(SWRC_Model):
    def __init__(self, param_bounds=None):
        super().__init__("SWRC_Model5", self.model5_function, ['theta_R', 'theta_S', 'sat_index', 'b0', 'b1', 'b2'], param_bounds)

    @staticmethod
    def model5_function(x, theta_R, theta_S, sat_index, b0, b1, b2):
        return theta_R + (theta_S - theta_R) * sat_index + b0 + b1 * x + b2 * x**2
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class Sindex:
    def __init__(self, theta_r, theta_s, alpha, n):
        """
        初始化 Sindex 类的实例。
        
        参数:
        theta_r: 残余土壤含水量 (体积比)
        theta_s: 饱和土壤含水量 (体积比)
        alpha: 与土壤特性有关的参数
        n: 孔隙结构指数
        """
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.alpha = alpha
        self.n = n

    def calculate_S_index(self, vcov=None, nsim=999, conf_level=0.95, graph=False):
        """
        计算土壤水分特征曲线的 S 指数。
        
        参数:
        vcov: 协方差矩阵，用于计算置信区间，默认为 None
        nsim: 模拟次数，用于计算置信区间，默认为 999
        conf_level: 置信水平，默认为 0.95
        graph: 是否绘制土壤水分特征曲线，默认为 True
        
        返回值:
        一个字典，包含 S 指数及相关信息
        """
        m = 1 - 1/self.n
        h_i = 1/self.alpha * (1/m)**(1/self.n)
        theta_i = self.theta_r + (self.theta_s - self.theta_r) * (1 + 1/m)**(-m)
        b1 = -self.n * (self.theta_s - self.theta_r) * (1 + 1/m)**(-(1 + m))
        b0 = theta_i - b1 * h_i

        if graph:
            x = np.linspace(0, 15000, num=1000)
            Se = (1 + (self.alpha * abs(x)) ** self.n) ** (-m)
            y = self.theta_r + (self.theta_s - self.theta_r) * Se
            plt.plot(x, y, label="Soil Water Retention Curve")
            plt.plot([h_i, h_i], [0, theta_i], linestyle='--', label="Threshold")
            plt.plot([0, h_i], [theta_i, theta_i], linestyle='--')
            plt.xlabel("Matric potential")
            plt.ylabel("Soil water content")
            plt.title("Soil Water Retention Curve")
            plt.legend()
            plt.show()

        S = abs(b1)
        if S >= 0.05:
            clas = "Very good"
        elif S < 0.05 and S >= 0.035:
            clas = "Good"
        elif S < 0.035 and S > 0.02:
            clas = "Poor"
        else:
            clas = "Very poor"

        if vcov is not None:
            mu = np.array([self.theta_r, self.theta_s, self.alpha, self.n])
            if mu.shape[0] != vcov.shape[0]:
                raise ValueError("vcov misspecified!")
            sim = stats.multivariate_normal.rvs(mean=mu, cov=vcov, size=nsim)
            m_sim = 1 - 1/sim[:, 3]
            sim = np.hstack((sim, m_sim.reshape(-1, 1)))
            S_sim = np.abs(-sim[:, 3] * (sim[:, 1] - sim[:, 0]) * (1 + 1/sim[:, 4])**(-(1 + sim[:, 4])))
            sig = (1 - conf_level)/2
            sci = np.percentile(S_sim, [sig, 100 - sig])
        else:
            sci = None

        result = {
            "h_i": h_i,
            "theta_i": theta_i,
            "S_index": S,
            "PhysicalQuality": clas,
            "simCI": sci,
            "conf_level": conf_level
        }

        return result

# 示例：使用 Sindex 类
if __name__ == "__main__":
    sindex = Sindex(theta_r=0.0, theta_s=0.395, alpha=0.0217, n=1.103)
    result = sindex.calculate_S_index()
    print(result)
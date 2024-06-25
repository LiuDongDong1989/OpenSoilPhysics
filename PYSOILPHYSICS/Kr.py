class Kr:
    def __init__(self, Ks, theta_r, theta_s, alpha, n):
        """
        构造函数，用于初始化 Kr 类的实例。
        
        :param Ks: 饱和渗透率 (cm/s)
        :param alpha: 与土壤特性有关的参数
        :param n: 孔隙结构指数
        :param theta_s: 饱和土壤含水量 (体积比)
        :param theta_r: 残余土壤含水量 (体积比)
        """
        self.Ks = Ks
        self.alpha = alpha
        self.n = n
        self.theta_s = theta_s
        self.theta_r = theta_r

    def Kr_h(self, h_values, f=0.5):
        """
        计算根据给定参数计算土壤水分特征曲线中的渗透率。
        
        :param h_values: 一个包含 h 值的列表或单个 h 值
        :param f: 形状因子，默认值为 0.5
        :return: 一个包含对应 h 值的渗透率列表或单个渗透率值
        """
        if isinstance(h_values, list):
            return [self._calculate_Kr_h(h, f) for h in h_values]
        else:
            return self._calculate_Kr_h(h_values, f)

    def _calculate_Kr_h(self, h, f):
        """
        内部辅助方法，用于计算给定 h 值下的渗透率。
        
        :param h: 土壤水头高度 (cm)
        :param f: 形状因子
        :return: 给定 h 值下的渗透率
        """
        m = 1 - (1 / self.n)
        Se = (1 / (1 + (self.alpha * h) ** self.n)) ** m
        b = (1 - (1 - Se ** (self.n / (self.n - 1))) ** m) ** 2
        out = self.Ks * (Se ** f) * b
        return out

    def Kr_theta(self, theta_values, f=0.5):
        """
        计算根据给定参数计算土壤水分特征曲线中的渗透率。
        
        :param theta_values: 一个包含 theta 值的列表或单个 theta 值
        :param f: 形状因子，默认值为 0.5
        :return: 一个包含对应 theta 值的渗透率列表或单个渗透率值
        """
        if isinstance(theta_values, list):
            return [self._calculate_Kr_theta(theta, f) for theta in theta_values]
        else:
            return self._calculate_Kr_theta(theta_values, f)

    def _calculate_Kr_theta(self, theta, f):
        """
        内部辅助方法，用于计算给定 theta 值下的渗透率。
        
        :param theta: 土壤含水量 (体积比)
        :param f: 形状因子
        :return: 给定 theta 值下的渗透率
        """
        m = 1 - (1 / self.n)
        Se = (theta - self.theta_r) / (self.theta_s - self.theta_r)
        a = (1 - (1 - Se ** (1 / m)) ** m) ** 2
        out = self.Ks * (Se ** f) * a
        return out

# 示例：使用 Kr 类计算非饱和导水率
if __name__ == "__main__":
    # 创建 Kr 类的实例并通过构造函数指定变量值
    kr_instance = Kr(Ks=1e-5, theta_r=0.1, theta_s=0.4, alpha=0.1, n=2)

    # 计算一列 h 值下的非饱和导水率
    h_values = [10, 20, 30]
    result = kr_instance.Kr_h(h_values)
    print("Result for h values:", result)

    # 计算一列 theta 值下的非饱和导水率
    theta_values = [0.2, 0.3, 0.4]
    result_theta = kr_instance.Kr_theta(theta_values)
    print("Result for theta values:", result_theta)
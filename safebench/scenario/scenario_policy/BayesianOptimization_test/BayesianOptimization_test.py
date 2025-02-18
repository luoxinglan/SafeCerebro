import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimization:
    def __init__(self, f, bounds, n_init=3):
        """
        初始化贝叶斯优化器
        f: 目标函数
        bounds: 参数边界，形如[(x1_min, x1_max), (x2_min, x2_max),...]
        n_init: 初始随机采样点数量
        """
        self.f = f
        self.bounds = np.array(bounds)
        self.n_init = n_init
        
        # 存储采样点和观测值
        self.X = []
        self.y = []
        
        # 高斯过程的超参数
        self.theta = 1.0
        self.sigma = 1.0
        
    def init_points(self):
        """初始化随机采样点"""
        for _ in range(self.n_init):
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            y = self.f(x)
            self.X.append(x)
            self.y.append(y)
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
    def kernel(self, x1, x2):
        """RBF核函数"""
        return self.sigma * np.exp(-np.sum((x1-x2)**2) / (2*self.theta))
    
    def kernel_matrix(self, X1, X2):
        """计算核矩阵"""
        K = np.zeros((len(X1), len(X2)))
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                K[i,j] = self.kernel(x1, x2)
        return K
    
    def gaussian_process(self, x):
        """高斯过程预测"""
        K = self.kernel_matrix(self.X, self.X)
        K_star = self.kernel_matrix(self.X, [x])
        K_star_star = self.kernel_matrix([x], [x])
        
        # 计算均值和方差
        mu = K_star.T @ np.linalg.inv(K) @ self.y
        sigma = K_star_star - K_star.T @ np.linalg.inv(K) @ K_star
        
        return mu[0], sigma[0,0]
    
    def expected_improvement(self, x):
        """计算期望改进"""
        mu, sigma = self.gaussian_process(x)
        sigma = max(sigma, 1e-9)  # 避免除零错误
        
        y_best = np.min(self.y)
        z = (y_best - mu) / sigma
        ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
        
        return -ei  # 最小化负EI相当于最大化EI
    
    def optimize(self, n_iter=10):
        """执行贝叶斯优化"""
        # 初始化采样点
        self.init_points()
        
        for _ in range(n_iter):
            # 找到下一个采样点
            x_next = None
            ei_max = float('inf')
            
            # 从多个随机起点开始优化EI
            for _ in range(5):
                x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                res = minimize(self.expected_improvement, x0, 
                             bounds=self.bounds,
                             method='L-BFGS-B')
                
                if res.fun < ei_max:
                    x_next = res.x
                    ei_max = res.fun
            
            # 在新的采样点评估目标函数
            y_next = self.f(x_next)
            
            # 更新数据
            self.X = np.vstack((self.X, x_next))
            self.y = np.append(self.y, y_next)
        
        # 返回最优解
        best_idx = np.argmin(self.y)
        return self.X[best_idx], self.y[best_idx]

# 使用示例
def objective_function(x):
    """示例目标函数：x^2"""
    return x[0]**3-x[1]**2+x[2]

# 定义参数范围
bounds = [(-5, 5),(-5,5),(-10,10)]

# 创建优化器实例
optimizer = BayesianOptimization(objective_function, bounds, n_init=3)

# 运行优化
best_x, best_y = optimizer.optimize(n_iter=30)
print(f"找到的最优解: x = {best_x}, f(x) = {best_y}")

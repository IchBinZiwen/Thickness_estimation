import numpy as np
from scipy.optimize import minimize

# 已知矩阵 A
A = np.array([[2, 3],
              [1, 4],
              [5, 6]])

# 参考信号 r
r = np.array([1, 2, 3])

# 初始猜测值 x
x_initial_guess = np.zeros(A.shape[1])

# 定义损失函数 J
def cost_function(x):
    y = np.dot(A, x)
    return np.dot((y - r).T, (y - r))

# 使用 minimize 函数最小化损失函数
result = minimize(cost_function, x_initial_guess, method='BFGS')

# 提取最优解和最小化目标函数值
x_optimal = result.x
J_optimal = result.fun

print("最优解 x:", x_optimal)
print("最小化目标函数值 J(x):", J_optimal)

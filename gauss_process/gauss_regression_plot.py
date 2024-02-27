from matplotlib import pyplot as plt
import japanize_matplotlib
import numpy as np
from gauss_regression import  kernel, gpr

n = 100
data_x = np.linspace(0, 4 * np.pi, n)
data_y = 2 * np.sin(data_x) + 3 * np.cos(2 * data_x) + 5 * np.sin(2/3 * data_x) + np.random.randn(len(data_x))

missing_value_rate = 0.2
sample_index = np.sort(np.random.choice(np.arange(n), int(n * missing_value_rate), replace = False))

xtrain = np.copy(data_x[sample_index])
ytrain = np.copy(data_y[sample_index])

xtest = np.copy(data_x)

mu_pred, var_pred = gpr(xtest, xtrain, ytrain, kernel)

plt.figure(figsize = (12, 5))
plt.title("ガウス過程による予測", fontsize=20)

plt.plot(data_x, data_y, 'x', color = "green", label = "correct signal")
plt.plot(data_x[sample_index], data_y[sample_index], 'o', color = "red", label = "サンプル点")

std = np.sqrt(var_pred)

plt.plot(xtest, mu_pred, color = "blue", label = "ガウス過程の平均（期待値）")
plt.fill_between(xtest, mu_pred + 2 * std, mu_pred - 2 * std, alpha = .2, color = "blue", label = "ガウス過程の標準偏差")
plt.legend()
plt.show()
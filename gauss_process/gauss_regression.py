import numpy as np
from matplotlib import pyplot as plt

# 実行用データ
def data(n, rate):
    n = n
    data_x = np.linspace(0, 4 * np.pi, n)
    data_y = 2 * np.sin(data_x) + 3 * np.cos(2 * data_x) + 5 * np.sin(2/3 * data_x) + np.random.randn(len(data_x))

    missing_value_rate = rate
    sample_index = np.sort(np.random.choice(np.arange(n), int(n * missing_value_rate), replace = False))
    return data_x[sample_index], data_y[sample_index], data_x

# カーネル関数
def kernel(x, x_prime, theta_1, theta_2, theta_3):
    if x == x_prime:
        delta = 1
    else:
        delta = 0

    return theta_1 * np.exp(-1 * (x - x_prime)**2 / theta_2) + theta_3 * delta

# ガウス過程アルゴリズム
def gpr(xtest, xtrain, ytrain, kernel):
    # 平均
    mu = []
    # 分散
    var = []
    # カーネルのパラメータ値
    theta_1 = 1.0
    theta_2 = 0.4
    theta_3 = 0.1

    train_length = len(xtrain)
    K = np.zeros((train_length, train_length))

    for n in range(train_length):
        for m in range(train_length):
            K[n, m] = kernel(xtrain[n], xtrain[m], theta_1, theta_2, theta_3)

    yy = np.dot(np.linalg.inv(K), ytrain)

    test_length = len(xtest)
    for m in range(test_length):
        k = np.zeros((train_length,))
        for n in range(train_length):
            k[n] = kernel(xtrain[n], xtest[m], theta_1, theta_2, theta_3)

        s = kernel(xtest[m], xtest[m], theta_1, theta_2, theta_3)
        mu.append(np.dot(k.T, yy))
        kK_ = np.dot(k.T, np.linalg.inv(K))
        var.append(s - np.dot(kK_, k))
    return mu, var

# 実行用データの定義
xtrain, ytrain, xtest = data(100, 0.2)

xtrain = np.copy(xtrain)
ytrain = np.copy(ytrain)

xtest = np.copy(xtest)

mu_pred, var_pred = gpr(xtest, xtrain, ytrain, kernel)
print("mu_pred:", mu_pred)
print("var_pred:", var_pred)
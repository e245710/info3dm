import numpy as np

class LinearRegression:
    def __init__(self):
        self.x = None
        self.theta = None
        self.y = None

    def fit(self, x, y):
        """
        行列演算を用いてパラメータ theta を求める
        """
        # temp = (X^T * X)^-1
        temp = np.linalg.inv(np.dot(x.T, x))
        # self.theta = temp * X^T * y
        self.theta = np.dot(np.dot(temp, x.T), y)

    def predict(self, x):
        """
        学習した theta を用いて予測値を計算する
        """
        return np.dot(x, self.theta)

    def score(self, x, y):
        """
        残差平方和 (RSS) を計算する
        """
        error = self.predict(x) - y
        return (error**2).sum()
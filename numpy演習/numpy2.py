import numpy2 as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# データ
X = np.array([[0.0], [2.0], [3.9], [4.0]])
Y = np.array([4.0, 0.0, 3.0, 2.0])

# 多項式特徴量（2次）
poly = PolynomialFeatures(degree=2)
X2 = poly.fit_transform(X)

# 描画用のx
samples_x = np.arange(0, 4.1, 0.1)
samples_x2 = poly.fit_transform(samples_x.reshape(-1, 1))

# alphaのリスト
alphas = [0, 0.1, 0.5, 1.0, 10.0]

# グラフ描画
for a in alphas:
    clf = Ridge(alpha=a)
    clf.fit(X2, Y)
    samples_y = clf.predict(samples_x2)
    plt.plot(samples_x, samples_y, label=f'alpha = {a}')

# 元データ
plt.scatter(X, Y, color='black', label='data set')

plt.legend()
plt.title("Ridge Regression (alpha comparison)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
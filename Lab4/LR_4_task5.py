import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X**2 + X + 2 + np.random.randn(m, 1)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

print("Лінійна регресія:")
print("  intercept =", lin_reg.intercept_[0])
print("  coef =", lin_reg.coef_[0])

print("  MAE =", mean_absolute_error(y, y_lin_pred))
print("  MSE =", mean_squared_error(y, y_lin_pred))
print("  R2  =", r2_score(y, y_lin_pred), "\n")

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)
y_poly_pred = lin_reg_poly.predict(X_poly)

print("Поліноміальна регресія:")
print("  intercept =", lin_reg_poly.intercept_[0])
print("  coef =", lin_reg_poly.coef_[0])

print("  MAE =", mean_absolute_error(y, y_poly_pred))
print("  MSE =", mean_squared_error(y, y_poly_pred))
print("  R2  =", r2_score(y, y_poly_pred), "\n")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label="Дані")

plt.plot(X, y_lin_pred, color='orange', linewidth=2, label="Лінійна регресія")

sort_idx = np.argsort(X[:, 0])
plt.plot(X[sort_idx], y_poly_pred[sort_idx], color='red', linewidth=2, label="Поліноміальна регресія (2 степінь)")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Лінійна та поліноміальна регресія (Варіант 2)")
plt.legend()
plt.grid(True)
plt.show()

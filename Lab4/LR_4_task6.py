import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.6 * X**2 + X + 2 + np.random.randn(m, 1)

def plot_learning_curves(model, X, y, title):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_errors, val_errors = [], []

    for m in range(2, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)

        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.figure(figsize=(9, 5))
    plt.plot(np.sqrt(train_errors), "r-", linewidth=2, label="Помилка на тренуванні")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=2, label="Перевірочна помилка")
    plt.title(title)
    plt.xlabel("Кількість тренувальних зразків")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    plt.show()

lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, "Криві навчання: Лінійна регресія")

poly10 = PolynomialFeatures(degree=10, include_bias=False)
X_poly10 = poly10.fit_transform(X)

poly_reg_10 = LinearRegression()
plot_learning_curves(poly_reg_10, X_poly10, y, "Криві навчання: Поліноміальна регресія (10-й ступінь)")

poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly2 = poly2.fit_transform(X)

poly_reg_2 = LinearRegression()
plot_learning_curves(poly_reg_2, X_poly2, y, "Криві навчання: Поліноміальна регресія (2-й ступінь)")

import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

input_file = 'data_multivar_regr.txt'

data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)

y_test_pred_linear = linear_regressor.predict(X_test)

print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_linear), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_linear), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_linear), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred_linear), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred_linear), 2))

polynomial = PolynomialFeatures(degree=10)
X_train_poly = polynomial.fit_transform(X_train)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_poly, y_train)

datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.transform(datapoint)

linear_pred_point = linear_regressor.predict(datapoint)
poly_pred_point = poly_linear_model.predict(poly_datapoint)

print("\nPrediction for datapoint", datapoint)
print("Linear regression:", round(linear_pred_point[0], 2))
print("Polynomial regression:", round(poly_pred_point[0], 2))

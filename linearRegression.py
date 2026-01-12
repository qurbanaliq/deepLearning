"""Linear Regression, predicting house prices from features like size and number of bedrooms
"""

import numpy as np

# support we have 5 houses with 2 features:
# [size in 1000 sqft, number of bedrooms]
X = np.array([
    [2.0, 3],
    [2.5, 4],
    [3.0, 3],
    [3.2, 5],
    [4.0, 4]
])

# prices in $100k
y = np.array([250, 300, 320, 360, 400])

# add column of ones to X for bias term
X_aug = np.c_[X, np.ones(X.shape[0])]

print("X_aug:", X_aug)

# compute weights using normal equation
w = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
print("weights:", w)

# make predictions
y_pred = X_aug @ w
print("predictions:", y_pred)
print("Actual:", y)

# we can compute mean squared error
print("y_pred - y:", y_pred - y)
print("(y_pred - y)**2:", (y_pred - y)**2)
mse = np.mean((y_pred - y)**2)
print("mse:", mse)
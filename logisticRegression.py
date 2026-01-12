import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

X = np.hstack([X, np.ones((X.shape[0], 1))])
print("X with bias:", X)

y = np.array([0, 0, 0, 1], dtype=float)

print("X.shpe:", X.shape)
print("y.shape", y.shape)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

theta = np.zeros(X.shape[1])
print("X.shape:", X.shape)
print("theta:", theta)
# learning rate
lr = 1.0
epochs = 100

for _ in range(epochs):
    z = np.dot(X, theta)
    predictions = sigmoid(z)

    # gradient of loss wrt theta
    gradient = np.dot(X.T, predictions - y) / len(y)

    # update rule
    theta -= lr * gradient
    print("learning theta:", theta)

print("theta trained:", theta)

# step 5: predictions
probs = sigmoid(np.dot(X, theta))
preds = (probs >= 0.5).astype(int)

print("Learning Weights:", theta)
print("Predicted probabilities:", probs)
print("Predicted classes:", preds)
"""using scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#1 create a non-linear dataset
X = np.array([
    [-1, 1], # +1
    [1, 1],  # +1
    [0, 0]   # -1
])

y = np.array([1, 1, -1])

#2 train an SVM with an RBF kernel
model = svm.SVC(kernel="rbf", C=100, gamma=1)
model.fit(X, y)

#3 create a grid of points to visulize decision boundry
xx, yy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-1, 2, 200))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#4 plot everything
plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap="coolwarm")
plt.contour(xx, yy, Z, levels=[0], colors="k", linewidths=2) # boundary

# plot points
plt.scatter(X[:,0], X[:, 1], c=y, s=100, cmap="coolwarm", edgecolors="k")

# highlight support vectors
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=200, facecolors="none", edgecolors="yellow", linewidths=2, label="Support Vectors")

plt.title("Non-linear SVM using RBF Kernel")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
"""


"""using explicit code to code svm
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. kernel function
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def rbf_kernel(x1, x2, gamma=1.0):
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

# 2. SVM implementation
class SimpleSVM():
    def __init__(self, kernel=rbf_kernel, C=1.0, tol=1e-4, max_passes=5, gamma=1.0):
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.gamma = gamma

    def fit(self, X, y):
        n_samples, n_features = X.shape
        print("n_samples:", n_samples)
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_samples)
        self.b = 0.0

        # compute the full kernel matrix
        self.K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.K[i, j] = self.kernel(X[i], X[j], self.gamma)

        # sequential minimal optimization algorithm
        passes = 0
        while passes < self.max_passes:
            num_changed_alpha = 0
            for i in range(n_samples):
                f_i = np.sum(self.alpha * y * self.K[:, i]) + self.b
                E_i = f_i - y[i]

                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    # pick a random j != i
                    j = np.random.choice([n for n in range(n_samples) if n != i])
                    
                    f_j = np.sum(self.alpha * y * self.K[:, j]) + self.b
                    E_j = f_j - y[j]

                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    # compute bounds L and H
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] -self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j], - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    #TODO: figure out why it's always True
                    if L == H:
                        continue

                    # compute eta
                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue

                    # update alpha_j
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    # check if change in alpha is significant
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # update alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    # compute b1 and b2 thresholds
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] \
                        - y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] \
                        - y[j] * (self.alpha[j], alpha_j_old) * self.K[j, j]
                    
                    # update bias b
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alpha += 1
                
            if num_changed_alpha == 0:
                passes += 1
            else:
                passes = 0
        print("support vectors:", self.alpha)
        # save support vectors
        self.support_vectors_ = X[self.alpha > 1e-5]
        self.support_y = y[self.alpha > 1e-5]
        self.support_alpha = self.alpha[self.alpha > 1e-5]
    
    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for alpha, y_sv, x_sv in zip(self.support_alpha, self.support_y, self.support_vectors_):
                s += alpha * y_sv * self.kernel(X[i], x_sv, self.gamma)
            y_predict[i] = s
        return y_predict + self.b
    
    def predict(self, X):
        return np.sign(self.project(X))

# training and visualization
X = np.array([
    [1, 2], [2, 3], [2, 0],
    [3, 2], [2, -2], [3, -1]
])
y = np.array([1, 1, 1, -1, -1, -1])

svm = SimpleSVM(kernel=lambda x, y, g: np.exp(-0.5 * np.linalg.norm(x - y)**2), C=1.0, gamma=1.0)
svm.fit(X, y)

# plot
xx, yy = np.meshgrid(np.linspace(-1, 5, 200), np.linspace(-3, 5, 200))
Z = svm.project(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap="coolwarm")
plt.contour(xx, yy, Z, levels=[0], colors="k", linewidths=2)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap="coolwarm", edgecolors="k")
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=200, facecolors="none",
            edgecolors="yellow", linewidths=2)
plt.title("SVM from Scratch (RBF Kernel)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
import numpy as np
import gzip
import struct
import os

base_path = "C:\\Users\\qurba\\Documents\\Datasets\\MNIST"

# load the dataset
def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        # magic number, number of images, rows, cols
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        print("Loading images:", filename, "count:", num_images)
        print("magic:", magic)
        print("rows, columns:", rows, cols)
        
        # Read the remaining bytes and reshape
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
        
        # Normalize to [0, 1]
        data = data.astype(np.float32) / 255.0
        return data

def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # magic number, number of labels
        magic, num_labels = struct.unpack(">II", f.read(8))
        print("Loading labels:", filename, "count:", num_labels)
        
        # Each label is just one byte
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels.astype(np.int64)


# Load the full dataset
X_train = load_images(os.path.join(base_path, "train-images-idx3-ubyte.gz"))
y_train = load_labels(os.path.join(base_path, "train-labels-idx1-ubyte.gz"))

X_test = load_images(os.path.join(base_path, "t10k-images-idx3-ubyte.gz"))
y_test = load_labels(os.path.join(base_path, "t10k-labels-idx1-ubyte.gz"))

print("Shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# build the mlp (forward pass)

# --- Xavier initialization ---
def xavier_init(size_in, size_out):
    limit = np.sqrt(6 / (size_in + size_out))
    return np.random.uniform(-limit, limit, (size_out, size_in))

# --- Model parameters ---
W1 = xavier_init(784, 128)
b1 = np.zeros((128, 1))

W2 = xavier_init(128, 10)
b2 = np.zeros((10, 1))

print("W1 Shape:", W1.shape)
print("W2 Shape:", W2.shape)

# --- Activations ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)

# --- Forward pass ---
def forward(x):
    # x shape: (784, batch_size)

    z1 = W1 @ x + b1
    h1 = relu(z1)

    z2 = W2 @ h1 + b2
    out = softmax(z2)

    return z1, h1, z2, out

# cross entropy

def cross_entropy_loss(y_true, y_pred):
    """
    y_true: one-hot labels (10, batch)
    y_pred: predicted probs from softmax (10, batch)
    """
    eps = 1e-12                   # avoid log(0)
    log_probs = np.log(y_pred + eps)
    loss = -np.sum(y_true * log_probs) / y_true.shape[1]
    return loss

# backprop
def backward(x, y_true, z1, h1, z2, y_pred, lr):
    """
    x:      (784, batch)
    y_true: (10, batch) one-hot labels
    h1:     (128, batch)
    y_pred: (10, batch)
    """

    global W1, b1, W2, b2

    batch_size = x.shape[1]

    # --- Output layer gradient ---
    # derivative of cross-entropy with softmax:
    # dL/dz2 = y_pred - y_true
    dz2 = (y_pred - y_true) / batch_size

    dW2 = dz2 @ h1.T       # (10, batch) @ (batch, 128) = (10, 128)
    db2 = np.sum(dz2, axis=1, keepdims=True)  # (10, 1)

    # --- Backprop into hidden layer ---
    dh1 = W2.T @ dz2                       # (128,10) @ (10,batch) = (128,batch)
    dz1 = dh1 * (z1 > 0).astype(float)     # ReLU derivative

    dW1 = dz1 @ x.T                         # (128,batch) @ (batch,784) = (128,784)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    # --- SGD update ---
    W2 -= lr * dW2
    b2 -= lr * db2

    W1 -= lr * dW1
    b1 -= lr * db1


# training loop
# --- Training settings ---
lr = 0.1
batch_size = 64
epochs = 5     # start small to confirm it's working

def one_hot(labels, num_classes=10):
    out = np.zeros((labels.size, num_classes))
    out[np.arange(labels.size), labels] = 1
    return out.T   # shape: (10, batch)

# Convert dataset:
X_train = X_train.reshape(-1, 784).T      # (784, N)
y_train_oh = one_hot(y_train)                    # (10, N)

N = X_train.shape[1]

for epoch in range(epochs):
    # shuffle dataset
    perm = np.random.permutation(N)
    print("{epoch} - perm:", perm)
    X_train = X_train[:, perm]
    y_train_oh = y_train_oh[:, perm]

    total_loss = 0
    correct = 0

    for i in range(0, N, batch_size):
        x = X_train[:, i:i+batch_size]          # (784, batch)
        y = y_train_oh[:, i:i+batch_size]       # (10, batch)
        labels = y_train[i:i+batch_size] if False else None

        # forward
        z1, h1, z2, out = forward(x)

        # loss
        loss = cross_entropy_loss(y, out)
        total_loss += loss * x.shape[1]

        # accuracy
        preds = np.argmax(out, axis=0)
        true_labels = np.argmax(y, axis=0)
        correct += np.sum(preds == true_labels)

        # backward + update
        backward(x, y, z1, h1, z2, out, lr)

    avg_loss = total_loss / N
    accuracy = correct / N * 100

    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")


# --- Prepare test set ---
X_test_flat = X_test.reshape(-1, 784).T      # shape: (784, N_test)
y_test_labels = y_test                                 # shape: (N_test,)

# one-hot is not needed for evaluation


# --- Forward pass through the entire test set ---
_, _, _, test_probs = forward(X_test_flat)

test_preds = np.argmax(test_probs, axis=0)

test_accuracy = np.mean(test_preds == y_test_labels) * 100

print(f"Test Accuracy: {test_accuracy:.2f}%")

import matplotlib.pyplot as plt

def show_example(i):
    img = X_test[i].reshape(28, 28)
    label = y_test[i]

    plt.imshow(img, cmap="gray")
    plt.title(f"True label: {label}, Predicted: {test_preds[i]}")
    plt.axis("off")
    plt.show()

show_example(5)
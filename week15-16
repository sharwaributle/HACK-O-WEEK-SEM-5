
import pickle
import joblib
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==================================================
# 1. MODEL DEPLOYMENT: SAVE AND LOAD
# ==================================================

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

print("Original Model Accuracy:",
      accuracy_score(y_test, model.predict(X_test)))

# Save using Pickle
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

# Load using Pickle
with open("model.pkl", "rb") as f:
    loaded = pickle.load(f)

print("Pickle Model Accuracy:",
      accuracy_score(
          y_test,
          loaded["model"].predict(loaded["scaler"].transform(data.data[
              train_test_split(
                  np.arange(len(y)), y, test_size=0.2,
                  random_state=42, stratify=y
              )[1]
          ]))
      ))

# Save using Joblib
joblib.dump({"model": model, "scaler": scaler}, "model.joblib")

# Load using Joblib
loaded_joblib = joblib.load("model.joblib")
print("Joblib model loaded successfully!")


# ==================================================
# 2. PERCEPTRON AND ACTIVATION FUNCTIONS
# ==================================================

def perceptron(x, weights, bias):
    return np.dot(x, weights) + bias

def step(z):
    return np.where(z >= 0, 1, 0)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

x = np.array([1.0, 0.0])
weights = np.array([0.7, -0.4])
bias = -0.2

z = perceptron(x, weights, bias)

print("\n--- ACTIVATION FUNCTIONS ---")
print("Weighted Sum:", z)
print("Step:", step(z))
print("Sigmoid:", sigmoid(z))
print("ReLU:", relu(z))
print("Tanh:", tanh(z))


# ==================================================
# 3. LOSS FUNCTIONS
# ==================================================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

y_true = np.array([1, 0, 1])
y_pred = np.array([0.9, 0.2, 0.7])

print("\n--- LOSS FUNCTIONS ---")
print("MSE:", mse(y_true, y_pred))
print("Binary Cross Entropy:",
      binary_cross_entropy(y_true, y_pred))


# ==================================================
# 4. SGD OPTIMIZER
# ==================================================

def sgd_update(weights, gradients, learning_rate=0.1):
    return weights - learning_rate * gradients

weights = np.array([2.0, -1.0])
gradients = np.array([0.5, -0.25])

print("\n--- SGD ---")
print("Before Update:", weights)
print("After Update:", sgd_update(weights, gradients))


# ==================================================
# 5. NEURAL NETWORK FROM SCRATCH USING NUMPY
# XOR CLASSIFICATION
# ==================================================

np.random.seed(42)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

Y = np.array([[0], [1], [1], [0]], dtype=float)

input_neurons = 2
hidden_neurons = 8
output_neurons = 1

W1 = np.random.randn(input_neurons, hidden_neurons) * np.sqrt(2 / input_neurons)
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons) * np.sqrt(1 / hidden_neurons)
b2 = np.zeros((1, output_neurons))


def forward(X):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    return A1, A2, Z1


def calculate_loss(Y, predictions):
    eps = 1e-8
    predictions = np.clip(predictions, eps, 1 - eps)
    return -np.mean(
        Y * np.log(predictions)
        + (1 - Y) * np.log(1 - predictions)
    )


# Adam optimizer state
parameters = [W1, b1, W2, b2]
m = [np.zeros_like(p) for p in parameters]
v = [np.zeros_like(p) for p in parameters]

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
learning_rate = 0.03
t = 0


def adam_update(parameters, gradients):
    global m, v, t
    t += 1

    for i in range(len(parameters)):
        m[i] = beta1 * m[i] + (1 - beta1) * gradients[i]
        v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] ** 2

        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)

        parameters[i] -= (
            learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        )


print("\n--- TRAINING NEURAL NETWORK ---")

for epoch in range(10001):

    # Forward propagation
    A1, A2, Z1 = forward(X)
    loss = calculate_loss(Y, A2)

    # Backpropagation
    n = len(X)

    dZ2 = (A2 - Y) / n
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update weights using Adam
    adam_update(
        parameters,
        [dW1, db1, dW2, db2]
    )

    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")


# Final predictions
_, predictions, _ = forward(X)
predicted_classes = (predictions >= 0.5).astype(int)

print("\n--- FINAL RESULTS ---")
print("Input:")
print(X.astype(int))

print("Predicted Probabilities:")
print(np.round(predictions, 3))

print("Predicted Classes:")
print(predicted_classes.ravel())

print("Actual Classes:")
print(Y.ravel().astype(int))

accuracy = np.mean(predicted_classes == Y) * 100
print("Training Accuracy:", accuracy, "%")

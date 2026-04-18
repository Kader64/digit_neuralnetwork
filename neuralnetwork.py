import numpy as np


class NeuronLayer:
    def __init__(self, input_size, output_size, activation):
        self.W = np.random.rand(output_size, input_size) - 0.5
        self.b = np.random.rand(output_size, 1) - 0.5
        self.activation = activation

        self.Z = None
        self.A = None

    def forward(self, A_prev):
        self.Z = self.W.dot(A_prev) + self.b
        self.A = self.activate(self.Z)
        return self.A

    def activate(self, Z):
        if self.activation == "relu":
            return np.maximum(Z, 0)
        elif self.activation == "softmax":
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return expZ / np.sum(expZ, axis=0, keepdims=True)
        return Z

    def activation_derivative(self, Z):
        if self.activation == "relu":
            return Z > 0
        return np.ones_like(Z)


class NeuralNetwork:
    def __init__(self):
        self.layers = [
            NeuronLayer(784, 64, "relu"),
            NeuronLayer(64, 32, "relu"),
            NeuronLayer(32, 10, "softmax")
        ]

    def forward(self, A):
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    def backward(self, X, Y):
        m = X.shape[1]
        one_hot_Y = self.one_hot(Y)

        grads = []

        # Output layer error
        A_last = self.layers[-1].A
        dZ = A_last - one_hot_Y

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            A_prev = X if i == 0 else self.layers[i - 1].A

            dW = (1 / m) * dZ.dot(A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            grads.insert(0, (dW, db))

            if i != 0:
                prev_layer = self.layers[i - 1]
                dZ = layer.W.T.dot(dZ) * prev_layer.activation_derivative(prev_layer.Z)

        return grads

    def update(self, grads, alpha):
        for layer, (dW, db) in zip(self.layers, grads):
            layer.W -= alpha * dW
            layer.b -= alpha * db

    def train(self, X, Y, alpha=0.2, iterations=500):
        X = X.T

        for i in range(1, iterations + 1):
            A = self.forward(X)
            grads = self.backward(X, Y)
            self.update(grads, alpha)

            if i % 50 == 0:
                preds = self.predict(X)
                acc = self.accuracy(preds, Y)
                print(f"Iteration {i}, Accuracy: {acc:.4f}")

    def predict(self, X):
        if X.shape[0] != 784:
            X = X.T
        A = self.forward(X)
        return np.argmax(A, axis=0)

    def accuracy(self, preds, Y):
        return np.sum(preds == Y) / Y.size

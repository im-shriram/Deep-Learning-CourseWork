from math import exp
import numpy as np

class Perceptron:
    def __init__(self, size, epochs, lr) -> None:
        # Initializing weights with random values
        self.weights = np.random.random(size=[size, 1]) # n * 1
        self.bias = np.zeros(shape=[1, 1])

        # Gradient descent attributes
        self.lr = lr
        self.epochs = epochs

    def predict(self, X_train):
        z = np.dot(X_train, self.weights) + self.bias # n * 1
        y_pred = 1 / (1 + np.exp(-z))
        return y_pred

    def train(self, X_train, y_train):
        for i in range(self.epochs):
            y_pred = self.predict(X_train)
            self.weights -= self.lr * np.dot(a = X_train.T, b = (y_pred - y_train))
            self.bias -= self.lr * np.sum(y_pred - y_train)
        
            if i % 100 == 0:
                y_pred = self.predict(X_train)
                loss = -np.mean(np.dot(y_train.T, np.log(y_pred)) + np.dot(1 - y_train.T, np.log(1 - y_pred)))
                print(f"BCE Loss at {i}th epoch = {loss}")

def main() -> None:
    # Binary classification dataset (y must be -1 or +1)
    np.random.seed(42)
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=float)
    y = np.where(X > 5, 1.0, 0.0) # labels: -1 or +1

    model = Perceptron(size=X.shape[1], epochs=2500, lr=0.01)
    model.train(X, y)

    print("\nLearned weight:", model.weights.flatten())
    print("Learned bias:", model.bias)
    print("Predictions:", np.where(model.predict(X) >= 0.5, 1, 0))

if __name__ == "__main__":
    main()
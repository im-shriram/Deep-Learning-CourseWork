import numpy as np

class Perceptron:
    def __init__(self, X_train, epochs, lr):
        # Initializing weights with random values
        self.parameters = np.random.randn(X_train.shape[1], 1)
        self.bias = 0.0

        # Gradient Descent parameters
        self.epochs = epochs
        self.lr = lr

    def predict(self, X):
        return np.dot(X, self.parameters) + self.bias # (n, 1)

    def train(self, X_train, y_train):
        n = X_train.shape[0]

        for epoch in range(self.epochs):
            scores = self.predict(X_train) # (n, 1)
            margin = y_train * scores # (n, 1)

            # Hinge loss gradient mask: only update misclassified points
            mask = (margin < 1).astype(float) # (n, 1)

            # Batch gradient of hinge loss
            dw = -np.dot(X_train.T, mask * y_train) / n # (c, 1)
            db = -np.sum(mask * y_train) / n

            self.parameters -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 100 == 0: # After every 100th epoch we are calculating the loss
                loss = np.mean(np.maximum(0, 1 - margin))

                # Hinge Loss
                term = -1 * y_train * np.where(scores >= 0, 1, -1)
                hinge_loss = np.sum(
                    np.where(term > 0, term , 0)
                )

                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                print(f"Epoch {epoch}, Hinge Loss: {hinge_loss:.4f}")

def main():
    # Binary classification dataset (y must be -1 or +1)
    np.random.seed(42)
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=float)
    y = np.where(X > 5, 1.0, -1.0)  # labels: -1 or +1

    model = Perceptron(X_train=X, epochs=2000, lr=0.01)
    model.train(X, y)

    print("\nLearned weight:", model.parameters.flatten())
    print("Learned bias:", model.bias)
    print("Predictions:", np.sign(model.predict(X)).flatten())

if __name__ == "__main__":
    main()
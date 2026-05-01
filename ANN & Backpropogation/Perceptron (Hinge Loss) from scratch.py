"""
    Confusion: Step function acts as an activation function, but the differentiation of the step function is not mentioned in the gradient equation.

        —→ The step function is not an activation function. It is used to transform the probabilities (i.e., model output) into class labels (i.e., 1 or 0).

        Your model output: 0.93, 0.38 (Model Probabilities).
            These probabilities (given by the sigmoid activation) are used to calculate gradients, not after those probability values are transformed into (-1, 1) using the step function.

        —→ If the probability is 0.85, there is some loss because of which the probability is off by 0.15. Your model needs to reduce that loss. However, if you transform it to 1 since 0.85 > 0.5, your model considers that it perfectly classified that sample, which is not actually the case.

        —→ The step function is used after the model is trained, so there is no impact of the step function in the gradient equation.

            model.predict(X_train) → step(model_probabilities)
            model.fit(X_train, y_train) → sigmoid(z)
        
    Note: The generalized equation of the step function is used in the geometric intuition of the Perceptron notebook.
"""

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
        """
            I have'nt usde any step function that converts to (-1 or 1) thats why there is no differentation term wrt step function in the gradient equation.
        """
        return np.dot(X, self.parameters) + self.bias # (n, 1)

    def train(self, X_train, y_train):
        n = X_train.shape[0]

        for epoch in range(self.epochs):
            scores = self.predict(X_train) # (n, 1)
            margin = y_train * scores # (n, 1) → [-1, -1, 1, -1, 1 ....]

            # Hinge loss gradient mask: only update misclassified points
            """
                1 * 1 (classified) → 1
                -1 * -1 (classified) → 1
                1 * -1 (missclassified) → -1
                -1 * 1 (missclassified) → -1

                We want -1 (missclassified records) → Thats why margin < 1
            """
            mask = (margin < 1).astype(float) # (n, 1)

            """
                e.g. values in masks → [1, -1, 0, -1, 1, 0]
                    0 - classified → belongs to 1 (classified)
                    1 - missclassified → belongs to -1 (missclassified)
            """

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
    
    def evaluate(self, X_test: np.array, y_test):
        y_pred = self.predict(X = X_test)
        y_pred_class = np.where(y_pred >= 0, 1, -1)

        accuracy = (y_test == y_pred_class).sum() / len(X_test)
        return accuracy


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
    
    print(f"Training Accuracy = {model.evaluate(X_test=X, y_test=y)}")

if __name__ == "__main__":
    main()
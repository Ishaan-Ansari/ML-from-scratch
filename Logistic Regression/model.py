"""
Workflow of the Logistic Regression Model:

Step 1: Set the learning rate & number of iterations. Initiate random weight and bias values.

Step 2: Build the logistic regression function (using the sigmoid function).

Step 3: Update the parameters using gradient descent.

Finally, we will get the best model (optimal weight and bias values) as it has the minimum cost function.

Step 4: Build the "predict" function to determine the class of the data point.
"""
import numpy as np

class LogisticRegression:
    """Logistic Regression using Gradient Descent"""

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations 
        self.weights = None
        self.bias = 0
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost(self, y, y_pred):
        cost = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return cost
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for _ in range(self.iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            self.cost_history.append(self.cost(y, y_pred))

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        y_pred_labels = [1 if i > 0.5 else 0 for i in y_pred]
        return np.array(y_pred_labels)
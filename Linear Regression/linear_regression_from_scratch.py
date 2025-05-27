import numpy as np

class Linear_Regression:
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        # training features
        self.m, self.n = X.shape # (m) rows & (n) columns

        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weights()

        return self


    def update_weights(self,):
        Y_prediction = self.predict(self.X)

        # calculate gradients        
        dw = -(2*(self.X.T).dot(self.Y - Y_prediction))/self.m

        db = -2*np.sum(self.Y - Y_prediction)/self.m

        # updating the weights
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db

    def predict(self, X):
        return X.dot(self.w)+self.b
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.priors = None
        self.mean = None
        self.variance = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes)
        
        # intialize attributes
        self.mean = np.zeros((n_classes, n_features))
        self.variance = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i, :] = X_c.mean(axis=0)
            self.variance[i, :] = X_c.var(axis=0) + 1e-9
            self.priors[i] = X_c.shape[0] / X.shape[0]

    def _gaussian_pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        variance = self.variance[class_idx]
        numerator = np.exp(-((x - mean)**2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            likelihood = np.sum(np.log(self._gaussian_pdf(i, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
if __name__ == "__main__":
    X = np.random.rand(100, 2)*10  # 100 samples, 2 features
    y = np.where(X[:, 0] + X[:, 1] > 10, 1, 0) # Create a simple classification rule

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = NaiveBayes()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(predictions)
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
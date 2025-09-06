import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate a synthetic dataset
np.random.seed(0)
X = np.random.rand(100, 2)*10  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 10).astype(int)  # Binary target based on a linear combination

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for better convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Import the LogisticRegression class from model.py
from log-regress import LogisticRegression

# Train and evaluate the model
model = LogisticRegression(learning_rate=0.1, iterations=1000)
model.fit(X_train, y_train)

# Evaluate accuracy
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Plotting the cost function over iterations
plt.plot(model.cost_history)
plt.title("Cost Function over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid()
plt.show()
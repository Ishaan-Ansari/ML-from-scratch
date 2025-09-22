import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pdb; pdb.set_trace()

class SVM:
    """
    Support Vector Machine using the Sequential Minimal Optimization (SMO) algorithm.
    """
    def __init__(self, C=1.0, kernel='linear', gamma=1.0, degree=3, coef0=1.0, max_iter=1000, tol=1e-3):
        """
        Initializes the SVM model.

        Parameters:
        C (float): Regularization parameter. The strength of the regularization is
            inversely proportional to C. Must be strictly positive. The penalty
            is given by C * x^2 when x is the distance between the current prediction
            and the target.
        kernel (str): Kernel type ('linear', 'rbf', 'poly')
        gamma (float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        degree (int): Degree of the polynomial kernel function ('poly'). Ignored by all other
            kernels.
        coef0 (float): Independent term for the polynomial kernel function. 
        max_iter (int): Hard limit on iterations within a single optimization run.
        tol (float): Tolerance for stopping criteria. 
            """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol

    def _kernel_function(self, x1, x2):
        """
        Computes the kernel function between two samples.

        Parameters:
        x1 (array-like): First sample.
        x2 (array-like): Second sample.

        Returns:
        float: Kernel value.
        """
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")
        
    def _compute_kernel_matrix(self, X1, X2=None):
        """
        Computes the kernel matrix between two datasets.

        Parameters:
        X1 (array-like): First dataset.
        X2 (array-like): Second dataset. If None, uses X1.

        Returns:
        array-like: Kernel matrix.
        """
        if X2 is None:
            X2 = X1
        
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_function(X1[i], X2[j])
        
        return K
    
    def _compute_error(self, i):
        """
        Computes the error for a given sample.

        Parameters:
        i (int): Index of the sample.

        Returns:
        float: Error value.
        """
        f_xi = np.sum(self.alphas * self.y * self.K[:, i]) + self.b
        return f_xi - self.y[i]
    
    def _select_second_alpha(self, i1):
        """
        Selects the second alpha using heuristic.
        """
        E1 = self._compute_error(i1)

        max_diff = 0
        i2 = -1

        for i in range(self.n_samples):
            if self.alphas[i] > 0 and self.alphas[i] < self.C:
                E2 = self._compute_error(i)
                diff = abs(E1 - E2)                                         
                if diff > max_diff:
                    max_diff = diff
                    i2 = i
            
        # If no non-bound alpha found, select randomly
        if i2 == -1:
            candidates = list(range(self.n_samples))
            candidates.remove(i1)
            i2 = np.random.choice(candidates)

        return i2
    
    def _clip_alpha(self, alpha, L, H):
        """
        Clips alpha to be within [L, H].
        """
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha
        
    def _smo_step(self, i1, i2):
        """
        Perform one step of SMO algorithm
        """
        if i1 == i2:
            return False
        
        # Get current alphas and labels
        alpha1, alpha2 = self.alphas[i1], self.alphas[i2]
        y1, y2 = self.y[i1], self.y[i2]

        # Compute errors
        E1 = self._compute_error(i1)
        E2 = self._compute_error(i2)

        # Compute L and H
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)

        if L == H:
            return False
        
        # Compute kernel values
        K11 = self.K[i1, i1]
        K22 = self.K[i2, i2]
        K12 = self.K[i1, i2]

        # Compute second derivative of objective function
        eta = K11 + K22 - 2 * K12

        if eta > 0:
            # Compute new alpha2
            alpha2_new = alpha2 + y2 * (E1 - E2) / eta
            alpha2_new = self._clip_alpha(alpha2_new, L, H)

        else:
            # Compute objective function at L and H
            f1 = y1 * (E1 + self.b) - alpha1 * K11 - y2 * alpha2 * y2 * K12
            f2 = y2 * (E2 + self.b) - alpha2 * K22 - y1 * alpha1 * y2 * K12

            L1 = alpha1 + y1 * y2 * (alpha2 - L)
            H1 = alpha1 + y1 * y2 * (alpha2 - H)

            Lobj = L1 * f1 + L * f2 + 0.5 * L1 * L1 * K11 + 0.5 * L * L * K22 + y1 * y2 * L * L1 * K12
            Hobj = H1 * f1 + H * f2 + 0.5 * H1 * H1 * K11 + 0.5 * H * H * K22 + y1 * y2 * H * H1 * K12

            if Lobj < Hobj - self.tol:
                alpha2_new = L
            elif Lobj > Hobj + self.tol:
                alpha2_new = H
            else:
                alpha2_new = alpha2

        # Check if change is significant
        if abs(alpha2_new - alpha2) < self.tol * (alpha2_new + alpha2 + self.tol):
            return False
        
        # Compute new alpha1
        alpha1_new = alpha1 + y1 * y2 * (alpha2 - alpha2_new)

        # Update alphas
        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new

        # Compute new bias
        b1 = E1 + y1 * (alpha1_new - alpha1) * K11 + y2 * (alpha2_new - alpha2) * K12 + self.b
        b2 = E2 + y1 * (alpha1_new - alpha1) * K12 + y2 * (alpha2_new - alpha2) * K22 + self.b

        if 0 < alpha1_new < self.C:
            self.b = b1
        elif 0 < alpha2_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        return True
    

    def fit(self, X, y):
        """Train the SVM using SMO algorithm."""
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape

        # Initialize parameters
        self.alphas = np.zeros(self.n_samples)
        self.b = 0.0

        # Compute kernel matrix
        self.K = self._compute_kernel_matrix(X)

        # Main SMO loop
        num_changed = 0
        examine_all = True
        iteration = 0

        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            if examine_all:
                for i in range(self.n_samples):
                    i2 = self._select_second_alpha(i)
                    if self._smo_step(i, i2):
                        num_changed += 1
            else:
                for i in range(self.n_samples):
                    if 0 < self.alphas[i] < self.C:
                        i2 = self._select_second_alpha(i)
                        if self._smo_step(i, i2):
                            num_changed += 1

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            iteration += 1

        # Store support vectors
        self.support_vectors_indices = np.where(self.alphas > self.tol)[0]
        self.support_vectors = X[self.support_vectors_indices]
        self.support_vector_labels = y[self.support_vectors_indices]
        self.support_vector_alphas = self.alphas[self.support_vectors_indices]

        print(f"Training completed in {iteration} iterations.")

        return self
    
    def predict(self, X):
        """Make predictions on new data."""
        if not hasattr(self, 'support_vectors'):
            raise ValueError("Model not trained yet. Please call 'fit' before 'predict'.")
        
        predictions = []

        for x in X:
            decision = 0
            for i, sv in enumerate(self.support_vectors):
                decision += (self.support_vector_alphas[i] * self.support_vector_labels[i] *
                             self._kernel_function(x, sv))
                
            decision += self.b

            predictions.append(1 if decision >= 0 else -1)

        return np.array(predictions)
    
    def decision_function(self, X):
        """Compute the decision function values"""
        if not hasattr(self, 'support_vectors'):
            raise ValueError("Model not trained yet. Please call 'fit' before 'decision_function'.")
        
        decisions = []

        for x in X:
            decision = 0
            for i, sv in enumerate(self.support_vectors):
                decision += (self.support_vector_alphas[i] * self.support_vector_labels[i] *
                             self._kernel_function(x, sv))
                
            decision += self.b

            decisions.append(decision)

        return np.array(decisions)
    
    def score(self, X, y):
        """Compute the accuracy of the model."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
def plot_svm_decision_boundary(svm, X, y, title="SVM Decision Boundary"):
    """Plot SVM decision boundary and support vectors"""
    plt.figure(figsize=(10, 8))
    
    # Create a mesh to plot the decision boundary
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.decision_function(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], 
               linestyles=['--', '-', '--'], colors=['red', 'black', 'red'])
    plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=50)
    
    # Highlight support vectors
    if hasattr(svm, 'support_vectors'):
        plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1], 
                   s=100, facecolors='none', edgecolors='black', linewidths=2,
                   label=f'Support Vectors ({len(svm.support_vectors)})')
    
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def demo_svm():
    """Demonstrate SVM with different kernels"""
    
    # Generate sample data
    np.random.seed(42)
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, 
                     random_state=42, cluster_std=1.5)
    y[y == 0] = -1  # Convert to -1, 1 labels
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Test different kernels
    kernels = [
        ('Linear', 'linear', {}),
        ('RBF', 'rbf', {'gamma': 1.0}),
        ('Polynomial', 'poly', {'degree': 3, 'gamma': 1.0})
    ]
    
    for name, kernel, params in kernels:
        print(f"\n=== {name} Kernel SVM ===")
        
        # Create and train SVM
        svm = SVM(C=1.0, kernel=kernel, **params)
        svm.fit(X_train, y_train)
        
        # Evaluate
        train_score = svm.score(X_train, y_train)
        test_score = svm.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        # Plot decision boundary
        plot_svm_decision_boundary(svm, X_scaled, y, 
                                 f"{name} Kernel SVM (C=1.0)")


if __name__ == "__main__":
    demo_svm()

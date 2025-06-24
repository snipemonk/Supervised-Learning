import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------
# 1. Load the Iris dataset
# -------------------------
iris = datasets.load_iris()
X = iris.data            # shape: (150, 4)
y = iris.target          # shape: (150,)

# -------------------------
# 2. Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 3. Custom binary SVM class
# -------------------------
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

# -------------------------
# 4. One-vs-Rest Multiclass SVM Wrapper
# -------------------------
class OneVsRestSVM:
    def __init__(self):
        self.classifiers = {}

    def fit(self, X, y):
        self.unique_classes = np.unique(y)
        for cls in self.unique_classes:
            # Label current class as 1, others as -1
            binary_y = np.where(y == cls, 1, -1)
            clf = SVM()
            clf.fit(X, binary_y)
            self.classifiers[cls] = clf

    def predict(self, X):
        # Evaluate decision function for each class and pick the highest
        scores = np.column_stack([clf.decision_function(X) for clf in self.classifiers.values()])
        return np.argmax(scores, axis=1)

# -------------------------
# 5. Train and Evaluate
# -------------------------
multi_svm = OneVsRestSVM()
multi_svm.fit(X_train, y_train)
predictions = multi_svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

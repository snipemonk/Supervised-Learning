# KNN Algorithm

import numpy as np
from collections import Counter

from pydeck.data_utils.viewport_helpers import k_nearest_neighbors


# define euclidean distance
def euclidean(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)

# define KNN class
class KNN:
    def __init__(self,k=3):
        self.k = k
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self,x):
        distances = [euclidean(x,x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def score(self,X_test,y_test):
        predictions = self.predict(X_test)
        return np.mean(predictions == y_test)

# Generate an Example Dataset for us to implement this code and view the Algorithm results

from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split

iris = load_iris()
X,y = iris.data,iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42)

# train and test the model

model = KNN(k=7)
model.fit(X_train,y_train)
accuracy = model.score(X_test,y_test)

print("Accuracy", accuracy)
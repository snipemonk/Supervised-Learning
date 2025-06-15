# Implementing KNN Python Algorithm on the cancer dataset

import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

 #define euclidean distance
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

data = load_breast_cancer()
X,y = data.data,data.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)


model = KNN(k=6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()

#import ace_tools as tools; tools.display_dataframe_to_user(name="KNN Classification Report", dataframe=class_report_df)

print("accuracy",accuracy)
print("confusion matrix",conf_matrix)
print("class report", class_report)
print(class_report_df)
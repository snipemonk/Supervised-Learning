from sklearn.tree import  DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

def train_decision_tree(X,y,test_size=0.2,max_depth=0.3,random_state=42):
    "split the data"
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)

    clf = DecisionTreeClassifier(max_depth=max_depth,random_state=random_state)

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    "Evaluate the data"
    acc = accuracy_score(y_test,y_pred)
    report =classification_report(y_test,y_pred)

    return clf,report,acc


# implementing on a sample dataset

from sklearn.datasets import load_iris

X,y = load_iris(return_X_y = True)

# Run the model
model,report,accuracy = train_decision_tree(X,y,test_size=0.5,max_depth=3)

print("Model Accuracy:", accuracy)
print("Classification Report:\n",report)





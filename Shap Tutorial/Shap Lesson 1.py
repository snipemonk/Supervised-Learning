import shap
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import  RandomForestClassifier
from sklearn.model_selection import train_test_split

# load iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data,columns=iris.feature_names)
y= iris.target

# train-test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Fit the model
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

# Explain with shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary plot
shap.summary_plot(shap_values[0],X_test,plot_type='bar')


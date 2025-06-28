import numpy as np
from collections import defaultdict

# Step 1 - split the training data by class
def separate_class(X,y):
    data = defaultdict(list)
    for features,label in zip(X,y):
        data[label].append(features)
    return {label: np.array(values) for label,values in data.items()}

# Step 2 - Calculate mean & variance for each class
def summarize(class_data):
    summaries ={}
    for label,rows in class_data.items():
        mean = np.mean(rows,axis=0)
        var = np.mean(rows,axis=0) + 1e-9
        summaries[label]=(mean,var)
    return summaries

# Step 3 - Calculate gaussian density function

def gaussian(x,mean,var):
    exponent = np.exp((-(x-mean)**2)/(2*var))
    return (1/np.sqrt(2*np.pi*var))*exponent

# Step 4 - Calculate the log probability for each class

def calculate_log_probability(summaries,x):
    log_probs = {}
    for label,(mean,var) in summaries.items():
        class_probs = gaussian(x,mean,var)
        log_probs[label]= np.sum(np.log(class_probs))
    return log_probs

# Step 5 - predict class with highest probability

def predict(summaries,x):
    predictions = []
    for sample in x:
        log_probs = calculate_log_probability(summaries,sample)
        predicted_class = max(log_probs,key=log_probs.get)
        predictions.append(predicted_class)
    return predictions

# Wrapper function to train and predict

def naive_bayes(X_train,y_train,X_test):
    class_data = separate_class(X_train,y_train)
    summaries = summarize(class_data)
    return predict(summaries,X_test)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X_train,X_test,y_train,y_test =(
    train_test_split(data.data,data.target,test_size=0.3,random_state=42 ))

y_pred = naive_bayes(X_train,y_train, X_test)

# Evaluate
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")


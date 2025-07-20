# This is an implementation of the SVM algorithm from scratch
import numpy as np


def train_svm(X,y,learning_rate=0.001,lambda_params=0.01,n_iters=1000):
    n_samples,n_features = X.shape
    y = np.where(y <= 0,-1,1)
    w = np.zeros(n_features)
    b = 0

    for _ in range(n_iters):
        for idx,x_i in enumerate(X):
            condition = y[idx]* (np.dot(x_i,w)+b) >= 1
        if condition:
            w -= learning_rate * (2*lambda_params*w)
        else:
            w -= learning_rate * (2*lambda_params*w - np.dot(x_i,y[idx]))
            b -= learning_rate * y[idx]

    return w,b

def predict_svm(X,w,b):
    return np.sign(np.dot(X,w)+b)

from sklearn.datasets import make_blobs

import matplotlib

#matplotlib.use('Agg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# Generate Synthetic data
X,y = make_blobs(n_samples=50,centers=2,random_state=42)
y = np.where(y==0,-1,1)

# Train
w,b = train_svm(X,y)

# predict
preds = predict_svm(X,w,b)

# Plot
import matplotlib.pyplot as plt
def plot_svm(X, y, w, b):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    ax = plt.gca()
    xlim = ax.get_xlim()
    x_vals = np.linspace(xlim[0], xlim[1], 30)
    y_vals = -(w[0]*x_vals + b)/w[1]
    plt.plot(x_vals, y_vals, 'k--')  # decision boundary
    plt.savefig("my_plot.png")

plot_svm(X,y,w,b)
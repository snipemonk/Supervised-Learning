import pandas as pd
import numpy as np
from sklearn.datasets import  load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    data = load_wine()
    X = pd.DataFrame(data.data,columns=data.feature_names)
    y = pd.Series(data.target,name='target')
    target_names = data.target_names
    return X,y, target_names

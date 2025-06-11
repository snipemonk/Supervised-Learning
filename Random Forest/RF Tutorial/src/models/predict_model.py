# This Code shows the implementation of the Random Forest Algorithm on the wine dataset

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

def train_random_forest(X_train,y_train,n_estimators=100,max_depth=None,criterion='gini'):
    model =RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,
                                  criterion=criterion,random_state=42)
    model.fit(X_train,y_train)
    return model

def evaluate_and_interpret(model,X_test,y_test,target_names):
    y_pred = model.predict(X_test)

    #Accuracy
    acc = accuracy_score(y_test,y_pred)
    print(f"\n Accuracy: {acc:.4f} — That’s {(acc * 100):.2f}% correct predictions.\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test,y_pred)
    df_cm = pd.DataFrame(cm,index=target_names,columns=target_names)
    print(" Confusion Matrix:")
    print(df_cm, "\n")

    # Classification Report
    report = classification_report(y_test,y_pred,target_names=target_names,output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("🧾 Classification Report:")
    print(report_df.round(2),"\n")

    # Plot confusion matrix
    ax = sns.heatmap(df_cm,annot=True,fmt='d',cmap='YlOrRd')
    ax.set(title='Confusion Matrix Heatmap',xlabel='Predicted Label',ylabel='True Label')

    # Interpretation
    print(" Interpretation of Classification Report:")
    for cls in target_names:
        precision = report[cls]['precision']
        recall = report[cls]['recall']
        f1 = report[cls]['f1-score']
        support = report[cls]['support']

        print(f"\nClass '{cls}':")
        print(f"  • Precision: {precision:.2f} — Out of predicted '{cls}', {(precision * 100):.1f}% were correct.")
        print(f"  • Recall: {recall:.2f} — Out of actual '{cls}', {(recall * 100):.1f}% were correctly identified.")
        print(f"  • F1 Score: {f1:.2f} — Balance between precision and recall.")
        print(f"  • Support: {support} — Number of actual instances in test data.")


def plot_feature_importance_seaborn(model, feature_names, top_n=10):
    # Get feature importances
    importances = model.feature_importances_

    # Get top N features
    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Create a DataFrame for plotting
    df_feat = pd.DataFrame({
        'Feature': top_features,
        'Importance': top_importances
    })

    # Sort for better visual layout
    df_feat = df_feat.sort_values(by='Importance', ascending=True)

    # Plot using seaborn only
    ax = sns.barplot(x='Importance', y='Feature', data=df_feat, palette='rocket')
    ax.set(title=' Top Feature Importances in Random Forest',
           xlabel='Importance',
           ylabel='Feature')

def run_pipeline(n_estimators=100, max_depth=None, criterion='gini', show_importance=True):
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    print(f" Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth}, criterion='{criterion}')...")
    model = train_random_forest(X_train, y_train, n_estimators=n_estimators,
                                max_depth=max_depth, criterion=criterion)

    evaluate_and_interpret(model, X_test, y_test, target_names)

    if show_importance:
        plot_feature_importance_seaborn(model, X.columns)

run_pipeline(n_estimators=150, max_depth=8, criterion='entropy', show_importance=True)

import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
plt.show()







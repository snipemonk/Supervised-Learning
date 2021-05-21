#Import Packages
import pandas as pd
import numpy as np
import sklearn
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import os
#Print the Directory of the Source File
print(os.listdir('C:/Users/shrirajmisra/OneDrive/Desktop/Work/ML/ML_Datasets/Adult'))

#Check the File Size
print('#File Sizes')
for f in os.listdir('C:/Users/shrirajmisra/OneDrive/Desktop/Work/ML/ML_Datasets/Adult/'):
    print(f.ljust(30)+str(round(os.path.getsize('C:/Users/shrirajmisra/OneDrive/Desktop/Work/ML/ML_Datasets/Adult/' + f)/1000000,2))+'MB')


def main():
    file=('C:/Users/shrirajmisra/OneDrive/Desktop/Work/ML/ML_Datasets/Adult/adult.csv')
    df = pd.read_csv(file,encoding='latin-1')
    print(df.head())
    print(df.info())

    #Encode  ? as Nan
    df[df=='?']=np.nan
    print(df.info())

    #Find Sum of  Missing Values
    df_missing_values = df.isnull().sum()

    #Impute Missing Values with the most frequent value
    for col in ['workclass','occupation','native.country']:
        df[col].fillna(df[col].mode()[0],inplace=True)

    # Setting Feature And Target Variable
    X= df.drop(['income'],axis=1)
    y= df['income']

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=0)

    from sklearn import preprocessing

    categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

    # Perform Feature Scaling
    from sklearn.preprocessing import StandardScaler
    scaler =StandardScaler()
    X_train=pd.DataFrame(scaler.fit_transform(X_train),columns=X.columns)
    X_test=pd.DataFrame(scaler.transform(X_test),columns=X.columns)

    # Logistic Regression Model with All Features
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    logreg = LogisticRegression()
    logreg.fit(X_train,y_train)
    y_pred =logreg.predict(X_test)

    print('Logistic Regression accuracy score with all features:{0:0.4f}'.format(accuracy_score(y_test,y_pred)))


    # Logistic Regression with PCA
    from sklearn.decomposition import PCA
    pca =PCA()
    X_train = pca.fit_transform(X_train)
    pca.explained_variance_ratio_

    # We can see the last variable only contributes 2.74 % of info, so this can be dropped
    # Now running Logistic Regression with 13 variables
    X = df.drop(['income', 'native.country'], axis=1)
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
    for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)

    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    print('Logistic Regression accuracy score with the first 13 features: {0:0.4f}'.format(
        accuracy_score(y_test, y_pred)))

    #Now We see that the accuracy has dropped,
    # However we now implement a function which will show no of variables needed to preserve 90% data
    X = df.drop(['income'], axis=1)
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex',
                   'native.country']
    for feature in categorical:
        le = preprocessing.LabelEncoder()
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)

    pca = PCA()
    pca.fit(X_train)
    cumsum =np.cumsum(pca.explained_variance_ratio_)
    dim = np.argmax(cumsum>=0.90)+1
    print('The number of Dimensions Required is',dim)



    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlim(0, 14, 1)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

if __name__=='__main__':
    main()

#third commit
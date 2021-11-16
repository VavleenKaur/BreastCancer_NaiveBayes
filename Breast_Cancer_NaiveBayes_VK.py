#  NAIVE BAYES APPROACH
# VAVLEEN KAUR

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, r2_score

import missingno as msn
import seaborn as sns


def missing_value_handle(df):
    for column in df:
        df[column] = df[column].replace(to_replace="?", value=np.nan)
        df[column] = df[column].fillna(int(df[column].mode()[0]))
        df[column] = df[column].astype(int)
    df.astype('int32').dtypes
    print("DATAFRAME AFTER HANDLING MISSING VALUES WITH MODE VALUES")
    print(df)
    # no missing values now confirmed by the given bar plot
    msn.bar(df)
    return df


def split_data(df):
    X = df.iloc[:, 0:10]
    # print(X)
    Y = (df.iloc[:, 10]).values
    # print(Y)
    return X, Y


def heat_map(X):
    # to know correlation among features
    sns.heatmap(X.corr(), annot=True)


def countplot(Y, df):
    # count of each class
    sns.countplot(y=Y, data=df)


def PCA(X_scaled, Y):
    features = X_scaled.T
    # features
    covmat = np.cov(features)
    # covmat
    values, vectors = np.linalg.eig(covmat)
    pt_var = []
    for i in range(len(values)):
        pt_var.append(values[i]/np.sum(values))
    # pt_var
    projected_1 = X_scaled.dot(vectors.T[0])
    projected_2 = X_scaled.dot(vectors.T[1])
    projected_3 = X_scaled.dot(vectors.T[3])
    projected_4 = X_scaled.dot(vectors.T[5])
    res = pd.DataFrame(data=projected_1, columns=["PC1"])
    res["PC2"] = projected_2
    res["PC3"] = projected_3
    res["PC4"] = projected_4
    res["Class"] = Y
    check = res[["PC1", "PC2", "PC3", "PC4"]]
    print("With the help of PCA, now there is very less correlation among features and dimensionality has been reduced")
    print(res)
    sns.heatmap(check.corr(), annot=True)
    # check
    return res, check


def train_predict_naivebayes(check, res):
    X_f = check
    Y_f = res["Class"]
    # print(X_f)
    # print(Y_f)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_f, Y_f, test_size=0.3, random_state=42)
    # print(X_train)
    # print(Y_train)
    model = GaussianNB()
    model = model.fit(X_train, Y_train)

    # predicting test and training data
    Y_pred = model.predict(X_test)
    return Y_pred, Y_test


def EvaluationMetrics(Y_test, Y_pred):
    cnf_matrix = confusion_matrix(Y_test, Y_pred)
    print(cnf_matrix)


def main():
    df = pd.read_csv('breast-cancer-wisconsin.data', header=None)
    df = df.rename(columns={0: "Sample code number", 1: "Clump Thickness", 2: "Uniformity of Cell Size", 3: "Uniformity of Cell Shape",
                   4: "Marginal Adhesion", 5: "Single Epithelial Cell Size", 6: "Bare Nuclei", 7: "Bland Chromatin", 8: "Normal Nucleoli", 9: "Mitoses", 10: "Class"})
    print(df)
    df = missing_value_handle(df)
    X, Y = split_data(df)
    heat_map(X)
    countplot(Y, df)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    res, check = PCA(X_scaled, Y)
    Y_pred, Y_test = train_predict_naivebayes(check, res)
    EvaluationMetrics(Y_test, Y_pred)
    acc = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred, average="macro")
    recall = recall_score(Y_test, Y_pred, average="macro")
    prec = precision_score(Y_test, Y_pred, average="macro")
    r2 = r2_score(Y_test, Y_pred)
    print("Accuracy: ", acc)
    print("F1 Score: ", f1)
    print("Recall: ", recall)
    print("Precision: ", prec)
    print("R2_score: ", r2)


if __name__ == "__main__":
    main()

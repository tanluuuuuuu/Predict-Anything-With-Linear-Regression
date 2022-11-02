import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def transform_column_to_onehot(df, selected_columns, dict_encoder ):
    X = np.array([])
    
    for column_label in selected_columns:
        col = df[column_label].to_numpy().reshape(-1, 1)
        if (df.dtypes[column_label] == 'object'):
            enc = OneHotEncoder(handle_unknown='ignore')
            enc = enc.fit(col)
            dict_encoder[column_label] = enc
            col = enc.transform(col).toarray()
        if len(X) == 0:
            X = col
        else:
            X = np.concatenate((X, col), axis=1)
    return X, dict_encoder

def split_data_and_train_model(X, y, train_split_ratio):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split_ratio / 100.0, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def transform_data_to_predict(selected_columns, dict_data_to_prediction):
    data_to_prediction = np.array([])
    for data_column in selected_columns:
        data_to_prediction = np.append(data_to_prediction, dict_data_to_prediction[data_column])
    data = np.array(data_to_prediction).reshape(1, -1)
    return data

def predict_and_evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    return mean_squared_error(y_test, prediction)

def train_with_kFold(X, y, n_splits):
    model = LinearRegression()
    kf = KFold(n_splits=n_splits)
    train_loss = []
    test_loss = []
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        loss_test = predict_and_evaluate_model(model, X_test, y_test)
        loss_train = predict_and_evaluate_model(model, X_train, y_train)
        train_loss.append(loss_train)
        test_loss.append(loss_test)
    return model, train_loss, test_loss
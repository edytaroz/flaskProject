import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf


def preprocess():
    data = pd.read_csv("C:\\Users\\Dell\\Downloads\\covtype.data",header=None)
    y = data[54]
    X = data.drop(54,axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=43)
    return X_train, X_test, y_train, y_test


def heuristic(y_test):
    res = np.array([random.randrange(1,7,1) for _ in range(len(y_test))])
    return accuracy_score(y_test,res)


def knnclassifier(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(algorithm='kd_tree')
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test,y_pred)


def dtclassifier(X_train, X_test, y_train, y_test):
    dc = DecisionTreeClassifier()
    dc.fit(X_train,y_train)
    dcpred = dc.predict(X_test)
    return accuracy_score(y_test,dcpred)


def nnclassifier(X_train,X_test,y_train,y_test):
    inputs = tf.keras.Input(shape=(54,), name="digits")
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    x_val = X_train[-10000:]
    y_val = y_train[-10000:]
    X_train = X_train[:-10000]
    y_train = y_train[:-10000]

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=15,
        validation_data=(x_val, y_val),
    )

    results = model.evaluate(X_test, y_test, batch_size=128)
    # print("test loss, test acc:", results)
    return results[1]

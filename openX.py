import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    return res


def knnclassifier(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(algorithm='kd_tree')
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    return y_pred


def dtclassifier(X_train, X_test, y_train, y_test):
    dc = DecisionTreeClassifier()
    dc.fit(X_train,y_train)
    dcpred = dc.predict(X_test)
    return dcpred


def nnclassifier(X_train,X_test,y_train,y_test):
    inputs = tf.keras.Input(shape=(54,), name="digits")
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model1 = tf.keras.Model(inputs=inputs, outputs=outputs)
    model2 = tf.keras.Model(inputs=inputs, outputs=outputs)

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
    model1.compile(
        optimizer='adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model2.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
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
    history1 = model1.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=15,
        validation_data=(x_val, y_val),
    )
    history2 = model2.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=15,
        validation_data=(x_val, y_val),
    )

    results = model.evaluate(X_test, y_test, batch_size=128)
    results1 = model1.evaluate(X_test, y_test, batch_size=128)
    results2 = model2.evaluate(X_test, y_test, batch_size=128)
    if results2[1] >= results1[1] and results2[1] >= results[1]:
        loss = history2.history2['loss']
        val_loss = history2.history2['val_loss']
        plt.plot(loss)
        plt.plot(val_loss)
        plt.show()
        return results2[1]
    elif results1[1] >= results2[1] and results1[1] >= results[1]:
        loss = history1.history1['loss']
        val_loss = history1.history1['val_loss']
        plt.plot(loss)
        plt.plot(val_loss)
        plt.show()
        return results1[1]
    else:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        plt.plot(loss)
        plt.plot(val_loss)
        plt.show()
        return results[1]
    # print("test loss, test acc:", results)

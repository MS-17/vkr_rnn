import os, json, shutil
import sys

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers as rg

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
# from sklearn.model_selection import train_test_split
# import scipy.stats as st
# import scipy

# from statsmodels.stats.outliers_influence import variance_inflation_factor

import plotly.graph_objs as go

# import warnings
# warnings.filterwarnings("ignore")

assert tf.__version__ == '1.14.0'
assert sys.version_info[0] == 3 and sys.version_info[1] == 7
#assert google.protobuf.__version__ == '3.20.0'
#assert h5py.__version__ == '2.10.0'

data_path = "data/"
train_data_path = data_path + "train_ds.json"
test_data_path = data_path + "test_ds.json"

with open(train_data_path) as f:
  train_data = json.load(f)
print("Train data keys:", train_data.keys())

X_train_l = train_data["x_train"]
y_train_l = train_data["y_train"]

with open(test_data_path) as f:
  test_data = json.load(f)
print("Test data keys:", test_data.keys())

X_test_l = test_data["x_test"]
y_test_l = test_data["y_test"]

X_train = np.array(X_train_l)
y_train = np.array(y_train_l)
X_test = np.array(X_test_l)
y_test = np.array(y_test_l)
print(f"Train data shape x: {X_train.shape}, y: {y_train.shape}")
print(f"Test data shape x: {X_test.shape}, y: {y_test.shape}")

input_shape = X_train.shape[1:]

model = Sequential()

model.add(tf.keras.layers.LSTM(32, input_shape=input_shape, return_sequences=True, bias_regularizer=rg.l1_l2(l1=0, l2=0.01)))
model.add(tf.keras.layers.LSTM(32, return_sequences=True, bias_regularizer=rg.l1_l2(l1=0, l2=0.01)))
model.add(tf.keras.layers.LSTM(32, return_sequences=True, bias_regularizer=rg.l1_l2(l1=0, l2=0.01)))

model.add(tf.keras.layers.Flatten())

model.add(Dense(8, activation="relu", bias_regularizer=rg.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))

model.add(Dense(8, activation="relu", bias_regularizer=rg.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))

model.add(Dense(1, activation="sigmoid"))
model.compile(
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy",] #, tf.compat.v1.metrics.auc()] #, "precision", "recall"])
)
model.summary()
callback_ = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min") #, start_from_epoch=3)
model.fit(X_train, y_train, batch_size=32, validation_split=0.2, epochs=100, callbacks=[callback_])

print("\nModel evaluate")
print(model.evaluate(X_test, y_test))

# save the model
print("Saving the model")
model_save_path = "saved_model/"
model_name = "model.h5"
os.makedirs(model_save_path, exist_ok=True)
model.save(model_save_path + model_name)
print("Model saved at", model_save_path + model_name)

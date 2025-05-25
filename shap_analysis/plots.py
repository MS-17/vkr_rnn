import os, json, shutil
import sys

import tensorflow as tf
assert tf.__version__ == '1.14.0'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import plotly.graph_objs as go

# import warnings
# warnings.filterwarnings("ignore")


data_path = "data/"
test_data_path = data_path + "test_ds.json"

with open(test_data_path) as f:
  test_data = json.load(f)
print("Test data keys:", test_data.keys())

X_test_l = test_data["x_test"]
y_test_l = test_data["y_test"]

X_test = np.array(X_test_l)
y_test = np.array(y_test_l)
print(f"Test data shape x: {X_test.shape}, y: {y_test.shape}")

# load the tf model
model_path = "saved_model/model.h5"
model = tf.keras.models.load_model(model_path)

# create the plots folder 
plots_path = "plots/train/"
print("Creating the folder for plots", plots_path)
os.makedirs(plots_path, exist_ok=True)


# roc
print("Creating the roc plot")
y_pred = model.predict(X_test)
print("AUC score:", roc_auc_score(y_test, y_pred))
fpr, tpr, thresh = roc_curve(y_test, y_pred)
plt.plot([0, 1], [0, 1])
plt.plot(fpr, tpr)
# plt.show()
print("Saving the roc plot at", plots_path + "roc_curve.png")
plt.savefig(plots_path + "roc_curve.png", dpi=150)
plt.close()


# roc plotly
print("Creating the plotly roc plot")
trace = go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='darkorange', width=2), text=thresh)
reference_line = go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Reference Line',
                            line=dict(color='navy', width=2, dash='dash'))
fig = go.Figure(data=[trace, reference_line])
fig.update_layout(title='Interactive ROC Curve',
                  xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate')
# fig.show()
print("Saving the plotly roc plot at", plots_path + "roc_plotly.html")
fig.write_html(plots_path + "roc_plotly.html")


# classification report and confusion matrix
thresh_ = 0.5
y_pred_class = (y_pred > thresh_).astype(int)
print("y_pred shape:", y_pred_class.shape)
print("Predicted classes:", np.unique(y_pred_class))

print("Classification report")
print(classification_report(y_true=y_test, y_pred=y_pred_class))

print("Creating the confusion matrix")
cm = confusion_matrix(y_true=y_test, y_pred=y_pred_class)
sns.heatmap(cm, annot=True, fmt="g")
# plt.show()
print("Saving the confusion matrix at", plots_path + "confusion_matrix.png")
plt.savefig(plots_path + "confusion_matrix.png", dpi=150)
plt.close()

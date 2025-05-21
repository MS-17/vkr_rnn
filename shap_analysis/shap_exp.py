import os, json, sys
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import shap

# python3.7
#!pip install h5py==2.10
#!pip install protobuf==3.20
#!pip install tensorflow==1.14

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
# y_train_l = train_data["y_train"]

with open(test_data_path) as f:
  test_data = json.load(f)
print("Test data keys:", test_data.keys())

X_test_l = test_data["x_test"]
y_test_l = test_data["y_test"]

X_train = np.array(X_train_l)
# y_train = np.array(y_train_l)
X_test = np.array(X_test_l)
y_test = np.array(y_test_l)
print(f"Test data shape x: {X_test.shape}, y: {y_test.shape}")
print(f"Train data shape x: {X_train.shape}") #, y: {y_train.shape}")


cols = test_data["ordered_test_columns"]
print("Dataset columns:", cols)
# cols = [
#     "apcp_mean", "rh_mean", "t_mean", "wind_speed_mean", "wind_dir_std", "aspect",
#     "elevation", "slope", "locality_dist", "river_dist", "road_dist",
#     "soilw10_mean", "vegetation_type",
# ]

model_path = "saved_model/model.h5"
model = tf.keras.models.load_model(model_path)
y_pred = model.predict(X_test)
print("AUC score:", roc_auc_score(y_test, y_pred))
fpr, tpr, thresh = roc_curve(y_test, y_pred)
plt.plot([0, 1], [0, 1])
plt.plot(fpr, tpr)
plt.show()
plt.close()


print("Starting shap")
shap.initjs()
# shap_samples = 10
shap_samples = X_test.shape[0]
print("Shap samples: ", shap_samples)
exp = shap.DeepExplainer(model, X_train[1000:2000])
print("Calculating shap values")
shap_val = exp.shap_values(X_test[:shap_samples])
print("End of shap values calculation")


# create the plots folder
shap_plots_path = "plots/shap/"
print("Creating the folder for plots", shap_plots_path)
os.makedirs(shap_plots_path, exist_ok=True)


# shap bar plot
shap_ds = pd.DataFrame(np.abs(shap_val[0]).mean(0), columns=cols)
print(shap_ds)
shap_col = shap_ds.values.mean(0)
df = pd.DataFrame(list(zip(cols, shap_col)), columns=["name", "importance"])
df.sort_values(by=["importance"], inplace=True)
print("Feature importance dataset\n", df)
df.plot.barh(x="name", y="importance", stacked=False, title="Feature importance")
print("Saving the shap bar plot at", shap_plots_path + "feature_importance_plot.png")
plt.savefig(shap_plots_path + "feature_importance_plot.png", bbox_inches="tight", dpi=150)
plt.close()


# shap summary
print("Creating the shap summary plot")
row_ = 5
shap.summary_plot(shap_val[0][:, row_, :], X_test[:shap_samples, row_, :], 
    feature_names=cols, max_display=20,
    # plot_size=(8, 6),
    show=False,
)
ax = plt.gca()
scale_ = 8
ax.set_xlim(-scale_ * 1e-5, scale_ * 1e-5)
print("Saving the shap summary plot at", shap_plots_path + "summary_plot.png")
ax.figure.savefig(shap_plots_path + "summary_plot.png", dpi=150)
plt.close()

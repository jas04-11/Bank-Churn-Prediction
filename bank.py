from google.colab import files

uploaded = files.upload()

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("Bank Customer Churn Prediction.csv")


df.head()

print(df.shape)

print(df.columns)

print(df.isnull().sum())

df = df.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1,
    errors="ignore"
)

X = df.drop("churn", axis=1)
y = df["churn"]

print("Features:")
print(X.head())
print("\nTarget:")
print(y.head())

X = pd.get_dummies(X, columns=["country", "gender"], drop_first=True)

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=67,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)

print("Logistic Regression Performance")
print("--------------------------------")
print("Accuracy :", lr_accuracy)
print("Precision:", lr_precision)
print("Recall   :", lr_recall)
print("F1 Score :", lr_f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)

print("Confusion Matrix:")
print(cm_lr)
plt.imshow(cm_lr)
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn)
knn_recall = recall_score(y_test, y_pred_knn)
knn_f1 = f1_score(y_test, y_pred_knn)

print("k-NN Performance")
print("-----------------")
print("Accuracy :", knn_accuracy)
print("Precision:", knn_precision)
print("Recall   :", knn_recall)
print("F1 Score :", knn_f1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn)

print("Confusion Matrix:")
print(cm_knn)
plt.imshow(cm_knn)
plt.title("k-NN - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Logistic Regression": [
        lr_accuracy,
        lr_precision,
        lr_recall,
        lr_f1
    ],
    "k-NN": [
        knn_accuracy,
        knn_precision,
        knn_recall,
        knn_f1
    ]
})

print(comparison)

comparison.set_index("Metric").plot(kind="bar", figsize=(8, 5))

plt.title("Logistic Regression vs k-NN")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

dt = DecisionTreeClassifier(random_state=67)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt)
dt_recall = recall_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt)

print("Decision Tree Performance")
print("-------------------------")
print("Accuracy :", dt_accuracy)
print("Precision:", dt_precision)
print("Recall   :", dt_recall)
print("F1 Score :", dt_f1)

cm_dt = confusion_matrix(y_test, y_pred_dt)

print("Confusion Matrix:")
print(cm_dt)

plt.imshow(cm_dt)
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_precision = precision_score(y_test, y_pred_dt)
dt_recall = recall_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt)
print("Decision Tree Performance")
print("-------------------------")
print("Accuracy :", dt_accuracy)
print("Precision:", dt_precision)
print("Recall :", dt_recall)
print("F1 Score :", dt_f1)

import seaborn as sns
cm_dt = confusion_matrix(y_test, y_pred_dt)
TN, FP, FN, TP = cm_dt.ravel()
labels = [[f"TN\n{TN}", f"FP\n{FP}"],[f"FN\n{FN}", f"TP\n{TP}"]]
plt.figure(figsize=(7, 5))
sns.heatmap(cm_dt,annot=labels,fmt="",cmap="Blues",cbar=True,linewidths=2,linecolor="white",xticklabels=
["Negative", "Positive"], yticklabels=["Negative", "Positive"],annot_kws={"size": 13, "weight": "bold"})
plt.title("Decision Tree - Confusion Matrix", fontsize=15, fontweight="bold")
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.show()

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Logistic Regression": [
        lr_accuracy, lr_precision, lr_recall, lr_f1
    ],
    "k-NN": [
        knn_accuracy, knn_precision, knn_recall, knn_f1
    ],
    "Decision Tree": [
        dt_accuracy, dt_precision, dt_recall, dt_f1
    ]
})

print(comparison)

comparison.set_index("Metric").plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Logistic Regression vs k-NN vs Decision Tree")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# working on random forest

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import sklearn
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib


# Task 1: Fetch the Data
# I chose city fairfax in Virginia
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 38.83,
    "longitude": -77.28,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)
print(df.shape)
print(df.head())

# Task 2: Engineer the Label


df['good_for_running'] = ((df['temperature_2m_max'] > 7) & (df['temperature_2m_max'] < 26) & (df['temperature_2m_min'] >= 0) & (df['precipitation_sum'] < 3) & (df['wind_speed_10m_max'] < 30)).astype(int)
print(df.head())
print(df['good_for_running'].value_counts())
# 41 percent of the days in a year are good. I think it's resonable for my climate. 


# Task 3: Build and Evaluate a KNN Classifier
# Split the data into training and test sets (80/20, random_state=42, stratify on the label). 
# Build a Pipeline of StandardScaler and KNeighborsClassifier. 
# Use 5-fold cross-validation on the training data to choose k from at least five values, 
# then fit your chosen model and print its test accuracy and full classification report.
features = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]
X = df[features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )


for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}")

best_k = 1
knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
])

knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Test accuracy: {acc:.3f}   F1: {f1:.3f}")

print(classification_report(y_test, y_pred, target_names=["skip", "good"]))
#------

# Task 4: Build and Evaluate a Logistic Regression Classifier
# Build a Pipeline of StandardScaler and LogisticRegression(max_iter=1000). 
# Fit it and print its test accuracy, precision, recall, F1, and classification report. 
# Then print each feature's coefficient and add a comment interpreting them: which conditions most strongly push a day toward "skip"?
logreg_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
logreg_model.fit(X_train, y_train)
y_pred = logreg_model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["skip", "good"]))

logreg = logreg_model.named_steps["logreg"]
for name, coef in zip(features, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")
# The precipitation_sum has the strongest negative coefficient, which means more rain would strongly push a day toward "skip".


# Task 5: Compare and Choose
# The KNN model has higher accuracy on the data.

# For a running app, I would weigh precision or recall more heavily. beause for users, it's better to miss a good day than being sent out on a bad day.

# I will deploy knn_model rather than logreg_model. because it has a much higher accuracy score that logreg_model(95% vs 74%).


#Task 6: Save the Model
# Save a metadata file to models/weather_classifier_metadata.json containing at least: the Python and scikit-learn versions, 
# the feature names in order, the label rule you used, your city, and the test accuracy and F1 of the saved model.
#  Print a confirmation message when both files are written.
joblib.dump(knn_model, "./assignments_03/models/weather_classifier.pkl")
print("Model saved.")

metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "city": "Virginia",
    "features": features,
    "label": "good_for_running",
    "test_accuracy": round(acc, 3),
    "test_f1": round(f1, 3),
    "label_rule": {
        "temperature_2m_max": "7 to 26 C",
        "temperature_2m_min": ">= 0 C",
        "precipitation_sum": "< 3 mm",
        "wind_speed_10m_max": "< 30 km/h",
    },
}

with open("./assignments_03/models/weather_classifier_metadata.json", "w") as f: #models/weather_classifier_metadata.json
    json.dump(metadata, f, indent=2)
print("Metadata saved.")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

script_dir = Path(__file__).parent
file_path = script_dir / "weather_classification.csv"
df = pd.read_csv(file_path)
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- preprocessing --- 

# Q1
#Fit a StandardScaler on X_train only, then transform both X_train and X_test. 
# Print the mean of each column of the scaled training data (they should all be very close to 0). 
# Add a comment explaining,in one sentence, why you fit the scaler on X_train only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  #returns a numpy array
X_test_scaled = scaler.transform(X_test)

scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
print(scaled_df.mean())
# Fitting on all the data lets information from the test set influence the scaling,
#  which is data leakage and makes evaluation look better than it really is.

# Q2

encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(df[['season']])
print(encoded_array.shape)
category_name = encoder.get_feature_names_out(['season'])
print(category_name)
#encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['season']))
#the single season column became a four columns array because OneHotEncoder 
# creates a separate binary column for each unique category present in the feature.

# Q3
# Build a ColumnTransformer that applies a StandardScaler to the four numeric columns and a OneHotEncoder to season.
# Wrap it with a LogisticRegression in a Pipeline. 
# Fit the pipeline on the training data (include season in your feature columns for this question) and print its test accuracy. 
# Add a comment: what does the pipeline do automatically that you would otherwise have to do by hand?
#numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
categorical_features = ["season"]

features = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "season",
]
X = df[features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(X_train, y_train)      # fits the scaler, the encoder, AND the classifier
predictions = model.predict(X_test)

test_accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
# the pipeline automatically scale the numeric features, one-hot encode the categorical feature, and then fit a model.


# --- KNN--- 
# Q1 
# Build a Pipeline of StandardScaler and KNeighborsClassifier(n_neighbors=5). 
# Fit it on the training data and print the test accuracy and the full classification report.
features = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
knn_scaled.fit(X_train, y_train)
print("Scaled accuracy:  ", accuracy_score(y_test, knn_scaled.predict(X_test)))

y_pred = knn_scaled.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["skip", "good"]))

#Q2
#Loop over k values [1, 3, 5, 7, 9, 11, 15, 21]. For each, build the scaled pipeline 
# and compute the mean 5-fold cross-validation accuracy on the training data. 
# Print each k with its mean CV score. Add a comment naming the k you would choose and why.
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"k={k:2d}:  mean={scores.mean():.3f}")

# I would choose k = 11. Because it has the highest average accuracy score.

# Q3
#  Fit KNN once without scaling (on raw X_train) and once with scaling, both using n_neighbors=5. Print both test accuracies.
#  Add a comment: did scaling help, and why would scaling matter for a distance-based model?
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
print("Unscaled accuracy:", accuracy_score(y_test, knn_unscaled.predict(X_test)))

knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
knn_scaled.fit(X_train, y_train)
print("Scaled accuracy:  ", accuracy_score(y_test, knn_scaled.predict(X_test)))
# It did help. distance is dominated by large-range features unless the features are standardized. 
# Scaling let precipitation and the other features contribute fairly, which changed the predictions.

#---  Logistic Regression --
print("Logistic Regression questions")
# Q1
# Build a Pipeline of StandardScaler and LogisticRegression(max_iter=1000). 
# Fit it and print the test accuracy, precision, recall, and F1 (for the "good" class).
X = df[features]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["skip", "good"]))

# Q2
# Print each feature name alongside its coefficient from the fitted logistic regression (use pipeline.named_steps["..."].coef_[0]). 
# Add a comment: which features push most strongly toward "skip," and does that match your intuition about running weather?
logreg = model.named_steps["logreg"]
for name, coef in zip(features, logreg.coef_[0]):
    print(f"{name:20s}: {coef:+.2f}")
# The wind_speed_10m_max and precipitation_sum both have strong negative coefficients, which means more rain and more wind both push a day strongly toward "skip." 
# It does mathch my intuition about running weather.

# Q3
#Use predict_proba to print the probability of "good" for the first five test days, rounded to two decimals. 
#Add a comment: what does a probability near 0.5 tell you about the model's confidence for that day?
probabilities = model.predict_proba(X_test)[:, 1]   # probability of "good"
print(probabilities[:5].round(2))
#  a probability near 0.5 tells that the model is not confident on "good for a run" for that day.


#---  Evaluation  --
print("Evaluation Question")

# Q1
# For your logistic regression pipeline, build a confusion matrix on the test set and display it with ConfusionMatrixDisplay, 
# using display_labels=["skip", "good"]. 
# Save the figure to outputs/logreg_confusion_matrix.png. 
# Add a comment: how many false positives and how many false negatives did the model make, 
# and which kind of error would matter more for a running app?

labels = ["skip", "good"]
cm = confusion_matrix(y_test, y_pred, labels=[0,1])

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(colorbar=False)
plt.title("Logistic Regression Confusion Matrix")
plt.show()
plt.savefig("./assignments_03/outputs/logreg_confusion_matrix.png")
# the model made 9 false positives and 7 false negatives. The false positives matter more for a running app.


# ---- Joblib -----
# Q1
#Take your fitted logistic regression pipeline and save it to models/warmup_model.pkl with joblib.dump. 
# Load it back with joblib.load and confirm it produces identical predictions:
joblib.dump(model, "./assignments_03/models/warmup_model.pkl")
loaded = joblib.load("./assignments_03/models/warmup_model.pkl")
assert (loaded.predict(X_test) == model.predict(X_test)).all()
print("Predictions match.")
# Add a comment: what would go wrong if you had saved only the LogisticRegression step without the scaler, 
# then called .predict() on raw data?
# The classifier was trained on scaled data, so the predicted results would be wrong if you passed the raw data directly to the classifier.
import joblib
import json
import pandas as pd

# Task 7: Load and Predict
# Load the pipeline from models/weather_classifier.pkl and the metadata from the JSON file.
#  Print the model's key metadata (city, features, test accuracy). 
# Then build a small DataFrame of at least four hypothetical days covering a clearly good day, 
# a clearly bad day, and at least one borderline day, using the feature names from the metadata 
# so the columns match. For each day, print the four inputs, 
# the predicted label (good or skip), and the model's confidence (the probability of "good").



clf = joblib.load("./assignments_03/models/weather_classifier.pkl")
with open("./assignments_03/models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)

print(metadata['city'])
print(metadata['features'])
print(metadata['test_accuracy'])

new_days = pd.DataFrame(
    [
        {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 12},
        {"temperature_2m_max": 25, "temperature_2m_min": 1, "precipitation_sum": 2, "wind_speed_10m_max": 27},
        {"temperature_2m_max": 28, "temperature_2m_min": -2, "precipitation_sum": 5, "wind_speed_10m_max": 45},
        {"temperature_2m_max": 22, "temperature_2m_min": 8, "precipitation_sum": 0.0, "wind_speed_10m_max": 38},
    ]
)[metadata["features"]]
preds = clf.predict(new_days)
probs = clf.predict_proba(new_days)[:, 1]

for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = "good for running" if pred == 1 else "skip"
    print(f"Day {i + 1}: {label}  (confidence: {prob:.2f})")


# Task 8: Reflect
# 1. Pick your borderline day. What probability did the model give it?
#  Would you call the model confident or uncertain, and how would you handle a day where it predicts 0.52?
# A: this is my boardline day: {"temperature_2m_max": 25, "temperature_2m_min": 1, "precipitation_sum": 2, "wind_speed_10m_max": 27}.
#.   the model gave a confidence: 1.00. I think the model is pretty confident. When it predicts 0.52, I would do "skip".


# 2. predict.py has no training code. What would break if someone ran predict.py before train_classifier.py had ever been run?
# How could you make the error message more helpful?
# A: The model would not be found to predict the data. I would deliver message: "couldn't find the training model."


# 3. In Week 10 this model will run inside a cloud pipeline that classifies each day's weather automatically.
#  Name one thing about your current predict.py that would need to change to support that.
# A: I don't know.
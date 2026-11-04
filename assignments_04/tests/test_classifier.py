import pytest
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from weather_model import WeatherClassifier, Prediction, FEATURES

# Task 4: The Test Suite
# So the tests do not depend on any particular saved file, build a tiny model inside a fixture and save it to pytest's tmp_path, then load a WeatherClassifier from it.
@pytest.fixture
def model_path(tmp_path):
    """Train a tiny model on obvious days and save it to a temp file."""
    good = [{"temperature_2m_max": 18, "temperature_2m_min": 9, "precipitation_sum": 0.0, "wind_speed_10m_max": 10}] * 15
    bad = [{"temperature_2m_max": 2, "temperature_2m_min": -6, "precipitation_sum": 12.0, "wind_speed_10m_max": 40}] * 15
    X = pd.DataFrame(good + bad)[FEATURES]
    y = [1] * 15 + [0] * 15

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)

    path = tmp_path / "weather_classifier.pkl"
    joblib.dump(pipe, path)
    return path


@pytest.fixture
def clf(model_path):
    """A WeatherClassifier loaded from the saved tiny model."""
    return WeatherClassifier(model_path)



@pytest.mark.parametrize("days", [[{"temperature_2m_max": 18, "temperature_2m_min": 9, "precipitation_sum": 0.0, "wind_speed_10m_max": 10}, {"temperature_2m_max": 3, "temperature_2m_min": -5, "precipitation_sum": 10.0, "wind_speed_10m_max": 38}], [{"temperature_2m_max": 25, "temperature_2m_min": 12, "precipitation_sum": 0.0, "wind_speed_10m_max": 8}, {"temperature_2m_max": 10, "temperature_2m_min": -3, "precipitation_sum": 8.0, "wind_speed_10m_max": 42}]])
def test_days_predicted_with_right_label(clf, days):
    results = clf.predict(days)
    assert results[0].label == "good"
    assert results[1].label == "skip"


def test_probability_is_between_zero_and_one(clf):
    result = clf.predict([
        {"temperature_2m_max": 15, "temperature_2m_min": 8, "precipitation_sum": 1.0, "wind_speed_10m_max": 12}
    ])
    assert 0.0 <= result[0].probability <= 1.0


def test_one_prediction_per_day_in_order(clf):
    days = [
        {"temperature_2m_max": 18, "temperature_2m_min": 9, "precipitation_sum": 0.0, "wind_speed_10m_max": 10},
        {"temperature_2m_max": 2, "temperature_2m_min": -6, "precipitation_sum": 12.0, "wind_speed_10m_max": 40},
        {"temperature_2m_max": 15, "temperature_2m_min": 8, "precipitation_sum": 1.0, "wind_speed_10m_max": 16},
    ]
    results = clf.predict(days)
    assert len(results) == 3
    assert all(isinstance(r, Prediction) for r in results)


def test_empty_list_returns_empty(clf):
    assert clf.predict([]) == []


def test_missing_reature_raise(clf):
    with pytest.raises(ValueError, match='features'):
        clf.predict([
        {"temperature_2m_max": 15, "temperature_2m_min": 8, "precipitation_sum": 1.0},
        ])

def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model file"):
        WeatherClassifier(tmp_path / "does_not_exist.pkl")

# when predict method only return "skip", the def test_days_predicted_with_right_label(days) catch the error.

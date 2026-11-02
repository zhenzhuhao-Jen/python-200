from weather_model import WeatherClassifier
from pathlib import Path


# Task 5: The Script

SAMPLE_DAYS = [
    {"temperature_2m_max": 18, "temperature_2m_min": 10, "precipitation_sum": 0.0, "wind_speed_10m_max": 12},
    {"temperature_2m_max": 22, "temperature_2m_min": 13, "precipitation_sum": 9.0, "wind_speed_10m_max": 14},
    {"temperature_2m_max": 25, "temperature_2m_min": 15, "precipitation_sum": 0.0, "wind_speed_10m_max": 33},
    {"temperature_2m_max": 26, "temperature_2m_min": 15, "precipitation_sum": 1.0, "wind_speed_10m_max": 10},

]


def main() -> None:
    """Load the component and print a prediction for each sample day."""
    classifier = WeatherClassifier("assignments_04/models/weather_classifier.pkl")
    for day, result in zip(SAMPLE_DAYS, classifier.predict(SAMPLE_DAYS)):
        print(f"{day['temperature_2m_max']:.0f}C, {day['precipitation_sum']:.0f}mm rain, "
              f"{day['wind_speed_10m_max']:.0f} km/h wind -> {result.label} ({result.probability:.2f})")


if __name__ == "__main__":
    main()

# If the if __name__ == "__main__": guard is omitted, importing this script from elsewhere would immediately execute all top-level code including 
# loading the model and running the main() prediction routine whenever the module is imported, rather than only running when executed directly.

# 1.Your predict() selects the four features in a fixed order rather than using whatever order the caller passed. Describe a concrete bug this prevents.
# A: If the features order are not fixed, a model silently produces wrong answers if features arrive in the wrong order.


# The graded tests train their own tiny model in a fixture instead of loading models/weather_classifier.pkl. 
# Why is that a better design for a test suite than depending on the real file?
# A: building a tiny model in a fixture and saving it to `tmp_path` means the tests do not depend on any external file and do not leave anything behind, which makes them reliable and repeatable.


# In Week 10 a pipeline will import WeatherClassifier and call predict() on batches of weather records from a database. 
# Name one thing about your component that makes it ready for that, and one thing you might add before it runs in production.
# A: I packaged and tested the component missing required feature keys and raises a helpful ValueError, preventing silent failures or unexpected model crashes during large batch runs.
#.   Batch processing optimization (Chunking/Vectorization): If the database returns tens of thousands of records at once, 
#    iterating through them sequentially or passing a massive list could cause memory pressure. 
#    Adding batch-size limits or integrating pandas/NumPy vectorization (or Extension A's DataFrame handling) would improve performance.
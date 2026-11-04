from pathlib import Path
import json
from dataclasses import dataclass
import pytest
#---------Warmup Question 1----------
#Write a class ModelInfo that reads a JSON file once in __init__ and stores its contents.
#__init__(self, path) opens the file at path, loads it with json.load, and stores the result in self.data.
#A method feature_count(self) -> int returns the number of items in self.data["features"].

#Create a small JSON file with a "features" list (you can write it from Python), build a ModelInfo from it, and print the feature count. 
# Add a comment: why is loading the file in __init__ better than loading it inside feature_count() if the method is called many times?

class ModelInfo:
    """loads the meta data of weather classifier."""
    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No file at {path}. save a file first."
            )
        with open(path, 'r', encoding="utf-8")  as f:
            self.data = json.load(f)

    def feature_count(self) -> int:
        return len(self.data["features"])

path = 'assignments_04/models/weather_classifier_metadata.json' #??????????
model_info = ModelInfo(path)
print(model_info.feature_count())
#Loading a file is slow, and loading it once in __init__ lets every later function call reuse the already-loaded model.


#---------Warmup Question 2----------
# Write a dataclass DayForecast with three type-hinted fields: date (str), high_c (float), and will_rain (bool). 
# Give it a method summary(self) -> str that returns a readable sentence such as "2023-06-15: high 24.1C, rain expected".
# Create two DayForecast objects and print each summary. 
# Add a comment: name one thing the @dataclass decorator wrote for you that you did not have to write by hand.
@dataclass
class DayForecast:
    """A class that return a readable sentence with the weather input."""
    date: str
    high_c: float
    will_rain: bool = True

    def summary(self) -> str:
        """transform the weather input into readable sentence with f-string."""
        return(f"{self.date}: high {self.high_c}C, {'rain' if self.will_rain else 'no rain'} expected")

day1_forecast = DayForecast("2026-10-31", 25.2, False)
day2_forecast = DayForecast("2026-11-01", 18)

print(day1_forecast.summary())
print(day2_forecast.summary())
# The @dataclass decorator wrote the method  __init__ for me.


#---------Warmup Question 2----------
def parse_temperature(text: str) -> float:
    """parse the temperature value to float type from string type."""
    try:
        return float(text)
    except ValueError:
        raise ValueError('temperature value must be a number')

def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)

def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
         parse_temperature("hot")
# Without match, an unrelated `ValueError` from a typo in the test setup would still make the test pass, so essentially it could miss catching the temperature value error.
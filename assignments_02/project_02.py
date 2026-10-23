import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Task 1
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


# Task 2
#check for missing values:
print(df.describe())
missing_values = df.isnull().sum()
print("number of missing values:")
print(missing_values)
# there are no missing values in the dataframe

print(df.corr(numeric_only=True)["temperature_2m_max"].sort_values())
#temp_min has a very strong positive correlation with temp_max (0.93)
plt.scatter(
    x=df['temperature_2m_min'],
    y=df['temperature_2m_max'], 
    alpha=0.7, 
    edgecolors='black'
            )
plt.title("daily low vs daily high")
plt.xlabel("daily low temperature")
plt.ylabel("daily high temperature")
plt.savefig("./assignments_02/outputs/fairfax_low_vs_high.png")
plt.show(block=False)
# From the scatter plot, I see the blue dots are scattered along a rough linear trend.

# Task3: Engineer a Feature

df['is_summer'] = ((df["date"].dt.month >=6) & (df["date"].dt.month <=8)).astype(int)
#summer_day_count = df['is_summer'].value_counts()
summer_days = (df['is_summer'] == 1).sum()
non_summer_days = (df['is_summer'] == 0).sum()
print("number of summer days:", summer_days)
print("number of non summer days:", non_summer_days )
#/I expect is_summer to help predict the daily high. because the more feature we add, the better the result.

#Task 4: Baseline Model
#Build the simplest useful model: use temperature_2m_min alone to predict temperature_2m_max. 
# Split into training and test sets (80/20, random_state=42), fit a LinearRegression model, 
# and print the slope, RMSE, and R² on the test set.
X = df[['temperature_2m_min']].values
y = df['temperature_2m_max'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

slope = model.coef_[0]
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Slope:", slope)
print("RMSE:", rmse)
print("R²:", r2)
# RMSE is about 3.63 in celsius degree. which means on a typical day, 
# the prediction for daily temperature_2m_max is off roughly by 3.63 degrees in  Celsius.

# Task 5: Full Model
# Split (80/20, random_state=42), fit a LinearRegression model, and print both train R² and test R², plus RMSE on the test set. 
# Compare the test R² to your baseline from Task 4. How much does adding features help?
feature_cols = ["temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "is_summer"]
X_full= df[feature_cols].values
y = df["temperature_2m_max"].values

X_train_f, X_test_f, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train_f, y_train)
y_train_pred = model.predict(X_train_f)
y_test_pred = model.predict(X_test_f)


train_R2 = r2_score(y_train, y_train_pred)
test_R2 = r2_score(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print("train R²: ", train_R2)
print("test R²: ", test_R2)
print("RMSE: ", rmse)

# Q: Compare the test R² to your baseline from Task 4. How much does adding features help?
# A: the test R2 is slightly higher than the baseline from task 4. by adding the precipation and wind features ,
#  it adds flexibility to the model and increase R2.

# Print each feature name with its coefficient:
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:+.3f}")
# Q: Add a comment interpreting the coefficients. Which features raise the predicted high, and which lower it? Are any signs surprising? 
# Then compare train R² to test R². Are they close, or is there a gap, and what does that tell you?
# A:  temperature_2m_min and is_summer raise the predicted high. precipitation_sum and wind_speed_10m_max lower the predicted high. the signs are not surprising.
# the train R2 is about 0.87 and test r2 is about 0.88, which is pretty close. It means that the model generalizes well and isn't severely overfitting.

#Task 6: Evaluate and Summarize
# Create a predicted vs actual plot for your full model on the test set. Add a diagonal reference line, 
# the title "Predicted vs Actual (Full Model)", labeled axes, and save to outputs/predicted_vs_actual_high.png. 
# Add a comment: does the model struggle more at the high end, the low end, or is the error roughly even across the range?
plt.figure(2)
plt.scatter(y_test_pred, y_test, color="blue", alpha=0.5, label="Data")
plt.plot([0,40], [0,40], color="red", label="reference")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted value")
plt.ylabel("Actual value")
plt.savefig("./assignments_02/outputs/predicted_vs_actual_high.png")
plt.show()

# the error ia roughly even across the range.

#The size of the dataset and the test set
# A: the dataset has 5 features with 365 data points. The test set has 5 features with 73 data points.

#The RMSE and R² of your full model, and what a typical error of that size means on a temperature scale
# A: RMSE is about 3.63 degrees Celsius. The R2 is about 0.88. the size means the predicted temperature is about 4 degrees higher/lower than the actual temperature.

# Which feature has the largest effect on the predicted high, and in which direction
# A: the temp_2m_min has the largest effect on the predicted high, it's in positive direction.

# One result that surprised you
# A: From the result, the wind_speed_10m_max feature is not having as much effect on the daily high as expected.
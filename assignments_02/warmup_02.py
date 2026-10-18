# --- scikit-learn API ---
#Q1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])
#Create a LinearRegression model, fit it to this data, and 
# then predict the high for a day with a low of 6 degrees and a day with a low of 18 degrees. 
# Print the slope (model.coef_[0]), 
# the intercept (model.intercept_), and the two predictions. Label each printed value.
X = temp_min
y = temp_max
model = LinearRegression()
model.fit(X, y)
y_pred1 = model.predict(np.array([6]).reshape(1,-1))
y_pred2 = model.predict(np.array([18]).reshape(1,-1))
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("predicted high for 6 degrees low:", y_pred1)
print("predicted high for 18 degrees low:", y_pred2)

#Q2
x = np.array([10, 20, 30, 40, 50])
print(x.shape)
x = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
print(x.shape)
# In scikit-learn, the input X must always be a 2D array with shape (num_samples, num_features), even if you only have one feature. 
# A 1D array like np.array([1, 2, 3]) would have caused an error.

# --- Linear Regression ---
rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

print("linear regression questions")
#Q1
plt.scatter(
    x=temp_min, y=temp_max,  
    c=is_rainy, 
    cmap='coolwarm', 
    alpha=0.7, 
    edgecolors='black'
)
plt.title("Daily High vs Low")
plt.xlabel("temp_min")
plt.ylabel("temp_max")
plt.savefig("./assignments_02/outputs/high_vs_low.png")
plt.show()
#Add a comment describing what you see. Are there two visible groups? What does that suggest about the is_rainy variable?
#I see a scatter plot. yes there are two visible groups distinguished by color. It means that it could have a potential influence on the model.


#Q2 Split the data into training and test sets using temp_min as the only feature, an 80/20 split, and random_state=42. 
# Reshape temp_min to a 2D array before using it as X. Print the shapes of all four arrays.
X = temp_min.reshape(-1, 1)
y = temp_max
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Q3
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_train_pred)))
print("R²:", model.score(X_test, y_test))
#Slope shows the sensitivity of temp_max on temp_min

#Q4
# Split (80/20, random_state=42), fit, and print the test R². Compare it to the R² from Question 3. 
# Does adding the rain flag help? Print both coefficients:
X_full = np.column_stack([temp_min, is_rainy])
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_full, y, test_size=0.2, random_state=42)
model_full = LinearRegression()
model_full.fit(X_train_f, y_train_f)
y_test_f_pred = model_full.predict(X_test_f)
print("R²:", model_full.score(X_test_f, y_test_f))
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])
# after adding the is_rainy.R² went up to 0.94 from last model 0.89.So it does help.

# Q5
# Using the two-feature model from Question 4, 
# create this plot for the test set. Add a diagonal reference line, the title "Predicted vs Actual", labeled axes, 
# and save to outputs/predicted_vs_actual_warmup.png.

plt.scatter(y_test_f_pred, y_test_f, color="blue", alpha=0.5, label="Data")
plt.plot([0,25], [0,25], color="red", label="reference")
plt.title("Predicted vs Actual")
plt.xlabel("predicted value")
plt.ylabel("actual value")
plt.savefig("./assignments_02/outputs/predicted_vs_actual_warmup.png")
plt.show()
# when the points fall above the diagonal, the predicted value is lower than the actual value. 
# when the points fall under the diagonal, means that the predicted values are higher that the actual values.

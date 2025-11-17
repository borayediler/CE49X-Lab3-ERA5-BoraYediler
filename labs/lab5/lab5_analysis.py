"""Lab 5 analysis script for bias-variance exploration."""

# Step 1.1 - Download the Data
# Data downloaded and added to the datasets folder

# Step 1.2 - Load it with pandas

import pandas as pd
from sklearn.model_selection import train_test_split

# File Path
file_path = r"C:\Users\EXCALIBUR\OneDrive\Masaüstü\CE49X-Fall25-master\datasets\AirQualityUCI.csv"

# Read the CSV File
df = pd.read_csv(file_path, sep=';', decimal=',', low_memory=False)

# Controls the first 5 line
df.head()

# Step 1.3 - Handle the missing values that are indicated as -200
df.replace(-200, pd.NA, inplace=True)

# Works only in the needed columns. Only these columns are converted into integers. Otherwise 
# Code did not create the graph below since it views the NAN Values as Strings.
for col in ['T', 'RH', 'AH', 'CO(GT)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Özellikler ve hedefte eksik olan satırları tamamen at
df = df.dropna(subset=['T', 'RH', 'AH', 'CO(GT)']).copy()


# Step 1.4 - Select the featured columns for modelling
features = ['T', 'RH', 'AH']
target = 'CO(GT)'

X = df[features] #The independent Variables
y = df[target] #Dependent Variable

# Step 1.5 - Split into training and testing sets

from sklearn.model_selection import train_test_split

# Train - Öğrendiği Valuelar
# Test - Test Ettiği Valuelar
# test_size= 0.3 sets our Training and Test Distribution As ½70 - ½30
# Random State Command makes our each result repeatable and not changing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 2 - Fit Models of Increasing Complexity

# REGRESSION MODELS 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error 
#MSE: How much model guesses wrongs returned as a numerical value.
 
degrees = range(1, 11)

# I opened empty lists to record the training and test errors into them respectively.
train_mse = []
test_mse = []

# This loop trains the model with testing each degree.
for d in degrees:
    # 1) Transform features
    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # 2) Train Linear Regression
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # 3) Compute MSE on train & test
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # Appending the errors to the list
    train_mse.append(mean_squared_error(y_train, y_train_pred))
    test_mse.append(mean_squared_error(y_test, y_test_pred))

# Step 3 - Plot the Validation Curve

# First plot the graph
import matplotlib.pyplot as plt

plt.plot(degrees, train_mse, label='Training Error')
plt.plot(degrees, test_mse, label='Testing Error')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Bias–Variance Tradeoff')

# Step 3.1 — Label the regions of underfitting, optimal complexity, and overfitting

import numpy as np

# Find the Best Degree, where the error is minimum.
best_degree = degrees[np.argmin(test_mse)]



plt.axvline(best_degree, color='gray', linestyle='--', linewidth=1)

# Underfitting
plt.text(1.3, max(test_mse)*0.995, 'Underfitting\n(low degree)',
         fontsize=10, color='blue', ha='center')

# Optimal (biraz sola ve aşağıya alındı)
plt.text(best_degree - 0.4, min(test_mse)*1.0015,
         f'Optimal\nDegree {best_degree}', fontsize=10, color='green', ha='left')

# Overfitting (biraz yukarı ve sağa alındı)
plt.text(9.5, max(test_mse)*0.997, 'Overfitting\n(high degree)',
         fontsize=10, color='red', ha='center')

plt.show()

# Discussion
#1. Which polynomial degree gives the best generalization?
# 9th degree. Both test and training errors are minimum. Testing error seems to increase after this degree.

#2. Describe how the training and testing errors change as degree increases.
# Both decrease. With increasing degree, both training and testing error decreases with model getting adjusted/more complex with the more complex data.
# But then, testing error starts to increase at some point which indicates overfitting.

#3. Explain how bias and variance manifest in this dataset.
# If the model is too basic, model would be too biased. If the model is too complex, it would have high variance
# In this model, it starts more basic with lower degrees (more biased) and it starts getting more complex (high variance)

#4. How might sensor noise or missing data affect the biasvariance tradeoff?
# It affects it negatively with causing an imbalance, therefore with more errors.
# With sensor noise, some great peaks may happen in the graph. Which may lead to higher variance.
# On the other hand, missing data generally decreases the variablility of the data. With it, it would be more biased.
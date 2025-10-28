# ==============================
# California Housing Price Prediction
# ==============================

# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------
# Step 2: Load California Housing Dataset
# ---------------------------------
data = fetch_california_housing(as_frame=True)
df = data.frame  # Convert to pandas DataFrame

print(" Data Loaded Successfully!")
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------
# Step 3: Data Cleaning & Exploration
# ---------------------------------
print("\nChecking for missing values:")
print(df.isnull().sum())  # Should be zero

print("\nSummary statistics:")
print(df.describe())

# Optional: Correlation heatmap to see relationships
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# ---------------------------------
# Step 4: Split Features (X) and Target (y)
# ---------------------------------
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nData split complete: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

# ---------------------------------
# Step 5: Train Linear Regression Model
# ---------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

# Evaluate
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\n Linear Regression Results:")
print(f"R² Score: {r2_lr:.3f}")
print(f"Mean Squared Error: {mse_lr:.3f}")

# ---------------------------------
# Step 6: Train Random Forest Model
# ---------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# Evaluate
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n Random Forest Results:")
print(f"R² Score: {r2_rf:.3f}")
print(f"Mean Squared Error: {mse_rf:.3f}")

# ---------------------------------
# Step 7: Compare Model Performance
# ---------------------------------
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'R² Score': [r2_lr, r2_rf],
    'Mean Squared Error': [mse_lr, mse_rf]
})
print("\nModel Performance Comparison:")
print(results)

# ---------------------------------
# Step 8: Feature Importance (Random Forest)
# ---------------------------------
importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# ---------------------------------
# Step 9: Predict vs Actual (Visualization)
# ---------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.4)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted Prices (Random Forest)")
plt.show()

print("\n Model training and evaluation completed successfully!")

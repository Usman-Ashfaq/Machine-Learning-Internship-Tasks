
#  IRIS FLOWER CLASSIFICATION


# Step 1: Import  Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------------------
# Step 2: Load Iris Dataset
# -----------------------------------------------
iris = load_iris(as_frame=True)
df = iris.frame                   # Convert to DataFrame

print(" Dataset Loaded Successfully!")
print("Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())

# -----------------------------------------------
# Step 3: Data Exploration
# -----------------------------------------------
print("\nClass Labels (Target Names):", iris.target_names)
print("\nUnique Target Values:", df['target'].unique())

# Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# -----------------------------------------------
# Step 4: Feature and Target Split
# -----------------------------------------------
X = df.drop('target', axis=1)
y = df['target']

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData Split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")

# -----------------------------------------------
# Step 5: Data Preprocessing (Standardization)
# -----------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------
# Step 6: Train Logistic Regression Model
# -----------------------------------------------
lr = LogisticRegression(max_iter=200)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)

print("\n Logistic Regression Performance:")
print(f"Accuracy: {acc_lr:.3f}")
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# -----------------------------------------------
# Step 7: Train Random Forest Model
# -----------------------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\n Random Forest Performance:")
print(f"Accuracy: {acc_rf:.3f}")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# -----------------------------------------------
# Step 8: Confusion Matrix Visualization
# -----------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, cmap="Blues", fmt='d',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# -----------------------------------------------
# Step 9: Compare Both Models
# -----------------------------------------------
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [acc_lr, acc_rf]
})
print("\nModel Comparison:\n", results)

print("\nModel Training and Evaluation Completed Successfully!")


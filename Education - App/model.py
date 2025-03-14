import pandas as pd

import numpy as np

import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("student_performance_dataset.csv")

# Ensure necessary columns exist
required_columns = ['hours_studied', 'previous_grades', 'attendance', 'final_grade']
if not all(col in df.columns for col in required_columns):
    raise KeyError("Dataset is missing required columns.")

# Select features and target variable
X = df[['hours_studied', 'previous_grades', 'attendance']]
y = df['final_grade']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model & scaler
joblib.dump(model, "student_performance_model.pkl")
joblib.dump(scaler, "scaler.pkl")


print("Model trained and saved successfully.")

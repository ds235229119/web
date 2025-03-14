import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv("sample_customer_data.csv")

# Encode categorical variables
label_encoders = {}
for col in ['Region', 'Product_Category']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoders for later use
    else:
        print(f"Warning: {col} column is missing in the dataset!")

# Define features and target
X = df.drop(columns=['CustomerID', 'Churn'])
y = df['Churn']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
with open("model.pkl", "wb") as f:
    pickle.dump((model, label_encoders), f)

print("✅ Model trained and saved successfully!")

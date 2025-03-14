import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("healthcare_dataset.csv")

# Validate required columns
required_columns = {'symptoms', 'disease', 'specialist', 'medicine', 'usage', 'side_effects', 'health_tips'}
if not required_columns.issubset(df.columns):
    raise KeyError(f"Dataset must contain the columns: {', '.join(required_columns)}")

# Handle missing values
df.fillna("Unknown", inplace=True)

# Convert symptoms column to a standardized list
df['symptoms'] = df['symptoms'].apply(lambda x: x.lower().split('|') if isinstance(x, str) else [])

# Save cleaned data
joblib.dump(df, "healthcare_model.pkl")

print("✅ Healthcare model saved successfully with cleaned data!")
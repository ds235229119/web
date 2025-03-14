from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load the trained model and encoders
with open("model.pkl", "rb") as f:
    model, label_encoders = pickle.load(f)

# Load dataset for clustering
df = pd.read_csv("sample_customer_data.csv")

# Encode categorical features for clustering
for col in ['Region', 'Product_Category']:
    if col in df.columns and col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])
    else:
        print(f"Warning: {col} is missing in label encoders!")

# Apply K-Means for customer segmentation
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df[['Spending_Score', 'Age']])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df_input = pd.DataFrame([data])

    try:
        for col in ['Region', 'Product_Category']:
            if col in label_encoders:
                df_input[col] = label_encoders[col].transform([data[col]])[0]
            else:
                return jsonify({"error": f"Invalid input: {col} not recognized"}), 400
    except Exception as e:
        return jsonify({"error": f"Encoding error: {str(e)}"}), 400

    # Ensure input columns match model features
    missing_cols = set(model.feature_names_in_) - set(df_input.columns)
    for col in missing_cols:
        df_input[col] = 0  # Fill missing columns with default values
    
    prediction = model.predict(df_input)[0]
    return jsonify({"prediction": "Likely to Churn" if prediction == 1 else "Not Likely to Churn"})

@app.route("/segmentation")
def segmentation():
    cluster_counts = df["Cluster"].value_counts().to_dict()
    return jsonify(cluster_counts)

if __name__ == "__main__":
    app.run(debug=True)
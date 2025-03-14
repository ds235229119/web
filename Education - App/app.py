from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model & scaler
model = joblib.load("student_performance_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Validate input fields
        required_fields = ['hours_studied', 'previous_grades', 'attendance']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # Convert input data to NumPy array
        features = np.array([[  
            float(data['hours_studied']),
            float(data['previous_grades']),
            float(data['attendance'])
        ]])

        # Standardize input using the same scaler used during training
        features_scaled = scaler.transform(features)

        # Predict final grade
        prediction = model.predict(features_scaled)[0]

        return jsonify({'final_grade_prediction': round(prediction, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

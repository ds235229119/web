from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from difflib import get_close_matches

app = Flask(__name__)

# Load preprocessed dataset
df = joblib.load("healthcare_model.pkl")

def find_closest_match(symptom, symptom_list):
    matches = get_close_matches(symptom, symptom_list, n=1, cutoff=0.6)
    return matches[0] if matches else None

@app.route('/')
def home():
    return render_template("index.html", health_tips=df["health_tips"].dropna().unique())

@app.route('/diagnose', methods=['POST'])
def diagnose():
    data = request.get_json()
    input_symptoms = [s.strip().lower() for s in data.get("symptoms", [])]
    all_symptoms = set(sym for sublist in df['symptoms'] for sym in sublist)

    matched_symptoms = [find_closest_match(sym, all_symptoms) for sym in input_symptoms]
    matched_symptoms = [sym for sym in matched_symptoms if sym]
    
    print("User Symptoms:", input_symptoms)  # Debugging
    print("Matched Symptoms:", matched_symptoms)  # Debugging

    if not matched_symptoms:
        return jsonify({"error": "No matching symptoms found. Try rephrasing."}), 404

    matched_diseases = {}
    for _, row in df.iterrows():
        disease_symptoms = row['symptoms']
        if any(sym in disease_symptoms for sym in matched_symptoms):
            matched_diseases[row['disease']] = row['specialist']  # Store unique diseases
    
    print("Matched Diseases:", matched_diseases)  # Debugging

    if not matched_diseases:
        return jsonify({"error": "No disease found for these symptoms."}), 404

    return jsonify({"diagnoses": [{"disease": disease, "specialist": specialist} for disease, specialist in matched_diseases.items()]})

@app.route('/doctor', methods=['GET'])
def get_doctor_recommendation():
    specialists = df["specialist"].dropna().unique().tolist()
    unique_specialists = list(set(specialists))  # Remove duplicates
    top_specialists = unique_specialists[:3]  # Limit to 3 recommendations
    return jsonify({"specialists": top_specialists})

if __name__ == "__main__":
    app.run(debug=True)

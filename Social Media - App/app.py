from flask import Flask, request, jsonify, render_template
import joblib
import re
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

def preprocess_text(text):
    """Clean and preprocess text before vectorization."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    data = request.json
    text = data.get('text', '')
    
    # Preprocess and vectorize text
    cleaned_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Predict sentiment
    sentiment = model.predict(text_vectorized)[0]
    sentiment_label = 'Positive 😊' if sentiment == 1 else 'Negative 😞'
    
    return jsonify({'sentiment': sentiment_label})

if __name__ == '__main__':
    app.run(debug=True)
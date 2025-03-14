import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess text before vectorization."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load dataset
df = pd.read_csv('social_media_data.csv')
df.dropna(subset=['text', 'sentiment'], inplace=True)  # Remove missing values

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Tokenize and vectorize the text data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment'].astype(int)  # Ensure sentiment is in numeric format

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Sentiment analysis model trained and saved successfully!")
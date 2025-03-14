from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the updated model
df = joblib.load("mood_recommendation_model.pkl")

@app.route('/')
def index():
    """Render homepage with trending movies in a card-based layout."""
    return render_template('index.html', trending_movies=df.sample(6)['title'].tolist())

@app.route('/recommend', methods=['POST'])
def recommend():
    """Recommend movies based on mood."""
    mood = request.json.get('mood', '')

    if mood not in df['mood'].explode().unique():
        return jsonify({"error": "Invalid mood or no movies available"}), 400

    mood_movies = df[df['mood'].apply(lambda x: mood in x if isinstance(x, list) else False)]
    
    if mood_movies.empty:
        return jsonify({"error": "No movies found for this mood"}), 400

    return jsonify({"recommended": mood_movies.sample(min(5, len(mood_movies)))['title'].tolist()})

@app.route('/genre-distribution', methods=['GET'])
def genre_distribution():
    """Return genre distribution for a mood."""
    mood = request.args.get('mood', '')

    mood_movies = df[df['mood'].apply(lambda x: mood in x if isinstance(x, list) else False)]
    if mood_movies.empty:
        return jsonify({"error": "No data for this mood"}), 400

    return jsonify({"genre_distribution": mood_movies['genres'].str.split('|').explode().value_counts().to_dict()})

if __name__ == '__main__':
    app.run(debug=True)

import pandas as pd
import joblib
from collections import defaultdict

# Load movie dataset
df = pd.read_csv("movies.csv")

# Ensure 'title' and 'genres' columns exist
if 'title' not in df.columns or 'genres' not in df.columns:
    raise KeyError("The dataset must contain 'title' and 'genres' columns.")

# Define mood-to-genre mapping
mood_to_genres = {
    "Happy": ["Comedy", "Animation", "Adventure", "Family"],
    "Sad": ["Drama", "Romance"],
    "Exciting": ["Action", "Thriller", "Sci-Fi"],
    "Relaxed": ["Documentary", "Biography"],
    "Fear": ["Horror", "Mystery"],
    "Fantasy": ["Fantasy", "Animation"],
}

# Reverse mapping: genre → moods
genre_to_mood = defaultdict(list)
for mood, genres in mood_to_genres.items():
    for genre in genres:
        genre_to_mood[genre].append(mood)

# Assign moods to movies based on genres
def get_moods(genres):
    genres_list = genres.split('|') if isinstance(genres, str) else []
    moods = [m for g in genres_list if g in genre_to_mood for m in genre_to_mood[g]]
    return moods if moods else ["Unknown"]  # Assign "Unknown" if no mood is found

df['mood'] = df['genres'].apply(get_moods)

# Save the processed dataset
joblib.dump(df, "mood_recommendation_model.pkl")

print("Mood-based recommendation model saved successfully!")

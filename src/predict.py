import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Preset baseline profiles aligned to Kaggle Spotify scale
CONTEXT_PRESETS = {
    "Gym / High Energy Workout": {
        "danceability": 0.75, "energy": 0.88, "valence": 0.70, "tempo": 132.0,
        "acousticness": 0.05, "instrumentalness": 0.05, "loudness": -5.0, "speechiness": 0.12, "liveness": 0.18
    },
    "Deep Focus / Study": {
        "danceability": 0.35, "energy": 0.28, "valence": 0.25, "tempo": 95.0,
        "acousticness": 0.82, "instrumentalness": 0.78, "loudness": -16.0, "speechiness": 0.04, "liveness": 0.10
    },
    "Chill & Relax": {
        "danceability": 0.58, "energy": 0.42, "valence": 0.52, "tempo": 98.0,
        "acousticness": 0.62, "instrumentalness": 0.15, "loudness": -11.0, "speechiness": 0.05, "liveness": 0.12
    },
    "Party & Dance": {
        "danceability": 0.86, "energy": 0.84, "valence": 0.82, "tempo": 124.0,
        "acousticness": 0.08, "instrumentalness": 0.02, "loudness": -4.8, "speechiness": 0.10, "liveness": 0.20
    },
    "Rainy / Melancholy": {
        "danceability": 0.42, "energy": 0.22, "valence": 0.18, "tempo": 84.0,
        "acousticness": 0.86, "instrumentalness": 0.12, "loudness": -17.0, "speechiness": 0.04, "liveness": 0.11
    }
}

class ContextPlaylistRecommender:
    def __init__(self, model_dir="models"):
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        self.kmeans = joblib.load(os.path.join(model_dir, "kmeans_model.pkl"))
        self.feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
        self.df = pd.read_parquet(os.path.join(model_dir, "processed_tracks.parquet"))

        # Precompute scaled feature matrix for fast similarity search
        self.scaled_matrix = self.scaler.transform(self.df[self.feature_cols])

    def recommend_by_context(self, context_name, target_genre="All", top_n=10):
        if context_name not in CONTEXT_PRESETS:
            raise ValueError(f"Unknown context: {context_name}")
        return self.recommend_by_vector(CONTEXT_PRESETS[context_name], target_genre=target_genre, top_n=top_n)

    def recommend_by_vector(self, feature_dict, target_genre="All", top_n=10):
        # Construct input vector matching exact feature columns
        input_vector = np.array([[feature_dict[col] for col in self.feature_cols]])
        scaled_input = self.scaler.transform(input_vector)

        # Filter by genre if specified
        if target_genre != "All" and "track_genre" in self.df.columns:
            subset_mask = self.df["track_genre"].str.lower() == target_genre.lower()
            subset_df = self.df[subset_mask].reset_index(drop=True)
            subset_matrix = self.scaled_matrix[subset_mask]
        else:
            subset_df = self.df
            subset_matrix = self.scaled_matrix

        if len(subset_df) == 0:
            return pd.DataFrame()

        # Compute cosine similarity
        similarities = cosine_similarity(scaled_input, subset_matrix).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]

        results = subset_df.iloc[top_indices].copy()
        results["Match Score (%)"] = (similarities[top_indices] * 100).round(1)

        display_cols = ["track_name", "artists", "album_name", "track_genre", "popularity", "Match Score (%)"]
        return results[[c for c in display_cols if c in results.columns]]

if __name__ == "__main__":
    recommender = ContextPlaylistRecommender()
    recs = recommender.recommend_by_context("Deep Focus / Study", top_n=5)
    print(recs)
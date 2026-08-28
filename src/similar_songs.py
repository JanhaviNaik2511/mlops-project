import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class SimilarSongRecommender:

    def __init__(self, model_dir="models"):

        # Load scaler used by the original project
        self.scaler = joblib.load(
            os.path.join(model_dir, "scaler.pkl")
        )

        # Load exact audio feature columns
        self.feature_cols = joblib.load(
            os.path.join(model_dir, "feature_cols.pkl")
        )

        # Load processed music dataset
        self.df = pd.read_parquet(
            os.path.join(model_dir, "processed_tracks.parquet")
        )

        # Create scaled feature matrix
        self.scaled_matrix = self.scaler.transform(
            self.df[self.feature_cols]
        )

    def search_songs(self, search_text, limit=20):

        """
        Search songs by track name or artist name.
        """

        if not search_text:
            return pd.DataFrame()

        search_text = search_text.strip().lower()

        mask = (
            self.df["track_name"]
            .fillna("")
            .str.lower()
            .str.contains(search_text, regex=False)
            |
            self.df["artists"]
            .fillna("")
            .str.lower()
            .str.contains(search_text, regex=False)
        )

        results = self.df[mask].copy()

        display_cols = [
            "track_name",
            "artists",
            "album_name",
            "track_genre",
            "popularity"
        ]

        available_cols = [
            col for col in display_cols
            if col in results.columns
        ]

        return results[available_cols].head(limit)

    def recommend_similar(
        self,
        track_name,
        artist=None,
        top_n=10,
        target_genre="All"
    ):

        """
        Find songs similar to the selected song
        using cosine similarity.
        """

        # Find the selected song
        mask = (
            self.df["track_name"]
            .fillna("")
            .str.lower()
            == track_name.strip().lower()
        )

        # If artist is provided, use it to identify the exact song
        if artist is not None and "artists" in self.df.columns:

            artist_mask = (
                self.df["artists"]
                .fillna("")
                .str.lower()
                == artist.strip().lower()
            )

            mask = mask & artist_mask

        matching_indices = np.where(mask)[0]

        if len(matching_indices) == 0:
            return pd.DataFrame()

        # Use the first matching song
        selected_index = matching_indices[0]

        selected_vector = self.scaled_matrix[
            selected_index
        ].reshape(1, -1)

        # Genre filtering
        if (
            target_genre != "All"
            and "track_genre" in self.df.columns
        ):

            subset_mask = (
                self.df["track_genre"]
                .fillna("")
                .str.lower()
                == target_genre.lower()
            )

            subset_df = self.df[subset_mask].copy()

            subset_matrix = self.scaled_matrix[
                subset_mask.values
            ]

            original_indices = np.where(subset_mask.values)[0]

        else:

            subset_df = self.df.copy()
            subset_matrix = self.scaled_matrix
            original_indices = np.arange(len(self.df))

        if len(subset_df) == 0:
            return pd.DataFrame()

        # Calculate cosine similarity
        similarities = cosine_similarity(
            selected_vector,
            subset_matrix
        ).flatten()

        # Remove the selected song itself
        similarities[
            original_indices == selected_index
        ] = -1

        # Get top similar songs
        top_indices = similarities.argsort()[::-1][:top_n]

        results = subset_df.iloc[top_indices].copy()

        results["Similarity (%)"] = (
            similarities[top_indices] * 100
        ).round(1)

        display_cols = [
            "track_name",
            "artists",
            "album_name",
            "track_genre",
            "popularity",
            "Similarity (%)"
        ]

        available_cols = [
            col for col in display_cols
            if col in results.columns
        ]

        return results[available_cols]


# Test the feature when this file is run directly
if __name__ == "__main__":

    recommender = SimilarSongRecommender()

    print("\nSearching for songs containing 'love':")

    search_results = recommender.search_songs(
        "love",
        limit=10
    )

    print(search_results)

    if len(search_results) > 0:

        selected_song = search_results.iloc[0]

        print(
            f"\nFinding songs similar to: "
            f"{selected_song['track_name']} "
            f"by {selected_song['artists']}"
        )

        recommendations = recommender.recommend_similar(
            track_name=selected_song["track_name"],
            artist=selected_song["artists"],
            top_n=10
        )

        print("\nSimilar Songs:")
        print(recommendations)
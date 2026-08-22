import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Audio feature columns present in the Kaggle dataset
AUDIO_FEATURES = [
    'danceability',
    'energy',
    'valence',
    'tempo',
    'acousticness',
    'instrumentalness',
    'loudness',
    'speechiness',
    'liveness'
]

def train_model(data_path="data/dataset.csv", model_dir="models", n_clusters=6):
    print("1. Loading dataset...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'. Place Kaggle's dataset.csv in 'data/' folder.")

    df = pd.read_csv(data_path)

    # Clean and filter duplicates
    metadata_cols = ['track_id', 'artists', 'track_name', 'album_name', 'track_genre', 'popularity']
    required_cols = [c for c in metadata_cols if c in df.columns] + AUDIO_FEATURES
    
    df = df[required_cols].dropna().drop_duplicates(subset=['track_name', 'artists']).reset_index(drop=True)
    print(f"Loaded {len(df):,} unique tracks across {df['track_genre'].nunique() if 'track_genre' in df.columns else 'N/A'} genres.")

    # 2. Standardize numerical audio features
    print("2. Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df[AUDIO_FEATURES])

    # 3. Fit K-Means Clustering
    print(f"3. Training K-Means model (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_matrix)

    # 4. Save trained models and processed data
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(model_dir, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(AUDIO_FEATURES, os.path.join(model_dir, "feature_cols.pkl"))

    # Save as parquet for faster read times during inference
    df.to_parquet(os.path.join(model_dir, "processed_tracks.parquet"), index=False)
    print(f"Artifacts successfully saved to '{model_dir}/'")

if __name__ == "__main__":
    train_model()
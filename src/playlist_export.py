import pandas as pd


def export_playlist(results, filename="recommended_playlist.csv"):
    """
    Export recommended songs to a CSV file.

    Parameters:
        results: Pandas DataFrame containing recommended songs
        filename: Name of the output CSV file

    Returns:
        CSV data as bytes for Streamlit download
    """

    # Create a copy so the original recommendation results
    # are not modified.
    playlist = results.copy()

    # Keep only useful columns if they exist
    preferred_columns = [
        "track_name",
        "artists",
        "track_genre",
        "popularity",
        "Similarity (%)"
    ]

    available_columns = [
        column for column in preferred_columns
        if column in playlist.columns
    ]

    if available_columns:
        playlist = playlist[available_columns]

    # Convert DataFrame to CSV
    csv_data = playlist.to_csv(index=False)

    return csv_data.encode("utf-8")
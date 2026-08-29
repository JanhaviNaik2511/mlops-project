import streamlit as st
from src.predict import ContextPlaylistRecommender, CONTEXT_PRESETS
from src.playlist_export import export_playlist
# Rebase workflow demonstration

st.set_page_config(
    page_title="Context Music Generator",
    page_icon="🎧",
    layout="wide"
)

st.title("🎧 Smart Context-Aware Playlist Generator")
st.caption(
    "Powered by K-Means Clustering & Cosine Similarity "
    "on Kaggle Spotify 114k Tracks"
)


@st.cache_resource
def load_model():
    return ContextPlaylistRecommender()


try:
    engine = load_model()

    # Extract unique genres for optional filtering
    available_genres = [
        "All"
    ] + sorted(
        engine.df["track_genre"]
        .dropna()
        .unique()
        .tolist()
    )

    col_left, col_right = st.columns(
        [1, 2],
        gap="medium"
    )

    # =========================================================
    # LEFT COLUMN - CONFIGURATION
    # =========================================================

    with col_left:

        st.subheader("1. Configure Context")

        mode = st.radio(
            "Selection Mode",
            ["Activity Preset", "Custom Sliders"]
        )

        selected_genre = st.selectbox(
            "Filter by Genre (Optional)",
            available_genres
        )

        num_tracks = st.slider(
            "Playlist Length",
            min_value=5,
            max_value=30,
            value=10
        )

        if mode == "Activity Preset":

            preset = st.selectbox(
                "Select Activity / Mood",
                list(CONTEXT_PRESETS.keys())
            )

        else:

            st.markdown("**Fine-tune Audio Dynamics:**")

            danceability = st.slider(
                "Danceability",
                0.0,
                1.0,
                0.65
            )

            energy = st.slider(
                "Energy",
                0.0,
                1.0,
                0.70
            )

            valence = st.slider(
                "Valence (Happiness)",
                0.0,
                1.0,
                0.50
            )

            tempo = st.slider(
                "Tempo (BPM)",
                60,
                200,
                120
            )

            acousticness = st.slider(
                "Acousticness",
                0.0,
                1.0,
                0.20
            )

            instrumentalness = st.slider(
                "Instrumentalness",
                0.0,
                1.0,
                0.10
            )

            custom_vector = {
                "danceability": danceability,
                "energy": energy,
                "valence": valence,
                "tempo": float(tempo),
                "acousticness": acousticness,
                "instrumentalness": instrumentalness,
                "loudness": -7.0,
                "speechiness": 0.06,
                "liveness": 0.15
            }

        generate = st.button(
            "Generate Playlist",
            use_container_width=True
        )

    # =========================================================
    # RIGHT COLUMN - RECOMMENDATIONS
    # =========================================================

    with col_right:

        st.subheader("2. Recommended Playlist")

        if generate:

            if mode == "Activity Preset":

                results = engine.recommend_by_context(
                    preset,
                    target_genre=selected_genre,
                    top_n=num_tracks
                )

            else:

                results = engine.recommend_by_vector(
                    custom_vector,
                    target_genre=selected_genre,
                    top_n=num_tracks
                )

            # =================================================
            # DISPLAY RESULTS
            # =================================================

            if len(results) > 0:

                st.dataframe(
                    results,
                    use_container_width=True
                )

                # =================================================
                # NEW FEATURE:
                # DOWNLOAD RECOMMENDED PLAYLIST AS CSV
                # =================================================

                st.markdown("### 📥 Export Playlist")

                csv_data = export_playlist(results)

                st.download_button(
                    label="⬇️ Download Playlist as CSV",
                    data=csv_data,
                    file_name="recommended_playlist.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.success(
                    "Playlist generated successfully! "
                    "You can download it using the button above."
                )

            else:

                st.warning(
                    "No tracks found for the selected genre filter."
                )

        else:

            st.info(
                "Choose your mood/activity parameters and click "
                "**Generate Playlist**."
            )


except Exception as err:

    st.error(
        f"Initialization Error: {err}"
    )

    st.warning(
        "Ensure `dataset.csv` is in `data/` and you ran "
        "`python src/train.py` first."
    )
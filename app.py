import streamlit as st
import pandas as pd

from src.predict import ContextPlaylistRecommender, CONTEXT_PRESETS
from src.similar_songs import SimilarSongRecommender


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


@st.cache_resource
def load_similar_song_model():
    return SimilarSongRecommender()


try:
    engine = load_model()
    similar_engine = load_similar_song_model()

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

    with col_left:

        st.subheader("1. Configure Context")

        mode = st.radio(
            "Selection Mode",
            [
                "Activity Preset",
                "Custom Sliders",
                "Similar Songs"
            ]
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

        # -------------------------------------------------
        # EXISTING FEATURE 1: ACTIVITY PRESET
        # -------------------------------------------------

        if mode == "Activity Preset":

            preset = st.selectbox(
                "Select Activity / Mood",
                list(CONTEXT_PRESETS.keys())
            )

        # -------------------------------------------------
        # EXISTING FEATURE 2: CUSTOM SLIDERS
        # -------------------------------------------------

        elif mode == "Custom Sliders":

            st.markdown(
                "**Fine-tune Audio Dynamics:**"
            )

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

        # -------------------------------------------------
        # YOUR NEW FEATURE: SIMILAR SONGS
        # -------------------------------------------------

        else:

            st.markdown(
                "**Find Songs Similar To:**"
            )

            song_search = st.text_input(
                "Search by song name or artist",
                placeholder="Example: love, imagine, weeknd..."
            )

            selected_song = None

            if song_search:

                search_results = similar_engine.search_songs(
                    song_search,
                    limit=20
                )

                if len(search_results) > 0:

                    song_options = [
                        f"{row['track_name']} — {row['artists']}"
                        for _, row in search_results.iterrows()
                    ]

                    selected_song_option = st.selectbox(
                        "Select a song",
                        song_options
                    )

                    selected_position = song_options.index(
                        selected_song_option
                    )

                    selected_song = search_results.iloc[
                        selected_position
                    ]

                else:

                    st.warning(
                        "No songs found. Try another search."
                    )

        generate = st.button(
            "Generate Playlist",
            use_container_width=True
        )

    # =====================================================
    # RIGHT SIDE: RECOMMENDATIONS
    # =====================================================

    with col_right:

        st.subheader("2. Recommended Playlist")

        if generate:

            # -------------------------------------------------
            # ACTIVITY PRESET
            # -------------------------------------------------

            if mode == "Activity Preset":

                results = engine.recommend_by_context(
                    preset,
                    target_genre=selected_genre,
                    top_n=num_tracks
                )

            # -------------------------------------------------
            # CUSTOM SLIDERS
            # -------------------------------------------------

            elif mode == "Custom Sliders":

                results = engine.recommend_by_vector(
                    custom_vector,
                    target_genre=selected_genre,
                    top_n=num_tracks
                )

            # -------------------------------------------------
            # SIMILAR SONGS
            # -------------------------------------------------

            else:

                if selected_song is not None:

                    results = similar_engine.recommend_similar(
                        track_name=selected_song["track_name"],
                        artist=selected_song["artists"],
                        target_genre=selected_genre,
                        top_n=num_tracks
                    )

                else:

                    results = pd.DataFrame()

            # -------------------------------------------------
            # DISPLAY RESULTS
            # -------------------------------------------------

            if len(results) > 0:

                st.dataframe(
                    results,
                    use_container_width=True
                )

            else:

                if mode == "Similar Songs":

                    st.warning(
                        "Please search for and select a song "
                        "before generating recommendations."
                    )

                else:

                    st.warning(
                        "No tracks found for the selected "
                        "genre filter."
                    )

        else:

            st.info(
                "Choose your mood/activity parameters "
                "and click **Generate Playlist**."
            )


except Exception as err:

    st.error(
        f"Initialization Error: {err}"
    )

    st.warning(
        "Ensure `dataset.csv` is in `data/` and you ran "
        "`python src/train.py` first."
    )
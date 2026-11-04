"""Streamlit content-based Spotify track discovery demo."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spotify_segmentation import recommend_similar_songs  # noqa: E402


st.set_page_config(page_title="Spotify Genre Segmentation", page_icon="🎧", layout="wide")
st.title("Spotify Songs: Genre Segmentation")
st.write(
    "Explore audio-based song clusters and find tracks with similar audio features. "
    "Playlist genre and name are used to inspect the clusters after they are formed."
)
st.info(
    "This is an offline educational demo based on a public 2020 dataset. "
    "It does not connect to Spotify or create playlists in your account."
)

model_path = PROJECT_ROOT / "outputs" / "spotify_segmentation_model.joblib"
songs_path = PROJECT_ROOT / "outputs" / "clustered_songs.csv"
if not model_path.exists() or not songs_path.exists():
    st.warning("Run `python train.py` from the project folder before starting the demo.")
    st.stop()


@st.cache_resource
def load_resources():
    bundle = joblib.load(model_path)
    songs = pd.read_csv(songs_path, low_memory=False)
    transformed = bundle["preprocessor"].transform(songs[bundle["audio_features"]])
    return bundle, songs, transformed


bundle, songs, transformed = load_resources()
left, right = st.columns([1, 1])
with left:
    st.metric("Tracks / playlist entries", f"{len(songs):,}")
with right:
    st.metric("Audio clusters", int(bundle["cluster_count"]))

st.subheader("Find a starting song")
genre_options = ["All genres", *sorted(songs["playlist_genre"].dropna().unique().tolist())]
selected_genre = st.selectbox("Filter by playlist genre", genre_options)
search = st.text_input("Search by song or artist", placeholder="Try a song title or artist name")

matches = songs.copy()
if selected_genre != "All genres":
    matches = matches[matches["playlist_genre"] == selected_genre]
if search.strip():
    mask = (
        matches["track_name"].astype(str).str.contains(search, case=False, na=False, regex=False)
        | matches["track_artist"].astype(str).str.contains(search, case=False, na=False, regex=False)
    )
    matches = matches[mask]
matches = matches.sort_values("track_popularity", ascending=False).head(500)

if matches.empty:
    st.warning("No matching song was found. Try a shorter search or another genre.")
    st.stop()

row_ids = matches["row_id"].astype(int).tolist()
song_lookup = songs.set_index("row_id")

def song_label(row_id: int) -> str:
    row = song_lookup.loc[row_id]
    return f"{row['track_name']} — {row['track_artist']} · {row['playlist_genre']} · cluster {int(row['cluster'])}"


selected_row_id = st.selectbox("Choose a song", row_ids, format_func=song_label)
same_cluster = st.checkbox("Recommend only from the same audio cluster", value=True)
number = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

if st.button("Find similar songs", type="primary"):
    seed = song_lookup.loc[selected_row_id]
    st.caption(
        f"Starting track: **{seed['track_name']} — {seed['track_artist']}** "
        f"(cluster {int(seed['cluster'])})"
    )
    recommendations = recommend_similar_songs(
        songs,
        transformed,
        int(selected_row_id),
        n_recommendations=number,
        same_cluster_only=same_cluster,
    )
    recommendations["similarity"] = recommendations["similarity"].map(lambda value: f"{value:.3f}")
    st.dataframe(recommendations, use_container_width=True, hide_index=True)

with st.expander("What do the clusters represent?"):
    profiles_path = PROJECT_ROOT / "outputs" / "cluster_profiles.csv"
    if profiles_path.exists():
        st.write("The values below are average audio features within each cluster. Cluster numbers are labels, not genre names.")
        st.dataframe(pd.read_csv(profiles_path), use_container_width=True, hide_index=True)

# Spotify Songs' Genre Segmentation

An end-to-end unsupervised learning project that groups Spotify playlist tracks by audio characteristics, compares the discovered clusters with playlist genres and names, and recommends similar songs from the resulting audio space.

## Project goals

- Clean and explore a playlist track dataset.
- Visualize genre balance, audio feature distributions, popularity patterns, missing data, and correlations.
- Segment tracks with K-Means using audio features, selecting the cluster count with a silhouette-score sweep.
- Inspect clusters by playlist genre and playlist name.
- Recommend similar tracks using cosine similarity over the processed audio features.

## Dataset and attribution

This project uses the [Spotify Songs dataset published for TidyTuesday on 2020-01-21](https://github.com/rfordatascience/tidytuesday/tree/main/data/2020/2020-01-21). The CSV and its data dictionary are included under `data/`. The source readme says the data came from Spotify via the `spotifyr` package; it credits Charlie Thompson, Josiah Parry, Donal Phipps, and Tom Wolff, with a hat tip to Jon Harmon and Neal Grantham. The TidyTuesday repository lists a [CC0-1.0 license](https://github.com/rfordatascience/tidytuesday/blob/main/LICENSE). This analysis is independent and is not affiliated with or endorsed by Spotify.

The dataset contains track metadata, playlist name, playlist genre/subgenre, popularity, and audio attributes. Playlist genre is a property of the source playlist, not a verified ground-truth genre label for every song. Playlist names can also describe mood or activity rather than genre.

## Features used for clustering

Continuous audio features: `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, and `duration_ms`. `key` and `mode` are treated as categories and one-hot encoded. Popularity, track names, artists, playlist names, playlist genres, and IDs are left out of the clustering input so metadata does not directly define the clusters.

Missing continuous values are imputed with the median; categorical values use the most frequent value. Continuous audio attributes are standardized and the key/mode categories are one-hot encoded inside the preprocessing pipeline. Duplicate rows for the same `(playlist_id, track_id)` are removed, while the same track can remain in different playlists so playlist membership can be analyzed.

The Pearson correlation matrix includes the numeric `key` and `mode` codes for completeness; correlations involving these two coded categories should be interpreted cautiously.

## Setup

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the analysis

The bundled CSV means the analysis does not need an API key or a Spotify connection.

```powershell
python train.py
```

You can supply a replacement CSV with the same columns:

```powershell
python train.py --data path\to\spotify_songs.csv
```

The script evaluates `k = 2` through `k = 8` on a seeded sample, selects the highest sampled silhouette score, then fits the selected K-Means model on all cleaned rows. It reports Adjusted Rand Index and normalized mutual information against playlist genre as descriptive external checks. These are not ground-truth accuracy scores: playlist labels are noisy context labels and are not used to fit the clusters.

### Example results

On the bundled data, cleaning produced **32,250 track-playlist rows** for **28,355 unique tracks**, removing 582 repeated entries for the same track within the same playlist. The best silhouette score in the tested range was **0.164 at k = 2** (the next best was 0.123 at k = 4). This is a modest score, so the resulting two groups should be viewed as broad, overlapping segments rather than sharply separated music types.

| Cluster | Rows | Energy (mean) | Loudness (mean dB) | Acousticness (mean) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 9,451 | 0.492 | -9.73 | 0.371 |
| 1 | 22,799 | 0.785 | -5.48 | 0.094 |

Cluster 0 is quieter and more acoustic on average; Cluster 1 is more energetic and louder on average. Playlist genre and audio clusters align only weakly (ARI **0.020**, NMI **0.032**). For example, 89% of EDM playlist entries fall in Cluster 1, while R&B playlist entries are split almost evenly between clusters. That mismatch illustrates why playlist labels are used as context for interpretation, not as ground truth.

![PCA projection of the audio clusters](outputs/plots/pca_cluster_map.png)

![Audio cluster shares by playlist genre](outputs/plots/clusters_by_playlist_genre.png)

The training script also writes `sample_recommendations.csv`, showing a cosine-similarity recommendation list for the most popular seed track.

## Outputs

`outputs/` contains:

- `clustered_songs.csv` with the assigned cluster for each cleaned track-playlist row
- `spotify_segmentation_model.joblib` with the fitted preprocessing and K-Means models
- `cluster_count_scores.csv` and `clustering_summary.json`
- `sample_recommendations.csv` for one example seed track
- `cluster_profiles.csv`, `genre_cluster_distribution.csv`, and `top_playlist_cluster_distribution.csv`
- EDA plots under `outputs/plots/`, including the correlation matrix, feature distributions, cluster profiles, PCA map, silhouette curve, cluster-by-genre heatmap, and top playlist comparisons

The PCA chart is a two-dimensional projection for visualization; cluster assignment uses the full processed feature matrix.

## Recommendation demo

Run the model once, then start the local app:

```powershell
streamlit run app.py
```

Choose a seed song, then the app ranks other tracks by cosine similarity across the processed audio features. You can limit recommendations to the seed song's cluster. The demo is content-based and offline: it does not log in to Spotify, call Spotify's API, play music, or write a playlist to your account.

## Notebook

```powershell
jupyter notebook notebooks/spotify_genre_segmentation.ipynb
```

## Upload to GitHub

Create an empty repository on GitHub, then run the following from this project folder and replace the URL with your repository's HTTPS URL:

```powershell
git init
git add .
git commit -m "Add Spotify song genre segmentation project"
git branch -M main
git remote add origin https://github.com/<USERNAME>/<REPOSITORY>.git
git push -u origin main
```

## Limitations

- Clusters are mathematical groupings of audio features, not official Spotify genres.
- Playlist labels and names are editorial/user-curated context, not verified labels for each track.
- The dataset is historical and does not provide current listening counts or user-specific behavior.
- Similarity scores describe feature proximity, not an individual listener's taste.

"""Clean the Spotify playlist data, choose a cluster count, and save results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spotify_segmentation import (  # noqa: E402
    create_eda_plots,
    load_data,
    recommend_similar_songs,
    run_clustering,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "spotify_songs.csv",
        help="CSV containing playlist names, playlist genres, and Spotify audio features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help="Directory for figures, clustered rows, metrics, and the model bundle.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning Spotify playlist data...", flush=True)
    data, cleaning_report = load_data(args.data)
    print(
        f"Clean rows: {len(data):,}; unique tracks: {data['track_id'].nunique():,}; "
        "creating EDA plots...",
        flush=True,
    )
    (output_dir / "cleaning_report.json").write_text(
        json.dumps(cleaning_report, indent=2), encoding="utf-8"
    )
    create_eda_plots(data, output_dir / "plots")
    print("Selecting cluster count and fitting final K-Means model...", flush=True)
    clustered, scores, result = run_clustering(data, output_dir)
    bundle = result["model_bundle"]
    transformed = bundle["preprocessor"].transform(clustered[bundle["audio_features"]])
    seed_row_id = int(clustered.sort_values("track_popularity", ascending=False).iloc[0]["row_id"])
    seed = clustered.iloc[seed_row_id]
    recommendations = recommend_similar_songs(
        clustered, transformed, seed_row_id, n_recommendations=10
    )
    recommendations.to_csv(output_dir / "sample_recommendations.csv", index=False)
    print("\nCluster-count scores")
    print(scores.round(3).to_string(index=False))
    print(
        f"\nExample recommendations for {seed['track_name']} — {seed['track_artist']} "
        f"(cluster {int(seed['cluster'])})"
    )
    print(recommendations[["track_name", "track_artist", "playlist_genre", "similarity"]].to_string(index=False))
    print("\nClustering summary")
    print(json.dumps({key: value for key, value in result.items() if key not in {"model_bundle", "playlist_genre_codes"}}, indent=2))
    print(f"\nRows saved: {len(clustered):,}")
    print(f"Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()

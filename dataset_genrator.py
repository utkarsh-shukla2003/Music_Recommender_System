import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import pandas as pd

sp = spotipy.Spotify(
    auth_manager=SpotifyClientCredentials(
        "1332d405ad36413b81cdb56b9bc9dda4", "6ceea3fc05c846c8bc3f25427cb8f11d"
    )
)


def fetch_songs(year):
    # Use Spotify API to search for tracks released in the given year
    results = sp.search(q=f"year:{year}", type="track", limit=50)
    songs = results["tracks"]["items"]

    # Extract relevant information for each song
    song_data = []
    for song in songs:
        song_info = {
            "name": song["name"],
        }

        song_data.append(song_info)

    return song_data


all_songs_data = []
for year in range(2020, 2024):
    for _ in range(3):
        all_songs_data.extend(fetch_songs(year))

# Convert song data to DataFrame
songs_df = pd.DataFrame(all_songs_data)

# Save DataFrame to CSV file
songs_df.to_csv("spotify_songs_2020_2023.csv", index=False)

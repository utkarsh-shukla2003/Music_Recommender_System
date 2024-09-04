from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import Spotify
from dataclasses import dataclass
from json import JSONDecodeError
from flask import json
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
from scipy.spatial.distance import cdist
from random import shuffle

songs_dataframe = pd.read_csv("dataset/data.csv")
number_cols: list[str] = [
    "valence",
    "year",
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "explicit",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "popularity",
    "speechiness",
    "tempo",
]


song_cluster_pipeline = pickle.load(
    open("model/pickle-files/song-cluster-pipeline.pkl", "rb")
)
data = pickle.load(open("model/pickle-files/data.pkl", "rb"))


@dataclass
class Song:
    name: str
    release_year: int
    artists: list[str]

    @classmethod
    def from_dataframe(cls, df: dict):
        name: str = str(df["name"])
        release_year: int = int(str(df["year"]))
        try:
            artists: list[str] = json.loads(str(df["artists"].replace("'", '"')))[:5]
        except JSONDecodeError:
            artists = [str(df["artists"])[:30].replace("[", "").replace("]", "")]
        return cls(name, release_year, artists)


def search_song(song_name: str, limit=20) -> list[Song]:
    # query dataframe using song name
    filtered_df = songs_dataframe[songs_dataframe["name"].str.contains(song_name)].head(
        limit
    )

    # sort dataframe with popularity
    sorted_df = filtered_df.sort_values(by="popularity", ascending=False)

    # convert data frame to list of object and return
    search_result = list(
        map(
            lambda df: Song.from_dataframe(df),
            sorted_df.to_dict(orient="records"),
        )
    )
    shuffle(search_result)
    return search_result


song_cluster_pipeline = pickle.load(
    open("model/pickle-files/song-cluster-pipeline.pkl", "rb")
)
data = pickle.load(open("model/pickle-files/data.pkl", "rb"))


sp = Spotify(
    auth_manager=SpotifyClientCredentials(
        "1332d405ad36413b81cdb56b9bc9dda4", "6ceea3fc05c846c8bc3f25427cb8f11d"
    )
)


class SpotifySDKException(Exception):
    pass


class TrackPosterNotFoundException(Exception):
    pass


def get_track_poster_image_url(song: str, year: int) -> str:
    results = sp.search(q=f"track: {song} year:{year}", limit=1)
    if results is None:
        raise SpotifySDKException("Couldn't search using Spotify API")

    if not results["tracks"]["items"]:
        raise TrackPosterNotFoundException()

    track = results["tracks"]["items"][0]
    album_id = track["album"]["id"]

    album_details = sp.album(album_id)
    if album_details is None:
        raise SpotifySDKException(f"Couldn't find album using album id: {album_id}")

    if album_details["images"]:
        return album_details["images"][0]["url"]

    raise TrackPosterNotFoundException()


def find_song(song_name: str, release_year: int):
    song_data = defaultdict()
    results = sp.search(q=f"track: {song_name} year:{release_year}", limit=1)
    if results["tracks"]["items"] == []:
        return None

    results = results["tracks"]["items"][0]
    track_id = results["id"]
    audio_features = sp.audio_features(track_id)[0]

    song_data["name"] = [song_name]
    song_data["year"] = [release_year]
    song_data["explicit"] = [int(results["explicit"])]
    song_data["duration_ms"] = [results["duration_ms"]]
    song_data["popularity"] = [results["popularity"]]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


def get_song_data(song, spotify_data):

    try:
        song_data = spotify_data[
            (spotify_data["name"] == song["name"])
            & (spotify_data["year"] == song["year"])
        ].iloc[0]

        return song_data

    except IndexError:
        return find_song(song["name"], song["year"])


def get_mean_vector(song_list, spotify_data):

    song_vectors = []

    for song in song_list:

        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print(
                "Warning: {} does not exist in Spotify or in database".format(
                    song["name"]
                )
            )
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(song_vectors)

    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):

    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict


def recommend_songs(songs: list[Song], limit=10) -> list[Song]:

    metadata_cols = ["name", "year", "artists"]
    songs = list(map(lambda i: {"name": i.name, "year": i.release_year}, songs))
    song_dict = flatten_dict_list(songs)

    song_center = get_mean_vector(songs, data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, "cosine")
    index = list(np.argsort(distances)[:, :limit][0])

    rec_songs = data.iloc[index]
    rec_songs = rec_songs[~rec_songs["name"].isin(song_dict["name"])]

    return list(
        map(
            lambda df: Song.from_dataframe(df),
            rec_songs[metadata_cols].to_dict(orient="records"),
        )
    )

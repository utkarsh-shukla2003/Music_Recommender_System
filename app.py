from flask import Flask, render_template
from flask import request as req

from service import music_service

app = Flask(
    __name__,
    static_url_path="",
    static_folder="web/static",
    template_folder="web/templates",
)

playlist: list[music_service.Song] = list()


@app.route("/")
def index():
    playlist.clear()
    return render_template("pages/index.html")


@app.route("/search", methods=["POST"])
def search_songs():
    query = req.form["query"]
    if not query or len(query) < 2:
        return render_template(
            "partials/error.html", msg="At least enter 2 characters to search songs"
        )

    # get music searches
    search_results = music_service.search_song(query, limit=50)

    return render_template("partials/search-response.html", results=search_results)


@app.route("/add-to-playlist", methods=["POST"])
def add_to_playlist():
    name = req.form["name"]
    year = int(req.form["year"])

    playlist.append(music_service.Song(name=name, release_year=year, artists=[]))
    return render_template("partials/playlist.html", playlist=playlist)


@app.route("/recommend", methods=["POST"])
def recommend_songs():
    song_names = req.form.getlist("name")
    song_release_years = req.form.getlist("year")

    playlist_songs = [
        music_service.Song(name=name, release_year=int(year), artists=[])
        for name, year in zip(song_names, song_release_years)
    ]

    recommedations = music_service.recommend_songs(playlist_songs)

    return render_template(
        "partials/recommendations.html",
        recommendations=recommedations,
        playlist=playlist_songs,
    )

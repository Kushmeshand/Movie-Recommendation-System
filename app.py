import streamlit as st
import requests
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import io

#-----------------theme-------------------------------#
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.9)),
    url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
    
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Make text white */
h1, h2, h3, h4, h5, h6, p, div {
    color: white;
}

/* Buttons */
.stButton>button {
    background-color: #e50914;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
}

/* Selectbox */
.stSelectbox label {
    color: white;
}
</style>
""", unsafe_allow_html=True)
# ---------------- LOAD PKL FROM DRIVE ----------------
def load_pkl_from_drive(url):
    response = requests.get(url)
    return pickle.load(io.BytesIO(response.content))

collab_similarity = load_pkl_from_drive(
    "https://drive.google.com/uc?export=download&id=1hLX4egbrSd4hG2Qf1jhB5x5SumqtqO5V"
)

collab_movies = load_pkl_from_drive(
    "https://drive.google.com/uc?export=download&id=15XF2K6hnPOHJG8e3RI-jnyXdxci43yrU"
)

API_KEY = "4e4b10932b7c2b31fd1e0a074c80f0c9"
OMDB_KEY = "e15bce82"

# ---------------- CLEAN TITLE ----------------
def clean_title(title):
    import re
    title = re.sub(r"\(.*?\)", "", title)
    title = re.sub(r"a\.k\.a.*", "", title, flags=re.IGNORECASE)
    if ", The" in title:
        title = "The " + title.replace(", The", "")
    return title.strip()

# ---------------- SESSION ----------------
if "selected_movie_details" not in st.session_state:
    st.session_state.selected_movie_details = None
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
# ---------------- API ----------------
def fetch_poster(movie_title):
    try:
        cleaned = clean_title(movie_title)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={cleaned}"
        data = requests.get(url).json()
        if data["results"]:
            poster = data["results"][0].get("poster_path")
            if poster:
                return "https://image.tmdb.org/t/p/w500" + poster
    except:
        pass
    return "https://via.placeholder.com/300x450.png?text=No+Image"
def fetch_trending():
    try:
        url = f"https://api.themoviedb.org/3/trending/movie/day?api_key={API_KEY}"
        data = requests.get(url).json()

        movies = []
        for m in data["results"][:5]:
            poster = "https://image.tmdb.org/t/p/w500" + m["poster_path"] if m.get("poster_path") else ""
            movies.append({
                "title": m["title"],
                "poster": poster,
                "rating": round(m["vote_average"], 1)
            })

        return movies
    except:
        return []

def fetch_details(movie_title):
    try:
        cleaned = clean_title(movie_title)

        search = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={cleaned}"
        data = requests.get(search).json()

        if not data["results"]:
            return None

        movie = data["results"][0]
        movie_id = movie["id"]
        overview = movie.get("overview")

        movie_info = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
        ).json()

        runtime = movie_info.get("runtime")
        release_date = movie_info.get("release_date")
        rating = movie_info.get("vote_average")
        popularity = movie_info.get("popularity")
        votes = movie_info.get("vote_count")
        videos = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}"
        ).json()

        trailer = None
        for v in videos["results"]:
            if v["type"] == "Trailer":
                trailer = v["key"]
                break

        credits = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
        ).json()

        cast = credits["cast"][:5]
        director = next((c for c in credits["crew"] if c["job"] == "Director"), None)

        omdb = requests.get(
            f"http://www.omdbapi.com/?t={cleaned}&apikey={OMDB_KEY}"
        ).json()

        rt_rating = "N/A"
        if omdb.get("Ratings"):
            for r in omdb["Ratings"]:
                if r["Source"] == "Rotten Tomatoes":
                    rt_rating = r["Value"]

        return {
            "trailer": trailer,
            "cast": cast,
            "director": director,
            "runtime": runtime,
            "release_date": release_date,
            "rating": rating,
            "rt": rt_rating,
            "popularity": popularity,
            "votes": votes,
            "overview": overview
        }

    except:
        return None

# ---------------- DATA ----------------
@st.cache_data
def load_data():
    movies = pd.read_csv("https://drive.google.com/uc?export=download&id=1MvT-iG8f837YFmZdXzcuuhhfh4WYTZQq")
    credits = pd.read_csv("https://drive.google.com/uc?export=download&id=15EBSjEpdVoSrtPRQIuv_fvxlbgQeSuTZ")
    return movies, credits


@st.cache_data
def preprocess():
    movies, credits = load_data()
    movies = movies.merge(credits, on="title")

    movies = movies[['id','title','overview','genres','keywords','cast','crew','vote_average']]
    movies.dropna(inplace=True)

    def convert(text):
        return [i['name'] for i in ast.literal_eval(text)]

    def convert_cast(text):
        return [i['name'] for i in ast.literal_eval(text)[:3]]

    def fetch_director(text):
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)

    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

    new_df = movies[['id','title','tags','overview','vote_average']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

    return new_df


@st.cache_data
def create_model(df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    return cosine_similarity(vectors)

new_df = preprocess()
similarity = create_model(new_df)

# ---------------- HYBRID ----------------
def hybrid_recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]

    content_scores = {}
    for i, score in enumerate(distances):
        title = new_df.iloc[i].title
        content_scores[title] = score

    collab_scores = {}

    match = None
    for m in collab_movies:
        if movie.lower() in m.lower():
            match = m
            break

    if match:
        idx = collab_movies.index(match)
        distances = collab_similarity[idx]

        for i, score in enumerate(distances):
            collab_scores[collab_movies[i]] = score

    # 🔥 NORMALIZATION
    max_content = max(content_scores.values())
    max_collab = max(collab_scores.values()) if collab_scores else 1

    final_scores = {}

    for title in content_scores:
        norm_content = content_scores[title] / max_content if max_content else 0
        norm_collab = collab_scores.get(title, 0) / max_collab if max_collab else 0

        final_scores[title] = 0.6 * norm_content + 0.4 * norm_collab

    sorted_movies = sorted(final_scores.items(),
                           key=lambda x: x[1],
                           reverse=True)[1:6]

    results = []
    for title, score in sorted_movies:
        results.append({
            "title": title,
            "poster": fetch_poster(title),
            "score": round(score * 100, 1)   # 🔥 convert to %
        })

    return results

# ---------------- UI ----------------
st.markdown("""
<h1 style='text-align:center; font-size:60px; color:#e50914;'>
CineMatch AI
</h1>
<p style='text-align:center; font-size:22px;'>
Unlimited Movie Recommendations 
</p>
""", unsafe_allow_html=True)
st.subheader("🔥 Trending Now")

trending = fetch_trending()
cols = st.columns(5)

for i, movie in enumerate(trending):
    with cols[i]:
        st.image(movie["poster"])
        st.write(movie["title"])
        st.write(f"⭐ {movie['rating']}")
st.subheader("❤️ My Watchlist")

if "watchlist" in st.session_state and st.session_state.watchlist:
    for item in st.session_state.watchlist:
        st.write("🎬", item)
else:
    st.write("No movies added yet")
selected_movie = st.selectbox("Select a movie", new_df['title'])

if st.button("Recommend"):
    st.session_state.recommendations = hybrid_recommend(selected_movie)

if "recommendations" in st.session_state:

    st.subheader("🎯 Recommended Movies")

    cols = st.columns(5)

    for i, movie in enumerate(st.session_state.recommendations):
        with cols[i]:
            st.image(movie["poster"])
            st.write(movie["title"])
            st.write(f"🔥 Score: {movie['score']}%")

            if st.button("View Details", key=i):
                st.session_state.selected_movie_details = movie
            if st.button("❤️ Add Watchlist", key=f"watch{i}"):
               if movie["title"] not in st.session_state.watchlist:
                  st.session_state.watchlist.append(movie["title"])
# ---------------- DETAILS ----------------
if st.session_state.selected_movie_details:
    movie = st.session_state.selected_movie_details
    details = fetch_details(movie["title"])
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
    
    st.markdown("---")
    st.header(movie["title"])

    if details:
        st.write(f"⭐ IMDb: {details['rating']}")
        st.write(f"🍅 Rotten Tomatoes: {details['rt']}")
        st.write(f"📅 Release Date: {details['release_date']}")
        st.write(f"⏱ Runtime: {details['runtime']} min")
    st.subheader("📝 Overview")
    if details and details.get("overview"):
        st.write(details["overview"])

    if details and details.get("trailer"):
        st.subheader("🎬 Trailer")
        st.video(f"https://www.youtube.com/watch?v={details['trailer']}")

    st.subheader("🎥 Director")
    if details and details.get("director"):
        director = details["director"]
        if director.get("profile_path"):
            st.image("https://image.tmdb.org/t/p/w200" + director["profile_path"])
        st.write(director["name"])

    st.subheader("👥 Cast")
    if details and details.get("cast"):
        cols = st.columns(5)
        for i, actor in enumerate(details["cast"]):
            with cols[i]:
                if actor.get("profile_path"):
                    st.image("https://image.tmdb.org/t/p/w200" + actor["profile_path"])
                st.write(actor["name"])

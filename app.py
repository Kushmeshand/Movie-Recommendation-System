import streamlit as st
import requests
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import io

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

# API KEYS
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
            "overview": overview
        }

    except:
        return None

# ---------------- REDDIT ----------------
def fetch_reddit_reviews(movie):
    try:
        query = clean_title(movie) + " movie"
        url = f"https://www.reddit.com/search.json?q={query}&limit=5&sort=top"
        headers = {"User-agent": "Mozilla/5.0"}

        data = requests.get(url, headers=headers).json()

        reviews = []
        for post in data["data"]["children"]:
            p = post["data"]

            reviews.append({
                "title": p["title"],
                "subreddit": p["subreddit"],
                "score": p["score"],
                "comments": p["num_comments"],
                "url": "https://reddit.com" + p["permalink"]
            })

        return reviews
    except:
        return []
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

# ---------------- CONTENT ----------------
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    results = []
    for i in movies_list:
        row = new_df.iloc[i[0]]
        results.append({
            "title": row.title,
            "poster": fetch_poster(row.title),
            "rating": round(row.vote_average, 1),
            "overview": " ".join(row.overview)
        })
    return results

# ---------------- COLLAB ----------------
def recommend_collab(movie):
    match = None
    for m in collab_movies:
        if movie.lower() in m.lower():
            match = m
            break

    if match is None:
        return []

    idx = collab_movies.index(match)
    distances = collab_similarity[idx]

    movies_list = sorted(list(enumerate(distances)),
                         key=lambda x: x[1],
                         reverse=True)[1:6]

    return [collab_movies[i[0]] for i in movies_list]

# ---------------- UI ----------------
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Select a movie", new_df['title'])

if st.button("Recommend"):
    st.session_state.cb = recommend(selected_movie)
    st.session_state.cf = recommend_collab(selected_movie)

if "cb" in st.session_state:

    st.subheader("🎯 Content-Based Recommendations")
    cols = st.columns(5)

    for i, movie in enumerate(st.session_state.cb):
        with cols[i]:
            st.image(movie["poster"])
            st.write(movie["title"])
            st.write(f"⭐ {movie['rating']}")

            if st.button("View Details", key=f"cb{i}"):
                st.session_state.selected_movie_details = movie

    st.subheader("🤝 Collaborative Recommendations")

    if st.session_state.cf:
        cols = st.columns(len(st.session_state.cf))

        for i, title in enumerate(st.session_state.cf):
            with cols[i]:
                st.image(fetch_poster(title))
                st.write(title)

                details = fetch_details(title)
                if details and details.get("rating"):
                    st.write(f"⭐ {round(details['rating'],1)}")
                else:
                    st.write("⭐ N/A")

                if st.button("View Details", key=f"cf{i}"):
                    st.session_state.selected_movie_details = {
                        "title": title,
                        "overview": ""
                    }

# ---------------- DETAILS ----------------
if st.session_state.selected_movie_details:
    movie = st.session_state.selected_movie_details
    details = fetch_details(movie["title"])

    st.markdown("---")
    st.header(movie["title"])

    if details:
        st.write(f"⭐ IMDb: {details['rating']}")
        st.write(f"🍅 Rotten Tomatoes: {details['rt']}")

    # -------- OVERVIEW --------
    st.subheader("📝 Overview")
    if details and details.get("overview"):
        st.write(details["overview"])
    else:
        st.write("Overview not available")

    # -------- TRAILER --------
    if details and details.get("trailer"):
        st.subheader("🎬 Trailer")
        st.video(f"https://www.youtube.com/watch?v={details['trailer']}")

    # -------- DIRECTOR --------
    st.subheader("🎥 Director")
    if details and details.get("director"):
        director = details["director"]

        if director.get("profile_path"):
            st.image("https://image.tmdb.org/t/p/w200" + director["profile_path"])

        st.write(director["name"])

    # -------- CAST --------
    st.subheader("👥 Cast")
    if details and details.get("cast"):
        cols = st.columns(5)

        for i, actor in enumerate(details["cast"]):
            with cols[i]:
                if actor.get("profile_path"):
                    st.image("https://image.tmdb.org/t/p/w200" + actor["profile_path"])
                st.write(actor["name"])

    # -------- REDDIT --------
    st.subheader("💬 Reddit Reviews")
    reviews = fetch_reddit_reviews(movie["title"])

    if reviews:
        for r in reviews:
            st.markdown(f"**r/{r['subreddit']}**")
            st.markdown(f"[{r['title']}]({r['url']})")
            st.write(f"⬆️ {r['score']}   💬 {r['comments']}")
            st.markdown("---")
    else:
        st.write("No Reddit reviews found")

import streamlit as st
import requests
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

API_KEY = "4e4b10932b7c2b31fd1e0a074c80f0c9"
OMDB_KEY = "e15bce82"

# ---------------- COLLAB MODEL ----------------
collab_similarity = pickle.load(open("collab_similarity.pkl", "rb"))
collab_movies = pickle.load(open("collab_movies.pkl", "rb"))

# ---------------- SESSION ----------------
if "selected_movie_details" not in st.session_state:
    st.session_state.selected_movie_details = None

# ---------------- API ----------------
def fetch_poster(movie_title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
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
        search = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
        data = requests.get(search).json()

        if not data["results"]:
            return None

        movie_id = data["results"][0]["id"]

        movie_info = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}").json()

        runtime = movie_info.get("runtime")
        release_date = movie_info.get("release_date")
        rating = movie_info.get("vote_average")

        videos = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}").json()
        trailer = None
        for v in videos["results"]:
            if v["type"] == "Trailer":
                trailer = v["key"]
                break

        credits = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}").json()
        cast = credits["cast"][:5]
        director = next((c for c in credits["crew"] if c["job"] == "Director"), None)

        providers_data = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={API_KEY}"
        ).json()

        providers = []
        if providers_data.get("results") and providers_data["results"].get("IN"):
            region = providers_data["results"]["IN"]

            if "flatrate" in region:
                for p in region["flatrate"]:
                    providers.append({
                        "name": p["provider_name"],
                        "logo": "https://image.tmdb.org/t/p/w200" + p["logo_path"],
                        "type": "Subscription"
                    })

            if "free" in region:
                for p in region["free"]:
                    providers.append({
                        "name": p["provider_name"],
                        "logo": "https://image.tmdb.org/t/p/w200" + p["logo_path"],
                        "type": "Free"
                    })

        omdb = requests.get(f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_KEY}").json()

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
            "providers": providers
        }

    except:
        return None

# ---------------- REDDIT ----------------
def fetch_reddit_reviews(movie):
    try:
        url = f"https://www.reddit.com/search.json?q={movie}&limit=5"
        headers = {"User-agent": "Mozilla/5.0"}
        data = requests.get(url, headers=headers).json()

        reviews = []
        for post in data["data"]["children"]:
            reviews.append({
                "title": post["data"]["title"],
                "url": "https://reddit.com" + post["data"]["permalink"]
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

# ---------------- CONTENT RECOMMEND ----------------
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
    if movie not in collab_movies:
        return []

    idx = list(collab_movies).index(movie)
    distances = collab_similarity[idx]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    return [collab_movies[i[0]] for i in movies_list]

# ---------------- UI ----------------
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Select a movie", new_df['title'])

if st.button("Recommend"):
    st.session_state.cb = recommend(selected_movie)
    st.session_state.cf = recommend_collab(selected_movie)

# ---------------- SHOW BOTH ----------------
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

                if st.button("View Details", key=f"cf{i}"):
                    st.session_state.selected_movie_details = {
                        "title": title,
                        "overview": "Overview not available"
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

    st.subheader("📝 Overview")
    st.write(movie["overview"])

    # -------- REDDIT --------
    st.subheader("💬 Reddit Reviews")

    reviews = fetch_reddit_reviews(movie["title"])

    if reviews:
        for r in reviews:
            st.markdown(f"[{r['title']}]({r['url']})")
    else:
        st.write("No Reddit reviews found")

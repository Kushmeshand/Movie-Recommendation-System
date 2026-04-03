import streamlit as st
import requests
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- API ----------------
API_KEY = "4e4b10932b7c2b31fd1e0a074c80f0c9"

def fetch_poster(movie_title):
    try:
        query = movie_title.replace(" ", "%20")
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"
        data = requests.get(url, timeout=5).json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path

        return "https://via.placeholder.com/300x450.png?text=No+Poster"

    except:
        return "https://via.placeholder.com/300x450.png?text=Error"


def fetch_details(movie_title):
    try:
        search = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
        data = requests.get(search).json()

        if not data['results']:
            return None

        movie_id = data['results'][0]['id']

        # trailer
        videos = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}").json()
        trailer = None
        for v in videos['results']:
            if v['type'] == 'Trailer':
                trailer = v['key']
                break

        # cast & crew
        credits = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}").json()

        cast = credits['cast'][:5]
        director = next((c for c in credits['crew'] if c['job'] == 'Director'), None)

        return {"trailer": trailer, "cast": cast, "director": director}

    except:
        return None


# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center; color:#E50914;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- LOAD ----------------
@st.cache_data
def load_data():
    movies = pd.read_csv("https://drive.google.com/uc?export=download&id=1MvT-iG8f837YFmZdXzcuuhhfh4WYTZQq")
    credits = pd.read_csv("https://drive.google.com/uc?export=download&id=15EBSjEpdVoSrtPRQIuv_fvxlbgQeSuTZ")
    return movies, credits


@st.cache_data
def preprocess():
    movies, credits = load_data()

    movies = movies.merge(credits, on='title')
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

# ---------------- RECOMMEND ----------------
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    names, posters, ratings, overviews, matches = [], [], [], [], []

    for i in movies_list:
        row = new_df.iloc[i[0]]

        names.append(row.title)
        posters.append(fetch_poster(row.title))
        ratings.append(round(row.vote_average, 1))
        overviews.append(" ".join(row.overview))
        matches.append(round(i[1]*100, 1))

    return names, posters, ratings, overviews, matches


# ---------------- UI ----------------
selected_movie = st.selectbox("Select a movie", new_df['title'])

if st.button("Recommend"):
    names, posters, ratings, overviews, matches = recommend(selected_movie)

    st.subheader("🎯 Recommended Movies")

    selected = st.radio("Select a movie for details", names)

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.image(posters[i])
            st.write(names[i])
            st.write(f"⭐ {ratings[i]}")
            st.write(f"🔥 {matches[i]}%")

    # ---------------- DETAILS ----------------
    idx = names.index(selected)
    details = fetch_details(selected)

    st.markdown("---")
    st.header(selected)

    st.subheader("📝 Overview")
    st.write(overviews[idx])

    if details and details["trailer"]:
        st.subheader("🎬 Trailer")
        st.video(f"https://www.youtube.com/watch?v={details['trailer']}")

    if details and details["director"]:
        st.subheader("🎥 Director")
        st.write(details["director"]["name"])

    if details and details["cast"]:
        st.subheader("👥 Cast")
        cast_cols = st.columns(5)

        for i, actor in enumerate(details["cast"]):
            with cast_cols[i]:
                if actor.get("profile_path"):
                    st.image(f"https://image.tmdb.org/t/p/w200{actor['profile_path']}")
                st.write(actor["name"])

import streamlit as st
import requests

API_KEY = "4e4b10932b7c2b31fd1e0a074c80f0c9"

def fetch_poster(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
    data = requests.get(url).json()
    
    if data['results']:
        poster_path = data['results'][0].get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    
    return "https://via.placeholder.com/300x450.png?text=No+Image"
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.markdown("<h1 style='text-align:center; color:#E50914;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
body {
    background-color: #0E1117;
}
h1 {
    color: #E50914;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    movies = pd.read_csv("https://drive.google.com/uc?export=download&id=1MvT-iG8f837YFmZdXzcuuhhfh4WYTZQq")
    credits = pd.read_csv("https://drive.google.com/uc?export=download&id=15EBSjEpdVoSrtPRQIuv_fvxlbgQeSuTZ")
    return movies, credits

# ---------------- PREPROCESS ----------------
@st.cache_data
def preprocess():
    movies, credits = load_data()
    
    movies = movies.merge(credits, on='title')
    movies = movies[['id','title','overview','genres','keywords','cast','crew','vote_average']]
    movies.dropna(inplace=True)

    def convert(text):
        return [i['name'] for i in ast.literal_eval(text)]

    def convert_cast(text):
        L = []
        for i in ast.literal_eval(text)[:3]:
            L.append(i['name'])
        return L

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

# ---------------- MODEL ----------------
@st.cache_data
def create_model(new_df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

new_df = preprocess()
similarity = create_model(new_df)
st.markdown("---")
# ---------------- RECOMMEND ----------------
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]

    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    names = []
    posters = []
    ratings = []
    overviews = []

    for i in movies_list:
        row = new_df.iloc[i[0]]
        
        names.append(row.title)
        posters.append(fetch_poster(row.title))
        ratings.append(row.vote_average)
        overviews.append(" ".join(row.overview))

    return names, posters, ratings, overviews
# ---------------- UI ----------------
selected_movie = st.selectbox("Select a movie", new_df['title'])

if st.button("Recommend"):
    names, posters, ratings, overviews = recommend(selected_movie)

    st.subheader("🎯 Recommended Movies")

    cols = st.columns(3)

    for i in range(5):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="
                background-color:#1c1c1c;
                padding:15px;
                border-radius:12px;
                text-align:center;
                margin-bottom:20px;
            ">
                <img src="{posters[i]}" width="100%" style="border-radius:10px;">
                
                <h4 style="
                    color:white;
                    font-size:16px;
                    white-space:nowrap;
                    overflow:hidden;
                    text-overflow:ellipsis;
                ">
                    {names[i]}
                </h4>

                <p style="color:gold; font-size:14px;">⭐ {ratings[i]}</p>

                <p style="
                    color:#ccc;
                    font-size:13px;
                    height:120px;
                    overflow-y:auto;
                    text-align:left;
                ">
                    {overviews[i]}
                </p>
            </div>
            """, unsafe_allow_html=True)

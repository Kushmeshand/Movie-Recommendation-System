import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")

# Load data from Google Drive
@st.cache_data
def load_data():
    movies = pd.read_csv("https://drive.google.com/uc?export=download&id=1MvT-iG8f837YFmZdXzcuuhhfh4WYTZQq")
    credits = pd.read_csv("https://drive.google.com/uc?export=download&id=15EBSjEpdVoSrtPRQIuv_fvxlbgQeSuTZ")
    return movies, credits

movies, credits = load_data()

# Merge
movies = movies.merge(credits, on='title')
movies = movies[['id','title','overview','genres','keywords','cast','crew']]
movies.dropna(inplace=True)

# Convert functions
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Create tags
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)),
                         reverse=True,
                         key=lambda x: x[1])[1:6]

    return [new_df.iloc[i[0]].title for i in movies_list]

# UI
selected_movie = st.selectbox("Select a movie", new_df['title'])

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.write(movie)

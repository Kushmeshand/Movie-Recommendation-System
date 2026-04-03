import streamlit as st
import pickle
import pandas as pd

st.title("🎬 Movie Recommendation System")

try:
    movies = pickle.load(open('movies.pkl','rb'))
    similarity = pickle.load(open('similarity.pkl','rb'))

    st.write("✅ Data loaded successfully")

    movie_list = movies['title'].values
    selected_movie = st.selectbox("Select a movie", movie_list)

    def recommend(movie):
        index = movies[movies['title'] == movie].index[0]
        distances = similarity[index]
        movies_list = sorted(list(enumerate(distances)),
                             reverse=True,
                             key=lambda x: x[1])[1:6]

        recommended_movies = []
        for i in movies_list:
            recommended_movies.append(movies.iloc[i[0]].title)
        return recommended_movies

    if st.button("Recommend"):
        recommendations = recommend(selected_movie)
        for movie in recommendations:
            st.write(movie)

except Exception as e:
    st.error(f"❌ ERROR: {e}")

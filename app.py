import streamlit as st
import pickle as pkl
import pandas as pd
import requests

st.title('Movie Recommender System')

movies_dict = pkl.load(open('./movie_list.pkl', 'rb'))
similarity = pkl.load(open('similarity.pkl', 'rb'))

movies = pd.DataFrame(movies_dict)
print(movies)

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    try:
        response = requests.get(url, timeout=5)  # Set timeout to 5 seconds
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            # Return a placeholder image if poster_path is missing
            full_path = "https://via.placeholder.com/500x750?text=No+Image"
    except requests.exceptions.RequestException as e:
        # Log error and return a placeholder image
        print(f"Error fetching poster for movie ID {movie_id}: {e}")
        full_path = "https://via.placeholder.com/500x750?text=No+Image"
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])[1:6]
    recommended = []
    recommended_posters = []
    for i in distances:
        movie_id = movies.iloc[i[0]].movie_id
        recommended.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
    return recommended, recommended_posters

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    cols = st.columns(5)
    for idx, col in enumerate(cols):
        with col:
            st.text(recommended_movie_names[idx])
            st.image(recommended_movie_posters[idx])
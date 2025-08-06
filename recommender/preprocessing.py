import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(movies_path='data/movies.csv', ratings_path='data/ratings.csv'):
    df_movies = pd.read_csv(movies_path)
    df_ratings = pd.read_csv(ratings_path)
    return df_movies, df_ratings

def clean_movies(df_movies):
    # Remove invalid or duplicate movies
    bad_movie_ids = [26958, 168358, 6003, 32600, 64997]
    df_movies = df_movies[~df_movies['movieId'].isin(bad_movie_ids)]
    df_movies = df_movies[df_movies['genres'] != '(no genres listed)']
    
    # Extract year from title
    df_movies['year'] = df_movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    
    # Normalize year
    scaler = MinMaxScaler()
    df_movies['normalized_year'] = scaler.fit_transform(df_movies[['year']])
    
    return df_movies

def clean_ratings(df_ratings, movie_replacement_map=None):
    if movie_replacement_map:
        df_ratings['movieId'] = df_ratings['movieId'].replace(movie_replacement_map)
    return df_ratings



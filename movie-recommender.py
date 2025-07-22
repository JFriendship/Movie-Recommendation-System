import numpy as np
import pandas as pd

# -=| Data Loading |=-
movies_file_path = 'Dataset/movies.csv'
ratings_file_path = 'Dataset/ratings.csv'

df_movies = pd.read_csv(movies_file_path)
df_ratings = pd.read_csv(ratings_file_path)

# -=| Feature Engineering |=-
# These movie ids are sourced from the EDA
bad_movie_ids = [26958, 168358, 6003, 32600, 64997]
replacement_movie_ids = [838, 2851, 144606, 147002, 34048]

# Remove Duplicates
boolean_mask_remove = ~df_movies['movieId'].isin(bad_movie_ids)
df_movies = df_movies[boolean_mask_remove]

# Replace Rating Duplicates
movie_replacement_map = {
    26958: 838,
    168358: 2851,
    6003: 144606,
    32600: 147002,
    64997: 34048
}
# df_ratings['movieId'].replace(bad_movie_ids, replacement_movie_ids)
df_ratings['movieId'].replace(movie_replacement_map)

# Remove less active users and movies
user_rating_threshold = 20
user_rating_counts = df_ratings['userId'].value_counts()
users_to_keep = user_rating_counts[user_rating_counts >= user_rating_threshold].index
df_ratings = df_ratings[df_ratings['userId'].isin(users_to_keep)]

ratings_per_movie_threshold = 10
movie_rating_counts = df_ratings['movieId'].value_counts()
movies_to_keep = movie_rating_counts[movie_rating_counts >= ratings_per_movie_threshold].index
df_ratings = df_ratings[df_ratings['movieId'].isin(movies_to_keep)]

# Multi-hot Encoding Genres
genre_split = df_movies['genres'].str.split('|')
flattened_genre_split = [item for sublist in genre_split for item in sublist]
unique_genres = set(flattened_genre_split)

genre_split = genre_split.to_frame()
df_genres = genre_split.reindex(genre_split.columns.tolist() + list(unique_genres), axis=1, fill_value=0)

for idx, genre_list in enumerate(genre_split['genres']):
    for genre in genre_list:
        df_genres.loc[idx, genre] = 1

df_genres = df_genres.drop(['genres'], axis=1)
df_movies = df_movies.drop(['genres'], axis=1)

df_movies = pd.concat([df_movies, df_genres], axis=1)

# User-Item Matrix
user_item_matrix = df_ratings.pivot(index='userId', columns='movieId', values='rating')

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
for bad_id in bad_movie_ids:
    df_movies.drop(df_movies[df_movies['movieId'] == bad_id].index, inplace=True)

# Replace Rating Duplicates
df_ratings['movieId'].replace(bad_movie_ids, replacement_movie_ids, inplace=True)

# -=| Multi-hot Encoding Genres |=-
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
print(df_movies.head())
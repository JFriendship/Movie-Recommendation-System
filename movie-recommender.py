import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(user_id, num_recommendations=5, movies_rated_by_user=None):
    user_similarities = df_user_movie_similarities.loc[user_id]

    if movies_rated_by_user is None:
        movies_rated_by_user = df_ratings[df_ratings['userId'] == user_id]['movieId'].values

    movies_rated_by_user = movies_rated_by_user[np.isin(movies_rated_by_user, df_movies['movieId'])]

    available_movies = user_similarities.drop(index=movies_rated_by_user, errors='ignore')
    print(f"Available movies for recommendation: {len(available_movies)}")

    df_recommendations = user_similarities.drop(index=movies_rated_by_user, errors='ignore').sort_values(ascending=False).head(num_recommendations)

    movie_subset = df_movies[['movieId', 'title']]
    df_recommendations = df_recommendations.reset_index(name='score').rename(columns={'index': 'movieId'})
    return movie_subset.merge(df_recommendations, on='movieId')

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
df_ratings['movieId'] = df_ratings['movieId'].replace(movie_replacement_map)

# Removing movies with "(no genres listed)"
df_movies = df_movies[df_movies['genres'] != '(no genres listed)']
df_ratings = df_ratings[df_ratings['movieId'].isin(df_movies['movieId'])]

# Remove less active users and movies
user_rating_threshold = 20
user_rating_counts = df_ratings['userId'].value_counts()
users_to_keep = user_rating_counts[user_rating_counts >= user_rating_threshold].index
df_ratings = df_ratings[df_ratings['userId'].isin(users_to_keep)]

ratings_per_movie_threshold = 10
movie_rating_counts = df_ratings['movieId'].value_counts()
movies_to_keep = movie_rating_counts[movie_rating_counts >= ratings_per_movie_threshold].index
df_ratings = df_ratings[df_ratings['movieId'].isin(movies_to_keep)]
df_movies = df_movies[df_movies['movieId'].isin(movies_to_keep)]    # Have to remove them from both dataframes

# Multi-hot Encoding Genres
genre_split = df_movies['genres'].str.split('|')
df_genres = (genre_split.explode().str.strip().pipe(pd.get_dummies).groupby(level=0).sum())

df_genres = df_genres.set_index(df_movies['movieId'])

df_ratings = df_ratings.merge(df_movies[['movieId', 'title']], on='movieId')
df_ratings_with_genres = df_ratings.join(df_genres, on='movieId')

user_profiles = df_ratings_with_genres.groupby('userId')[np.array(df_genres.columns.tolist())].apply(lambda df: np.average(df, axis=0, weights=df_ratings_with_genres.loc[df.index, 'rating']))
user_profiles = pd.DataFrame(user_profiles.tolist(), index=user_profiles.index, columns=np.array(df_genres.columns.tolist()))
user_profiles = user_profiles.fillna(0) # assume preference for a genre is 0 if the user hasn't rated a movie in said genre

print(df_ratings_with_genres.head())

# print("User profile for user 2:\n", user_profiles.loc[2])
# print("Sum of profile vector:", user_profiles.loc[2].sum())

similarity_matrix = cosine_similarity(user_profiles.values, df_genres.values)
df_user_movie_similarities = pd.DataFrame(similarity_matrix, index=user_profiles.index, columns=df_genres.index)

recommendations = recommend_movies(user_id=4, num_recommendations=10)
print(recommendations)

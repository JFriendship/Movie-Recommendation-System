import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def create_user_profiles(df_ratings, df_movies, df_genres):
    """
    Creates a feature dataframe which is then used to create user profiles.
    
    Args:
        df_ratings (pd.Dataframe): The user ratings that are going to be used to create the user profiles.
        df_movies (pd.Dataframe): Movie data.
        df_genres (pd.Dataframe): Multi-hot encoded genre data.
    
    Returns:
        pd.Dataframe: The user profiles
    """

    df_ratings = df_ratings.merge(df_movies[['movieId', 'title']], on='movieId')
    df_features = df_ratings.join(df_genres, on='movieId')
    df_features = df_features.merge(df_movies[['movieId', 'normalized_year']], on='movieId', how='left')
    df_features = df_features.rename(columns={'normalized_year': 'year'})

    user_profiles = df_features.groupby('userId')[np.array(df_genres.columns.tolist())].apply(lambda df: np.average(df, axis=0, weights=df_features.loc[df.index, 'rating']))
    user_profiles = pd.DataFrame(user_profiles.tolist(), index=user_profiles.index, columns=np.array(df_genres.columns.tolist()))
    return user_profiles.fillna(0) # assume preference for a genre is 0 if the user hasn't rated a movie in said genre

def compute_similarity_matrix(user_profiles, df_genres):
    """
    Computes a similarity matrix
    
    Args:
        user_profiles (pd.Dataframe): User profiles.
        df_genres (pd.Dataframe): A dataframe with multi-hot encoded genre data.
        
    Returns:
        pd.Dataframe: A similarity matrix"""
    
    similarity_matrix = cosine_similarity(user_profiles.values, df_genres.values)
    return pd.DataFrame(similarity_matrix, index=user_profiles.index, columns=df_genres.index)

def recommend_movies(user_id, df_user_movie_similarities, df_ratings, df_movies, num_recommendations=10):
    """
    Provides the top-k recommended movies for a given user.
    
    Args:
        user_id (int): The user's id that the recommendations are for.
        df_user_movie_similarities (pd.Dataframe): A user-movie similarity matrix.
        df_ratings (pd.Dataframe): Users' movie rating data.
        df_movies (pd.Dataframe): Movie data.
        num_recommendations (int): the number of recommendations the function should output.
        
    Returns: 
        pd.Dataframe: The top-k recommended movies for the user, along with their scores.
    """

    # Get the similarity vector for this user
    user_similarities = df_user_movie_similarities.loc[user_id]

    # Exclude already rated items
    train_movies = df_ratings[df_ratings['userId'] == user_id]['movieId'].values
    available_movies = user_similarities.drop(index=train_movies, errors='ignore')

    # Recommend top-k items
    top_k = available_movies.sort_values(ascending=False).head(num_recommendations)
    top_k = top_k.reset_index(name='score').rename(columns={'index': 'movieId'})

    return df_movies[['movieId', 'title']].merge(top_k, on='movieId').sort_values(by='score', ascending=False)

def recommend_movies_all_users(user_movie_similarities, df_ratings, df_movies, num_recommendations=10):
    """
    Returns a dictionary of top-k movie IDs per user for every user.

    Args:
        user_movie_similarities (pd.Dataframe): A user-movie similarity matrix.
        df_ratings (pd.Dataframe): The training ratings that will be excluded from recommendations.
        df_movies (pd.Dataframe): Movie data.
        num_recommendations (int): Number of recommendations per user.
    
    Returns:
        pd.Dataframe: All of the top-k recommended movies for each user.
    """

    all_recommendations = {}
    for user_id in df_ratings['userId'].unique():
        recommendations = recommend_movies(user_id=user_id, df_user_movie_similarities=user_movie_similarities, df_ratings=df_ratings, df_movies=df_movies, num_recommendations=num_recommendations)
        all_recommendations[user_id] = recommendations['movieId'].tolist()

    return all_recommendations

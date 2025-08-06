import numpy as np
import pandas as pd


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

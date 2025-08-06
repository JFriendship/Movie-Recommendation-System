import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(movies_path='data/movies.csv', ratings_path='data/ratings.csv'):
    df_movies = pd.read_csv(movies_path)
    df_ratings = pd.read_csv(ratings_path)
    return df_movies, df_ratings

def clean_movies(df_movies):
    """
    Removes invalid and duplicate movies. Also extracts year values from movies' title strings and normalizes them.
    
    Args:
        df_movies (pd.Dataframe): A dataframe with movie information.
        
    Returns:
        pd.Dataframe: A cleaned version of the movies dataframe.
    """

    bad_movie_ids = [26958, 168358, 6003, 32600, 64997]
    df_movies = df_movies[~df_movies['movieId'].isin(bad_movie_ids)]
    df_movies = df_movies[df_movies['genres'] != '(no genres listed)']

    df_movies['year'] = df_movies['title'].str.extract(r'\((\d{4})\)').astype(float)
    
    scaler = MinMaxScaler()
    df_movies['normalized_year'] = scaler.fit_transform(df_movies[['year']])
    
    return df_movies

def clean_ratings(df_ratings, movie_replacement_map=None):
    """
    Replaces user ratings of duplicate movies with the correct movieId
    
    Args: 
        df_ratings (pd.Dataframe): A dataframe with user rating information.
        movie_replacement_map (dictionary): Bad movieIds that are mapped to their replacement movieIds.
        
    Returns:
        pd.Dataframe: A cleaned version of the ratings dataframe.
    """

    if movie_replacement_map:
        df_ratings['movieId'] = df_ratings['movieId'].replace(movie_replacement_map)
    return df_ratings

def filter_less_active_data(df_ratings, df_movies, min_user_ratings=5, min_movie_ratings=10):
    """
    Removes less active users and movies from their respective dataframes.
    
    Args:
        df_ratings (pd.Dataframe): A dataframe with user rating information.
        df_movies (pd.Dataframe): A dataframe with movie information.
        min_user_ratings (int): Users with less ratings than this threshold will be removed.
        min_movie_ratings (int): Movies with less ratings than this threshold will be removed.
    
    Returns:
        pd.Dataframe: An updated ratings dataframe without the less active users.
        pd.Dataframe: An updated movies dataframe without the less active movies.
    """
    # Remove users with few ratings
    user_rating_counts = df_ratings['userId'].value_counts()
    users_to_keep = user_rating_counts[user_rating_counts >= min_user_ratings].index
    df_ratings = df_ratings[df_ratings['userId'].isin(users_to_keep)]
    
    # Remove movies with few ratings
    movie_rating_counts = df_ratings['movieId'].value_counts()
    movies_to_keep = movie_rating_counts[movie_rating_counts >= min_movie_ratings].index
    df_ratings = df_ratings[df_ratings['movieId'].isin(movies_to_keep)]
    df_movies = df_movies[df_movies['movieId'].isin(movies_to_keep)]
    
    return df_ratings, df_movies

def encode_genres(df_movies):
    """
    Extracts genres and multi-hot encodes them.
    
    Args: 
        df_movies (pd.Dataframe): A dataframe with movie information.
    
    Returns: 
        pd.Dataframe: A dataframe with multi-hot encoded genre values.
    """
    genre_split = df_movies['genres'].str.split('|')
    df_genres = (genre_split.explode()
                           .str.strip()
                           .pipe(pd.get_dummies)
                           .groupby(level=0)
                           .sum())
    df_genres = df_genres.set_index(df_movies['movieId'])
    return df_genres

def user_rating_train_test_split(df_ratings, test_size=0.2, random_state=24):
    """
    Performs a train_test_split on users' movie ratings.

    Args: 
        df_ratings (pd.Dataframe): A dataframe with user movie rating information.
        test_size (float): Percentage of values that are used for testing.
        random_state (int): Acts as a seed for the pseudo-random number generator.
    
    Returns:
        pd.dataFrame: Ratings for training the model.
        pd.dataFrame: Ratings for testing the model.
    """
    train_list, test_list = [], []

    for user_id, group in df_ratings.groupby('userId'):
        if len(group) >= 5:
            train, test = train_test_split(group, test_size=test_size, random_state=random_state)
            train_list.append(train)
            test_list.append(test)
        else:
            train_list.append(group)

    return pd.concat(train_list), pd.concat(test_list)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(movies_path='data/movies.csv', ratings_path='data/ratings.csv'):
    df_movies = pd.read_csv(movies_path)
    df_ratings = pd.read_csv(ratings_path)
    return df_movies, df_ratings
import numpy as np
import pandas as pd

movies_file_path = 'Dataset/movies.csv'
ratings_file_path = 'Dataset/ratings.csv'

df_movies = pd.read_csv(movies_file_path)
df_ratings = pd.read_csv(ratings_file_path)


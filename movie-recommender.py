import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import recommender.preprocessing as pp
import recommender.model as model

def recommend_movies(user_id, num_recommendations=10):
    # Get the similarity vector for this user
    user_similarities = df_user_movie_similarities.loc[user_id]

    # Exclude only training items
    train_movies = df_train_ratings[df_train_ratings['userId'] == user_id]['movieId'].values
    available_movies = user_similarities.drop(index=train_movies, errors='ignore')

    # Recommend top-k items
    top_k = available_movies.sort_values(ascending=False).head(num_recommendations)
    top_k = top_k.reset_index(name='score').rename(columns={'index': 'movieId'})

    return df_movies[['movieId', 'title']].merge(top_k, on='movieId')

def recommend_movies_all_users(num_recommendations=10):
    all_recommendations = {}
    for user_id in df_test_ratings['userId'].unique():
        recommendations = recommend_movies(user_id=user_id, num_recommendations=num_recommendations)
        all_recommendations[user_id] = recommendations['movieId'].tolist()
    return all_recommendations

def recall_at_k(recommended_items, relevant_items):
    if not relevant_items:
        return None
    hits = len(set(recommended_items) & set(relevant_items))
    return hits / len(relevant_items)

# -=| Data Loading |=-
df_movies, df_ratings = pp.load_data('data/movies.csv', 'data/ratings.csv')

# -=| Feature Engineering |=-
movie_replacement_map = {
    26958: 838,
    168358: 2851,
    6003: 144606,
    32600: 147002,
    64997: 34048
}

df_movies = pp.clean_movies(df_movies)
df_ratings = pp.clean_ratings(df_ratings, movie_replacement_map=movie_replacement_map)

# Remove less active users and movies
df_ratings, df_movies = pp.filter_less_active_data(df_ratings=df_ratings, df_movies=df_movies)

df_train_ratings, df_test_ratings = pp.user_rating_train_test_split(df_ratings=df_ratings)

df_genres = pp.encode_genres(df_movies=df_movies)

user_profiles = model.create_user_profiles(df_ratings=df_train_ratings, df_movies=df_movies, df_genres=df_genres)

# print("User profile for user 2:\n", user_profiles.loc[2])
# print("Sum of profile vector:", user_profiles.loc[2].sum())

similarity_matrix = cosine_similarity(user_profiles.values, df_genres.values)
df_user_movie_similarities = pd.DataFrame(similarity_matrix, index=user_profiles.index, columns=df_genres.index)

# Example Recommendation
recommendations = recommend_movies(user_id=5, num_recommendations=10)
print(recommendations)

# Evaluation
test_recommendations = recommend_movies_all_users(num_recommendations=10)
ground_truth = df_test_ratings.groupby('userId')['movieId'].apply(set).to_dict()

recalls = []

for user_id in test_recommendations:
    recs = test_recommendations[user_id]
    true_items = ground_truth.get(user_id, set())
    score = recall_at_k(recs, true_items)
    if score is not None:
        recalls.append(score)

average_recall = np.mean(recalls)
print(f"Average Recall@10: {average_recall:.4f}")
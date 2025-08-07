import recommender.preprocessing as pp
import recommender.model as model
import recommender.evaluation as eval

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

df_user_movie_similarities = model.compute_similarity_matrix(user_profiles=user_profiles, df_genres=df_genres)

# Example Recommendation
recommendations = model.recommend_movies(user_id=5, 
                                         df_user_movie_similarities=df_user_movie_similarities, 
                                         df_ratings=df_train_ratings,
                                         df_movies=df_movies,
                                         num_recommendations=10)
print(recommendations)

# Evaluation
test_recommendations = model.recommend_movies_all_users(user_movie_similarities=df_user_movie_similarities, df_ratings=df_train_ratings, df_movies=df_movies, num_recommendations=10)
ground_truth = df_test_ratings.groupby('userId')['movieId'].apply(set).to_dict()

average_recall = eval.average_recall_at_k(test_recommendations, ground_truth)
print(f"Average Recall@10: {average_recall:.4f}")
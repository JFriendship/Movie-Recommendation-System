from flask import Flask, render_template, request
import recommender.model as model
import recommender.preprocessing as pp
import pandas as pd

app = Flask(__name__)

df_movies, df_ratings = pp.load_data_from_db()

# Remove less active users and movies
df_ratings, df_movies = pp.filter_less_active_data(df_ratings=df_ratings, df_movies=df_movies)

df_train_ratings, df_test_ratings = pp.user_rating_train_test_split(df_ratings=df_ratings)

df_genres = pp.encode_genres(df_movies=df_movies)

user_profiles = model.create_user_profiles(df_ratings=df_train_ratings, df_movies=df_movies, df_genres=df_genres)

df_user_movie_similarities = model.compute_similarity_matrix(user_profiles=user_profiles, df_genres=df_genres)

@app.route('/', methods=['GET', 'POST'])
def recommend():
    recommendations = None
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        try:
            user_id = int(user_id)
        except (ValueError, TypeError):
            print("Invalid user_id submitted.")

        if user_id:
            recommendations = model.recommend_movies(user_id=user_id, 
                                                     df_user_movie_similarities=df_user_movie_similarities, 
                                                     df_ratings=df_train_ratings,
                                                     df_movies=df_movies,
                                                     num_recommendations=10)
            recommendations = recommendations['title'].tolist()
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
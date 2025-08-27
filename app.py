from flask import Flask, render_template, request, session, redirect, url_for, g
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import recommender.model as model
import recommender.preprocessing as pp
import pandas as pd
from dotenv import load_dotenv
from functools import wraps
import os, sys

app = Flask(__name__)

load_dotenv()
app.secret_key = os.environ["FLASK_SECRET_KEY"]

df_movies, df_ratings = pp.load_data_from_db()

# Remove less active users and movies
df_ratings, df_movies = pp.filter_less_active_data(df_ratings=df_ratings, df_movies=df_movies)
df_train_ratings, df_test_ratings = pp.user_rating_train_test_split(df_ratings=df_ratings)
df_genres = pp.encode_genres(df_movies=df_movies)
user_profiles = model.create_user_profiles(df_ratings=df_train_ratings, df_movies=df_movies, df_genres=df_genres)
df_user_movie_similarities = model.compute_similarity_matrix(user_profiles=user_profiles, df_genres=df_genres)

def get_db():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '..'))

    db_path = os.path.join(BASE_DIR, 'db', 'movies.db')

    if "db" not in g:
        g.db = sqlite3.connect(db_path)
        g.db.row_factory = sqlite3.Row
    return g.db

def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(**kwargs)
    return wrapped_view


# -=| Routes |=- #
@app.route('/')
def index():
    # if user is logged in, load the user's page

    #else, return welcome page template with link to login page
    return render_template('index.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        db = get_db()
        try:
            db.execute("""
                INSERT INTO users (username, password_hash, is_dataset_user)
                VALUES (?, ?, 0)
            """, (username, generate_password_hash(password)))
            db.commit()
        except sqlite3.IntegrityError:
            return "Username already taken. Try again.", 400

        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        db = get_db()
        user = db.execute("""
            SELECT * FROM users WHERE username = ? AND is_dataset_user = 0
        """, (username,)).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["user_id"]
            return redirect(url_for("recommendations"))
        else:
            return "Invalid login credentials", 401

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route('/recommendations', methods=['GET', 'POST'])
@login_required
def recommendations():
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
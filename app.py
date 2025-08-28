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

def get_db():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(BASE_DIR, '..'))

    db_path = os.path.join(BASE_DIR, 'db', 'movies.db')

    if "db" not in g:
        g.db = sqlite3.connect(db_path)
        g.db.row_factory = sqlite3.Row
    return g.db

def user_has_ratings(user_id):
    db = get_db()
    cur = db.execute("""
        SELECT EXISTS(
            SELECT 1 FROM ratings WHERE userId = ?
        )
    """, (user_id,))
    return cur.fetchone()[0] == 1

def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(**kwargs)
    return wrapped_view


# -=| Routes |=- #
@app.route('/')
def home():
    # if user is logged in, load the user's page

    #else, return welcome page template with link to login page
    return render_template('home.html')

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
        # Add  AND is_dataset_user = 0  after  username = ?  to ignore dataset accounts
        user = db.execute("""
            SELECT * FROM users WHERE username = ?
        """, (username,)).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["user_id"]
            session["username"] = username
            return redirect(url_for("recommendations"))
        else:
            return "Invalid login credentials", 401

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

@app.route('/recommendations')
@login_required
def recommendations():
    recommendations = None

    user_id = session["user_id"]
    if user_has_ratings(user_id) == 0:
        return render_template('recommendations.html', 
                               recommendations=recommendations, 
                               username=session["username"]
                            )

    db = get_db()

    df_movies = pd.read_sql_query("SELECT * FROM movies", db)
    df_ratings = pd.read_sql_query("SELECT * FROM ratings", db)
    # df_movies, df_ratings = pp.load_data_from_db()

    # Remove less active users and movies
    # df_ratings, df_movies = pp.filter_less_active_data(df_ratings=df_ratings, df_movies=df_movies)

    df_genres = pp.encode_genres(df_movies=df_movies)
    user_profiles = model.create_user_profiles(df_ratings=df_ratings, df_movies=df_movies, df_genres=df_genres)
    df_user_movie_similarities = model.compute_similarity_matrix(user_profiles=user_profiles, df_genres=df_genres)

    
    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        print("Something went wrong. Invalid user_id.")

    recommendations = model.recommend_movies(user_id=user_id, 
                                                df_user_movie_similarities=df_user_movie_similarities, 
                                                df_ratings=df_ratings,
                                                df_movies=df_movies,
                                                num_recommendations=10)
    recommendations = recommendations['title'].tolist()
    return render_template('recommendations.html', recommendations=recommendations, username=session["username"])

@app.route("/ratings")
@login_required
def ratings():
    return render_template('ratings.html')

if __name__ == '__main__':
    app.run(debug=True)
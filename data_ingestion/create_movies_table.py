
import pandas as pd
import sqlite3
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import recommender.preprocessing as pp

def create_movies_table():
    db_path = os.path.join(BASE_DIR, '..', 'db', 'movies.db')
    csv_path = os.path.join(BASE_DIR, '..', 'data', 'movies.csv')

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    try:
        print(f"Connecting to database at: {db_path}")
        conn = sqlite3.connect(db_path)

        print(f"Reading CSV from: {csv_path}")
        df_movies = pd.read_csv(csv_path)

        print("Cleaning movies data...")
        df_movies = pp.clean_movies(df_movies)

        print("Writing to SQLite DB...")
        df_movies.to_sql("movies", conn, if_exists="replace", index=False)

        conn.commit()
        print("Table 'movies' created in database.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_movies_table()
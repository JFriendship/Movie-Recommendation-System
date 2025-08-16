import pandas as pd
import sqlite3
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import recommender.preprocessing as pp

movie_replacement_map = {
    26958: 838,
    168358: 2851,
    6003: 144606,
    32600: 147002,
    64997: 34048
}

def create_ratings_table():
    db_path = os.path.join(BASE_DIR, '..', 'db', 'movies.db')
    csv_path = os.path.join(BASE_DIR, '..', 'data', 'ratings.csv')
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    try:
        print(f"Connecting to database at: {db_path}")
        conn = sqlite3.connect(db_path)

        print(f"Reading CSV from: {csv_path}")
        df_ratings = pd.read_csv(csv_path)

        print("Cleaning ratings data...")
        df_ratings = pp.clean_ratings(df_ratings, movie_replacement_map)

        print("Writing to SQLite DB...")
        df_ratings.to_sql("ratings", conn, if_exists="replace", index=False)

        conn.commit()
        print("Table 'ratings' created in database.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_ratings_table()
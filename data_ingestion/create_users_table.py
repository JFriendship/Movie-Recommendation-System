import sqlite3
from werkzeug.security import generate_password_hash
import pandas as pd
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

def create_users_table():
    db_path = os.path.join(BASE_DIR, '..', 'db', 'movies.db')
    csv_path = os.path.join(BASE_DIR, '..', 'data', 'ratings.csv')

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    try:
        print(f"Connecting to database at: {db_path}")
        conn = sqlite3.connect(db_path)

        cur = conn.cursor()

        # Drop if exists for testing
        # WARNING: IF THIS IS RUN AFTER REAL USER DATA IS ADDED, IT WILL BE DELETED
        cur.execute("DROP TABLE IF EXISTS users")   # THIS LINE WOULD BE THE CULPRIT

        cur.execute("""
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            is_dataset_user BOOLEAN NOT NULL DEFAULT 0
        )
        """)

        print(f"Reading CSV from: {csv_path}")
        df_ratings = pd.read_csv(csv_path)

        print("Inserting user data...")
        for uid in df_ratings["userId"].unique():
            uid = int(uid)
            username = f"user{uid}"
            dummy_pw = generate_password_hash(username)  # username and password are the same
            cur.execute("""
                INSERT INTO users (user_id, username, password_hash, is_dataset_user)
                VALUES (?, ?, ?, 1)
            """, (uid, username, dummy_pw))

        # Prime Autoincrement for real users starting at 1000
        print("Preparing autoincrement for real users...")
        cur.execute("""
            INSERT INTO users (user_id, username, password_hash, is_dataset_user)
            VALUES (?, ?, ?, 0)
        """, (999, "placeholder", generate_password_hash("placeholder")))
        cur.execute("DELETE FROM users WHERE user_id = 999")

        conn.commit()
        print("Table 'users' created in database.")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_users_table()
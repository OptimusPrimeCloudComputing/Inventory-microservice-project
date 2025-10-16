# db.py
import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DATABASE_USER")
DB_PASSWORD = os.getenv("DATABASE_PASSWORD")
DB_HOST = os.getenv("DATABASE_HOST")
DB_NAME = os.getenv("DATABASE_NAME")
DB_PORT = int(os.getenv("DATABASE_PORT"))


def get_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        # port=DB_PORT,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )


if __name__ == "__main__":
    try:
        conn = get_connection()
        print("✅ Connection successful!")
        print("Server info:", conn.get_server_info())

        with conn.cursor() as cursor:
            cursor.execute("SELECT DATABASE();")
            db_in_use = cursor.fetchone()
            print("Current database:", db_in_use)

    except Exception as e:
        print("❌ Connection failed:", e)
    finally:
        if 'conn' in locals() and conn.open:
            conn.close()
            print("Connection closed.")

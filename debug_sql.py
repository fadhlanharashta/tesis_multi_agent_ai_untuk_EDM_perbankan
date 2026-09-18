import sqlite3
from pathlib import Path

DATABASE_PATH = Path(__file__).parent / "database.db"

conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()

# cursor.execute("SELECT * FROM percobaan")
# cursor.execute("SELECT * FROM customers")
cursor.execute("SELECT * FROM banks")
# cursor.execute("SELECT * FROM nasabah")


rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
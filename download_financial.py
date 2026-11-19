
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "database.db"
# DB_PATH = Path(__file__).parent / "financial.db"

print("Database path:", DB_PATH)
print("File exists:", DB_PATH.exists())

if not DB_PATH.exists():
    print("ERROR: financial.db tidak ditemukan!")
    exit()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Ambil seluruh tabel
cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
      AND name NOT LIKE 'sqlite_%'
    ORDER BY name;
""")

tables = cursor.fetchall()

print("\n=== DAFTAR TABEL ===")

if not tables:
    print("Tidak ada tabel di database.")
else:
    for (table_name,) in tables:
        # Hitung jumlah record setiap tabel
        safe_name = table_name.replace('"', '""')
        cursor.execute(f'SELECT COUNT(*) FROM "{safe_name}"')
        count = cursor.fetchone()[0]

        print(f"{table_name}: {count:,} records")

conn.close()

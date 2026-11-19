
import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host="relational.fel.cvut.cz",
    port=3306,
    user="guest",
    password="ctu-relational",
    database="financial"
)

query = """
SELECT *
FROM account
LIMIT 10;
"""

df = pd.read_sql(query, conn)

print("Jumlah kolom:", len(df.columns))
print("\nNama kolom:")
print(df.columns.tolist())

print("\nHasil data:")
print(df.to_string(index=False))

conn.close()

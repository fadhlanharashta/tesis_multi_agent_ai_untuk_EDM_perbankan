from langchain_core.tools import tool
import sqlite3
from pathlib import Path
from difflib import get_close_matches

DATABASE_PATH = Path(__file__).parent.parent / "database.db"

@tool
def get_schema():
    """Get the tables and columns available in the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table';
    """)

    tables = cursor.fetchall()
    schema = {}

    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        schema[table_name] = [
            column[1]
            for column in columns
        ]
    conn.close()
    return str(schema)

@tool
def query_database(sql: str):
    """Execute a SQL query against the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        return rows
    except sqlite3.Error as e:
        return {
            "error": str(e),
            "sql": sql
        }
    finally:
        conn.close()

@tool
def create_table(table_name: str, columns: str):
    """Create a new table in the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    try:
        sql = f"CREATE TABLE {table_name} ({columns})"
        cursor.execute(sql)
        conn.commit()
        return f"Table '{table_name}' created successfully."

    except sqlite3.Error as e:
        return {
            "error": str(e)
        }
    finally:
        conn.close()

@tool
def delete_table(table_name: str):
    """Delete an existing table from the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    try:
        sql = f"DROP TABLE IF EXISTS {table_name}"
        cursor.execute(sql)
        conn.commit()
        return f"Table '{table_name}' deleted successfully."
    except sqlite3.Error as e:
        return {
            "error": str(e)
        }
    finally:
        conn.close()

@tool
def insert_data(table_name: str, columns: str, values: str):
    """
    Insert ONE row of data into an existing database table.

    IMPORTANT:
    - `columns` contains COLUMN NAMES only.
    - `values` contains VALUES only.
    - The order of values MUST match the order of columns.
    - Do NOT put parentheses around `values`.
    - String values MUST use single quotes.

    Example:

    table_name = "customers"
    columns = "ID, Name, address, income, debt"
    values = "'C00001', 'Raden Panggeas', 'Jl Laswi no 12', 5000000, 12000000"

    This produces:

    INSERT INTO customers (ID, Name, address, income, debt)
    VALUES ('C00001', 'Raden Panggeas', 'Jl Laswi no 12', 5000000, 12000000)
    """

    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    try:
        sql = f"""
        INSERT INTO {table_name} ({columns})
        VALUES ({values})
        """
        cursor.execute(sql)
        conn.commit()
        return f"Data inserted successfully into '{table_name}'."
    except sqlite3.Error as e:
        return {
            "error": str(e),
            "sql": sql
        }
    finally:
        conn.close()

@tool
def check_table(table_name: str):
    """Check whether a table exists in the database.
    If the table does not exist, return similar table names as suggestions.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table';
        """)
        tables = [
            row[0]
            for row in cursor.fetchall()
        ]
    finally:
        conn.close()
    if table_name in tables:
        return {
            "exists": True,
            "table": table_name,
            "suggestions": []
        }
    suggestions = get_close_matches(
        table_name,
        tables,
        n=3,
        cutoff=0.5
    )
    return {
        "exists": False,
        "table": table_name,
        "suggestions": suggestions
    }

@tool
def update_data(table_name: str, set_values: str, where_condition: str):
    """
    Update existing data in a database table.

    IMPORTANT:
    - table_name = table name only.
    - set_values = column assignments only.
    - where_condition = condition used to select which rows to update.
    - ALWAYS provide a WHERE condition.
    - Do NOT omit the WHERE condition.

    Example:

    table_name = "customers"
    set_values = "income = 6000000, debt = 10000000"
    where_condition = "ID = 'C00001'"

    This produces:

    UPDATE customers
    SET income = 6000000, debt = 10000000
    WHERE ID = 'C00001'
    """

    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    try:
        if not where_condition.strip():
            return {
                "error": "UPDATE requires a WHERE condition."
            }

        sql = f"""
        UPDATE {table_name}
        SET {set_values}
        WHERE {where_condition}
        """

        cursor.execute(sql)
        affected_rows = cursor.rowcount

        conn.commit()

        return {
            "success": True,
            "table": table_name,
            "affected_rows": affected_rows
        }

    except sqlite3.Error as e:
        conn.rollback()

        return {
            "success": False,
            "error": str(e),
            "sql": sql
        }

    finally:
        conn.close()
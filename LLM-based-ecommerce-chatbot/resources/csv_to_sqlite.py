"""
Loads a CSV file into a SQLite database table.
Usage:
    python csv_to_sqlite.py
Or import and call load_csv_to_sqlite() directly with your own paths.
"""

import sqlite3
import pandas as pd

# ---- Config ----
CSV_PATH = "resources/yoga.csv"   # change to your CSV file
DB_PATH = "products.db"
TABLE_NAME = "products"


def load_csv_to_sqlite(
    csv_path: str = CSV_PATH,
    db_path: str = DB_PATH,
    table_name: str = TABLE_NAME,
    if_exists: str = "replace",  # "replace", "append", or "fail"
):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {csv_path}")
    print("Columns:", df.columns.tolist())

    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        print(f"Wrote {len(df)} rows into '{table_name}' table in {db_path}")
    finally:
        conn.close()


def query_sample(db_path: str = DB_PATH, table_name: str = TABLE_NAME, limit: int = 5):
    """Quick sanity check — print a few rows back from the DB."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT {limit}", conn)
        print(df)
    finally:
        conn.close()


if __name__ == "__main__":
    load_csv_to_sqlite()
    query_sample()
"""python
# duckdb_loader.py
import duckdb
import tempfile
import os
import pandas as pd
from io import BytesIO

DB_FILE = "data.duckdb"
TABLE_NAME = "items"

def _connect(db_file: str = DB_FILE):
    return duckdb.connect(database=db_file, read_only=False)

def table_exists(table: str = TABLE_NAME, db_file: str = DB_FILE) -> bool:
    con = _connect(db_file)
    try:
        res = con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
            [table]
        ).fetchall()
        return len(res) > 0
    finally:
        con.close()

def ingest_from_path(path: str, table: str = TABLE_NAME, db_file: str = DB_FILE, overwrite: bool = True, read_options: dict | None = None):
    read_options = read_options or {}
    con = _connect(db_file)
    try:
        if overwrite:
            con.execute(f"DROP TABLE IF EXISTS {table}")
        opts = ",".join(f"{k}='{v}'" for k, v in read_options.items())
        opts_str = f", {opts}" if opts else ""
        # Use read_csv_auto to infer types; safe for most CSVs
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM read_csv_auto('{path}'{opts_str})")
    finally:
        con.close()

def ingest_from_bytes(contents: bytes, table: str = TABLE_NAME, db_file: str = DB_FILE, overwrite: bool = True, read_options: dict | None = None):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        tmp.write(contents)
        tmp.flush()
        tmp.close()
        ingest_from_path(tmp.name, table=table, db_file=db_file, overwrite=overwrite, read_options=read_options)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

def run_query(sql: str, params: list | None = None, db_file: str = DB_FILE) -> pd.DataFrame:
    con = _connect(db_file)
    try:
        if params:
            res = con.execute(sql, params).fetchdf()
        else:
            res = con.execute(sql).fetchdf()
        return res
    finally:
        con.close()

def get_sample(table: str = TABLE_NAME, n: int = 1000, db_file: str = DB_FILE) -> pd.DataFrame:
    if not table_exists(table, db_file=db_file):
        return pd.DataFrame()
    return run_query(f"SELECT * FROM {table} LIMIT {int(n)}", db_file=db_file)
"""
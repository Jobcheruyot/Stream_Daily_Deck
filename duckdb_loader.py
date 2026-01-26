import duckdb

def ingest_from_bytes(data: bytes):
    # Implement logic to ingest data from bytes
    pass

def ingest_from_path(file_path: str):
    # Implement logic to ingest data from a file at the given path
    pass

def run_query(query: str):
    # Implement logic to run a DuckDB query
    conn = duckbd.connect(':memory:')  # Connecting to an in-memory database for example
    return conn.execute(query).fetchall()

def get_sample(table_name: str, n: int):
    # Implement logic to get a sample of n rows from the given table
    query = f"SELECT * FROM {table_name} LIMIT {{n}};"
    return run_query(query)
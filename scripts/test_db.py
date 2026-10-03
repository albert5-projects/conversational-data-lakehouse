import duckdb
import time

print("Connecting to DuckDB...")

# We use the '*' symbol to tell DuckDB to read all the monthly files as one giant table
query = """
    SELECT 
        COUNT(*) AS total_rides,
        SUM(total_amount) AS total_revenue
    FROM 'nyc_taxi_data/*.parquet'
"""

print("Running query on millions of rows. Watch how fast this is...")
start_time = time.time()

# Execute the query and convert to a Pandas DataFrame to print it nicely
result = duckdb.query(query).df()

end_time = time.time()

print("\n--- BIG DATA RESULT ---")
print(result)
print(f"\nSuccess! Queried massive data in {end_time - start_time:.3f} seconds.")
print("And your laptop didn't crash! 🚀")
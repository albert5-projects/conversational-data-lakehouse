import duckdb
import time

Query = """
        DESC 'nyc_taxi_data/*.parquet'
"""

print("Results\n")
start = time.time()
result = duckdb.query(Query).df()
end  = time.time()

print(result)
print()
print("Finished in: ", end-start," sec")
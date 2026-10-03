"""One-off helper. Run:  python prepare_data.py [sample_rows]

1. Downloads taxi_zone_lookup.csv (tiny) into data/
2. Builds data/sample.parquet: a random sample of your full local dataset, small enough
   to commit to GitHub (<100 MB) so Streamlit Cloud can run without the 2 GB of raw files.
3. Writes data/sample_info.json so the app can tell users it's a sample.
"""
import glob
import json
import os
import sys
import urllib.request

import duckdb

import config

rows = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
os.makedirs("data", exist_ok=True)

if not os.path.exists(config.ZONES_PATH):
    print("Downloading taxi zone lookup...")
    urllib.request.urlretrieve(config.ZONES_URL, config.ZONES_PATH)

files = glob.glob(config.LOCAL_DATA_GLOB)
if not files:
    sys.exit(f"No parquet files found at {config.LOCAL_DATA_GLOB}. Run download_data_windows.py first.")

where = (f"tpep_pickup_datetime >= TIMESTAMP '{config.DATA_MIN_YEAR}-01-01' "
         f"AND tpep_pickup_datetime < TIMESTAMP '{config.DATA_MAX_YEAR + 1}-01-01'")
src = f"read_parquet('{config.LOCAL_DATA_GLOB}', union_by_name=true)"

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
total = con.execute(f"SELECT COUNT(*) FROM {src} WHERE {where}").fetchone()[0]
rows = min(rows, total)
print(f"Sampling {rows:,} of {total:,} rows...")
con.execute(f"""
    COPY (SELECT * FROM {src} WHERE {where} USING SAMPLE {rows} ROWS)
    TO '{config.SAMPLE_PATH}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
with open(config.SAMPLE_INFO_PATH, "w") as f:
    json.dump({"total_rows": total, "sample_rows": rows}, f)

size_mb = os.path.getsize(config.SAMPLE_PATH) / 1e6
print(f"Wrote {config.SAMPLE_PATH} ({size_mb:.1f} MB)")
if size_mb > 95:
    print("WARNING: over GitHub's 100 MB limit. Re-run with fewer rows, e.g. python prepare_data.py 2000000")
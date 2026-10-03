"""Central settings. Every value can be overridden with an environment variable
(or a Streamlit secret exposed as an env var), so nothing is hard-coded in agent.py."""
import os


def _env(name, default):
    return os.getenv(name, default)


# ---- LLM -------------------------------------------------------------------
MODEL_NAME = _env("MODEL_NAME", "openai/gpt-oss-120b")

# ---- Data source -----------------------------------------------------------
# "auto"   -> local files if present, else the committed sample, else remote
# "local"  -> nyc_taxi_data/*.parquet on disk (full dataset)
# "sample" -> data/sample.parquet (small file you can commit to GitHub)
# "remote" -> read the public NYC TLC files over HTTPS (no local data needed)
DATA_MODE = _env("DATA_MODE", "auto")
LOCAL_DATA_GLOB = _env("LOCAL_DATA_GLOB", "nyc_taxi_data/*.parquet")
SAMPLE_PATH = _env("SAMPLE_PATH", "data/sample.parquet")
SAMPLE_INFO_PATH = _env("SAMPLE_INFO_PATH", "data/sample_info.json")
ZONES_PATH = _env("ZONES_PATH", "data/taxi_zone_lookup.csv")
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
REMOTE_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
REMOTE_YEARS = (2023, 2024, 2025)

# Pickups outside this range are junk (the raw files contain dates like 2001 or 2098).
# The `trips` view filters them out so the LLM never has to.
DATA_MIN_YEAR = int(_env("DATA_MIN_YEAR", "2023"))
DATA_MAX_YEAR = int(_env("DATA_MAX_YEAR", "2025"))

# ---- Safety / resource limits ---------------------------------------------
ROW_LIMIT = int(_env("ROW_LIMIT", "10000"))
QUERY_TIMEOUT_SEC = int(_env("QUERY_TIMEOUT_SEC", "60"))
MAX_RETRIES = int(_env("MAX_RETRIES", "3"))
MEMORY_LIMIT = _env("MEMORY_LIMIT", "1500MB")
THREADS = int(_env("THREADS", "4"))

# ---- Chat ------------------------------------------------------------------
HISTORY_TURNS = int(_env("HISTORY_TURNS", "3"))
MAX_QUESTIONS_PER_SESSION = int(_env("MAX_QUESTIONS_PER_SESSION", "25"))
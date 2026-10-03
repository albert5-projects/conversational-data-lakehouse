"""Self-correcting text-to-SQL agent on DuckDB.

Pipeline for every question:
    LLM -> {sql, chart hint}  ->  validate (sqlglot)  ->  execute (locked-down DuckDB,
    row limit, timeout)  ->  on SQL error: feed error back to the LLM and retry
    ->  plain-English summary
"""
import glob
import json
import logging
import os
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd
import sqlglot
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sqlglot import exp

import config

load_dotenv()
log = logging.getLogger("agent")


# =============================================================================
# 1. Data dictionary (hand-written descriptions, merged with the real schema)
# =============================================================================
COLUMN_DOCS = {
    "vendorid": "1 = Creative Mobile Technologies, 2 = VeriFone Inc. (6, 7 or others = 'Other/New Vendors').",
    "tpep_pickup_datetime": "Date and time when the meter was engaged (start of trip).",
    "tpep_dropoff_datetime": "Date and time when the meter was disengaged (end of trip).",
    "passenger_count": "Number of passengers (entered by the driver).",
    "trip_distance": "Trip distance in miles.",
    "ratecodeid": "1=Standard, 2=JFK, 3=Newark, 4=Nassau/Westchester, 5=Negotiated fare, 6=Group ride (99 = anomaly).",
    "store_and_fwd_flag": "'Y' = store and forward trip, 'N' = not.",
    "pulocationid": "TLC Taxi Zone ID where the trip started. Join zones.LocationID for names.",
    "dolocationid": "TLC Taxi Zone ID where the trip ended. Join zones.LocationID for names.",
    "payment_type": "1=Credit card, 2=Cash, 3=No charge, 4=Dispute, 5=Unknown, 6=Voided trip (0 = Unrecorded/Error).",
    "fare_amount": "Base time-and-distance fare from the meter.",
    "extra": "Miscellaneous extras and surcharges (rush hour, overnight).",
    "mta_tax": "$0.50 MTA tax.",
    "tip_amount": "Tip. ONLY recorded for credit-card payments; cash tips are NOT recorded.",
    "tolls_amount": "Total tolls paid in the trip.",
    "improvement_surcharge": "$0.30 improvement surcharge on hailed trips.",
    "total_amount": "Total charged to the passenger (fare + taxes + surcharges + credit-card tip).",
    "congestion_surcharge": "Surcharge for trips in Manhattan's congestion zone.",
    "airport_fee": "$1.25 fee for pickups at LaGuardia or JFK.",
    "cbd_congestion_fee": "Congestion pricing fee (Manhattan central business district), 2025 onwards.",
}

ZONES_DOC = """Table: zones  (taxi zone lookup)
- LocationID (BIGINT): matches trips.PULocationID / trips.DOLocationID
- Borough (VARCHAR): Manhattan, Brooklyn, Queens, Bronx, Staten Island, EWR, Unknown
- Zone (VARCHAR): neighbourhood name, e.g. 'JFK Airport', 'Times Sq/Theatre District'
- service_zone (VARCHAR): Yellow Zone, Boro Zone, Airports, EWR"""

BUSINESS_RULES = f"""
BUSINESS RULES (always follow):
- "Revenue" means SUM(total_amount). Exclude voided trips and refunds: payment_type <> 6 AND total_amount > 0.
- The `trips` view already contains only pickups from {config.DATA_MIN_YEAR} to {config.DATA_MAX_YEAR}.
  The most recent year in the data is {config.DATA_MAX_YEAR}. "This year"/"latest year" = {config.DATA_MAX_YEAR},
  "last year" = {config.DATA_MAX_YEAR - 1}. Never use CURRENT_DATE or today's date.
- For averages of fares/distances, ignore bad rows: trip_distance > 0 AND fare_amount > 0.
- Tips exist only for credit-card trips (payment_type = 1). For tip analysis filter on payment_type = 1.
- For readable place names, JOIN zones ON trips.PULocationID = zones.LocationID (or DOLocationID).
- Give columns clear aliases (e.g. total_revenue, trip_count). When asked for a "contribution" or
  "share", return a percentage column rounded to 2 decimals.
- Put the category/label column FIRST and the main numeric column SECOND.
- For a time series, return a date column (use DATE_TRUNC) first, sorted ascending.
- Use only the tables `trips` and `zones`. Read-only SELECT queries only.
"""

FEW_SHOT = """
EXAMPLES
Q: What was the total revenue by payment type and each type's share?
SQL: SELECT payment_type, SUM(total_amount) AS total_revenue,
            ROUND(100.0 * SUM(total_amount) / SUM(SUM(total_amount)) OVER (), 2) AS pct_of_revenue
     FROM trips WHERE payment_type <> 6 AND total_amount > 0
     GROUP BY payment_type ORDER BY total_revenue DESC

Q: Top 10 pickup neighbourhoods by number of trips
SQL: SELECT z.Zone AS pickup_zone, COUNT(*) AS trip_count
     FROM trips t JOIN zones z ON t.PULocationID = z.LocationID
     GROUP BY z.Zone ORDER BY trip_count DESC LIMIT 10

Q: Monthly revenue trend
SQL: SELECT DATE_TRUNC('month', tpep_pickup_datetime)::DATE AS month, SUM(total_amount) AS total_revenue
     FROM trips WHERE payment_type <> 6 AND total_amount > 0
     GROUP BY month ORDER BY month
"""


# =============================================================================
# 2. SQL safety: validate before executing
# =============================================================================
ALLOWED_TABLES = {"trips", "zones"}
BLOCKED_FUNCTIONS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "read_text", "read_blob", "parquet_scan", "csv_scan", "glob", "getenv",
    "current_setting", "query", "query_table", "duckdb_settings", "write_csv",
    "sniff_csv", "parquet_metadata", "parquet_schema", "read_ndjson",
}


def validate_sql(sql: str) -> None:
    """Raise ValueError unless `sql` is exactly one read-only SELECT on trips/zones."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.SqlglotError as e:
        raise ValueError(f"SQL could not be parsed: {e}")

    if len(statements) != 1:
        raise ValueError("Exactly one SQL statement is allowed.")
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union)):
        raise ValueError("Only SELECT statements are allowed.")

    cte_names = {c.alias.lower() for c in stmt.find_all(exp.CTE)}
    for table in stmt.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise ValueError("Table functions and file paths are not allowed. Use FROM trips or FROM zones.")
        if table.args.get("db") or table.args.get("catalog"):
            raise ValueError("Schema-qualified tables are not allowed. Use FROM trips or FROM zones.")
        name = table.name.lower()
        if name not in ALLOWED_TABLES and name not in cte_names:
            raise ValueError(f"Unknown table '{table.name}'. Only 'trips' and 'zones' exist.")

    for func in stmt.find_all(exp.Func):
        name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if name in BLOCKED_FUNCTIONS:
            raise ValueError(f"Function '{name}' is not allowed.")


# =============================================================================
# 3. DuckDB connection: views + resource limits + lockdown
# =============================================================================
def _lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _url_exists(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=8):
            return True
    except Exception:
        return False


def resolve_data_source():
    """Decide where trip data comes from. Returns (kind, source) where source is a
    glob/path string or a list of URLs."""
    mode = config.DATA_MODE.lower()
    local_files = glob.glob(config.LOCAL_DATA_GLOB)
    if mode == "auto":
        if local_files:
            mode = "local"
        elif Path(config.SAMPLE_PATH).exists():
            mode = "sample"
        else:
            mode = "remote"

    if mode == "local":
        if not local_files:
            raise FileNotFoundError(f"No parquet files match {config.LOCAL_DATA_GLOB}")
        return "local", config.LOCAL_DATA_GLOB
    if mode == "sample":
        if not Path(config.SAMPLE_PATH).exists():
            raise FileNotFoundError(f"{config.SAMPLE_PATH} not found. Run prepare_data.py first.")
        return "sample", config.SAMPLE_PATH
    if mode == "remote":
        urls = [
            f"{config.REMOTE_BASE_URL}/yellow_tripdata_{y}-{m:02d}.parquet"
            for y in config.REMOTE_YEARS for m in range(1, 13)
        ]
        urls = [u for u in urls if _url_exists(u)]  # skip months not published yet
        if not urls:
            raise RuntimeError("Could not reach any remote NYC TLC files.")
        return "remote", urls
    raise ValueError(f"Unknown DATA_MODE '{config.DATA_MODE}'")


def _build_connection(kind, source, lock: bool):
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit={_lit(config.MEMORY_LIMIT)}")
    con.execute(f"SET threads={int(config.THREADS)}")

    zones_local = Path(config.ZONES_PATH).exists()
    zones_src = config.ZONES_PATH if zones_local else config.ZONES_URL
    if kind == "remote" or not zones_local:
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")

    if isinstance(source, list):
        files_expr = "[" + ", ".join(_lit(u) for u in source) + "]"
    else:
        files_expr = _lit(source)

    # union_by_name: files from different years have slightly different columns
    # (e.g. Airport_fee vs airport_fee, cbd_congestion_fee added in 2025).
    con.execute(f"""
        CREATE VIEW trips AS
        SELECT * FROM read_parquet({files_expr}, union_by_name=true)
        WHERE tpep_pickup_datetime >= TIMESTAMP '{config.DATA_MIN_YEAR}-01-01'
          AND tpep_pickup_datetime <  TIMESTAMP '{config.DATA_MAX_YEAR + 1}-01-01'
    """)
    con.execute(f"CREATE VIEW zones AS SELECT * FROM read_csv({_lit(zones_src)}, header=true)")

    # smoke test BEFORE locking (a locked connection can't be reconfigured)
    con.execute("SELECT 1 FROM trips LIMIT 1").fetchall()
    con.execute("SELECT 1 FROM zones LIMIT 1").fetchall()

    if lock:
        allowed = set()
        for p in (source if isinstance(source, list) else [source]) + [zones_src]:
            if p.startswith("http"):
                allowed.add(p.rsplit("/", 1)[0] + "/")
            else:
                allowed.add(Path(p.split("*")[0]).resolve().parent.as_posix() + "/"
                            if "*" in p else Path(p).resolve().parent.as_posix() + "/")
        con.execute("SET allowed_directories=[" + ", ".join(_lit(a) for a in sorted(allowed)) + "]")
        con.execute("SET enable_external_access=false")
        con.execute("SET lock_configuration=true")
        con.execute("SELECT 1 FROM trips LIMIT 1").fetchall()  # views must still work
        con.execute("SELECT 1 FROM zones LIMIT 1").fetchall()
    return con


def build_connection(kind, source):
    """Try the locked-down connection first; fall back to unlocked (SQL validation
    still protects us) if this DuckDB version can't combine lockdown with views."""
    try:
        return _build_connection(kind, source, lock=True), True
    except Exception as e:
        log.warning("DuckDB lockdown failed (%s); falling back to validation-only protection.", e)
        return _build_connection(kind, source, lock=False), False


def build_schema_text(con) -> str:
    """Real column names/types from DESCRIBE, merged with the hand-written descriptions."""
    lines = ["Table: trips  (NYC yellow taxi trips)"]
    for name, dtype, *_ in con.execute("DESCRIBE trips").fetchall():
        lines.append(f"- {name} ({dtype}): {COLUMN_DOCS.get(name.lower(), '(no description)')}")
    return "\n".join(lines) + "\n\n" + ZONES_DOC


# =============================================================================
# 4. LLM plumbing
# =============================================================================
class SQLPlan(BaseModel):
    sql: str = Field(description="One DuckDB SELECT statement. No markdown, no comments, no trailing explanation.")
    chart_type: Literal["auto", "metric", "bar", "line", "pie", "table"] = Field(
        default="auto", description="Best way to display the result.")
    x_column: str = Field(default="", description="Result column for the x-axis / labels (must be in the SELECT).")
    y_column: str = Field(default="", description="Result column for the y-axis / values (must be in the SELECT).")


PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert data analyst writing DuckDB SQL for a business user.\n\n"
     "DATABASE SCHEMA:\n{schema}\n\n{rules}\n{examples}\n"
     "Return the SQL query plus a chart suggestion (chart_type, x_column, y_column). "
     "Query only the views `trips` and `zones`, e.g. FROM trips. Never reference file paths."),
    ("human",
     "{history}Question: {question}\n{error_block}"),
])


def _load_secrets():
    """Make GROQ_API_KEY available from .env, the environment, or Streamlit secrets."""
    if os.getenv("GROQ_API_KEY"):
        return
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass


def build_llm():
    from langchain_groq import ChatGroq
    _load_secrets()
    return ChatGroq(model=config.MODEL_NAME, temperature=0, timeout=40, max_retries=2)


def _parse_plain_text(text: str) -> SQLPlan:
    """Fallback if structured output isn't available: accept JSON or raw SQL."""
    cleaned = text.replace("```json", "").replace("```sql", "").replace("```", "").strip()
    try:
        return SQLPlan(**json.loads(cleaned))
    except Exception:
        return SQLPlan(sql=cleaned)


class QueryTimeout(Exception):
    pass


# =============================================================================
# 5. The agent
# =============================================================================
@dataclass
class AgentResult:
    question: str
    sql: str = ""
    df: pd.DataFrame | None = None
    truncated: bool = False
    attempts: int = 0
    plan: SQLPlan | None = None
    summary: str = ""
    error: str = ""
    attempt_log: list = field(default_factory=list)


class Agent:
    def __init__(self, con, schema_text, llm, source_kind="local", locked=True):
        self.con = con
        self.schema_text = schema_text
        self.llm = llm
        self.source_kind = source_kind
        self.locked = locked
        self._chain = None

    # ---- LLM calls ---------------------------------------------------------
    def _generate(self, question, history, error, previous_sql) -> SQLPlan:
        hist = ""
        if history:
            turns = history[-config.HISTORY_TURNS:]
            hist = "Previous conversation (use it for follow-up questions):\n" + "\n".join(
                f"Q: {h['question']}\nSQL: {h['sql']}" for h in turns) + "\n\n"
        error_block = ""
        if error:
            error_block = (f"\nYour previous SQL:\n{previous_sql}\nfailed with:\n{error}\n"
                           "Fix it and return a corrected query.")
        variables = dict(schema=self.schema_text, rules=BUSINESS_RULES, examples=FEW_SHOT,
                         history=hist, question=question, error_block=error_block)
        try:
            if self._chain is None:
                self._chain = PROMPT | self.llm.with_structured_output(SQLPlan)
            return self._chain.invoke(variables)
        except Exception as e:
            # structured output unsupported/malformed -> plain text fallback.
            # If the real problem was the API itself, this second call raises and surfaces it.
            log.warning("Structured output failed (%s); using plain-text fallback.", e)
            self._chain = None
            msg = (PROMPT | self.llm).invoke(variables)
            return _parse_plain_text(msg.content)

    def summarize(self, question, df, truncated) -> str:
        sample = df.head(15).to_csv(index=False)
        note = f" (showing only the first {len(df)} rows)" if truncated else ""
        prompt = (
            "You are a business analyst. In 2 short sentences, answer the question using ONLY "
            "the numbers in this result table. Do not invent figures. Format money with $ and "
            f"thousands separators.\n\nQuestion: {question}\nResult{note}:\n{sample}")
        try:
            return str(self.llm.invoke(prompt).content).strip()
        except Exception as e:
            log.warning("Summary failed: %s", e)
            return ""

    # ---- SQL execution -----------------------------------------------------
    def execute_sql(self, sql):
        """Validate, wrap in a row limit, run with a timeout. Returns (df, truncated)."""
        validate_sql(sql)
        inner = sql.strip().rstrip(";")
        wrapped = f"SELECT * FROM ({inner}) AS q LIMIT {config.ROW_LIMIT + 1}"
        cur = self.con.cursor()
        timer = threading.Timer(config.QUERY_TIMEOUT_SEC, cur.interrupt)
        timer.start()
        try:
            df = cur.execute(wrapped).df()
        except duckdb.InterruptException:
            raise QueryTimeout(f"Query ran longer than {config.QUERY_TIMEOUT_SEC}s and was stopped.")
        finally:
            timer.cancel()
            cur.close()
        truncated = len(df) > config.ROW_LIMIT
        return df.head(config.ROW_LIMIT), truncated

    # ---- main loop ---------------------------------------------------------
    def ask(self, question, history=None) -> AgentResult:
        result = AgentResult(question=question)
        error, previous_sql = "", ""
        for attempt in range(1, config.MAX_RETRIES + 1):
            result.attempts = attempt
            plan = self._generate(question, history, error, previous_sql)
            result.plan, result.sql = plan, plan.sql
            try:
                df, truncated = self.execute_sql(plan.sql)
            except QueryTimeout as e:
                result.error = str(e) + " Try a narrower question (e.g. one year or one borough)."
                return result
            except (duckdb.Error, ValueError) as e:
                # Only SQL/validation problems are fed back to the LLM.
                # API/network errors are NOT caught here, so they surface once instead of
                # being "fixed" three times.
                error, previous_sql = str(e), plan.sql
                result.attempt_log.append({"attempt": attempt, "sql": plan.sql, "error": error})
                log.info("Attempt %d failed: %s", attempt, error)
                continue
            result.df, result.truncated, result.error = df, truncated, ""
            if not df.empty:
                result.summary = self.summarize(question, df, truncated)
            return result
        result.error = f"Could not produce a working query after {config.MAX_RETRIES} attempts. Last error: {error}"
        return result


def build_agent() -> Agent:
    kind, source = resolve_data_source()
    con, locked = build_connection(kind, source)
    return Agent(con, build_schema_text(con), build_llm(), kind, locked)


def load_sample_info():
    try:
        return json.loads(Path(config.SAMPLE_INFO_PATH).read_text())
    except Exception:
        return None


# --- CLI test: python agent.py --------------------------------------------------
if __name__ == "__main__":
    agent = build_agent()
    print(f"Data source: {agent.source_kind} | lockdown active: {agent.locked}")
    res = agent.ask("What was the total revenue generated by each payment type and the contribution of each?")
    print("\nSQL:\n", res.sql)
    print("\nError:", res.error or "none")
    if res.df is not None:
        print(res.df)
        print("\nSummary:", res.summary)
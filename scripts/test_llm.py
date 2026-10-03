import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 1. Load the API key
load_dotenv()

# 2. Wake up the LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0 
)

# 3. THE PERFECT SCHEMA (Extracted directly from the Parquet files)
database_schema = """
Table Name: 'nyc_taxi_data/*.parquet'
Columns:
- VendorID (BIGINT)
- tpep_pickup_datetime (TIMESTAMP)
- tpep_dropoff_datetime (TIMESTAMP)
- passenger_count (DOUBLE)
- trip_distance (DOUBLE)
- RatecodeID (DOUBLE)
- store_and_fwd_flag (VARCHAR)
- PULocationID (BIGINT)
- DOLocationID (BIGINT)
- payment_type (BIGINT)
- fare_amount (DOUBLE)
- extra (DOUBLE)
- mta_tax (DOUBLE)
- tip_amount (DOUBLE)
- tolls_amount (DOUBLE)
- improvement_surcharge (DOUBLE)
- total_amount (DOUBLE)
- congestion_surcharge (DOUBLE)
- airport_fee (DOUBLE)
"""

# 4. A slightly harder CEO Question
user_question = "What was the average trip_distance, the maximum passenger_count on a single ride, and the total tolls_amount collected?"

# 5. The Prompt
prompt = f"""
You are an expert Data Engineer.
Here is the exact schema for our DuckDB database:
{database_schema}

The CEO asks: "{user_question}"

Write a valid DuckDB SQL query to answer this. 
RULES:
1. Reply ONLY with the raw SQL code. 
2. Do not add markdown formatting like ```sql.
3. Do not explain your answer. Just give me the exact SELECT statement.
"""

print("Sending the CEO's question to the AI Brain...")

# 6. Ask the AI and print the answer
response = llm.invoke(prompt)

print("\n--- AI GENERATED SQL ---")
print(response.content)
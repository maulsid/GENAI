"""
Two-stage pipeline:
1. sql_prompt   -> Groq generates a SQL query (wrapped in <SQL></SQL> tags) from a natural-language question.
2. comprehension_prompt -> Groq turns the query results back into a natural-language answer.
"""

import os
import re
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from config import GROQ_MODEL

load_dotenv()

# ---- Config ----
DB_PATH = "products.db"
TABLE_NAME = "products"

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---- Prompt 1: SQL generation ----
sql_prompt = """You are an expert in understanding the database schema and generating SQL queries for a natural language question asked
pertaining to the data you have. The schema is provided in the schema tags.
<schema>
table: products

fields:
name - string (name of the product)
main_category - string (top-level category, e.g. "sports")
sub_category - string (more specific category, e.g. "yoga")
image - string (image URL)
link - string (hyperlink to product)
ratings - float (average rating of the product. Range 0-5, 5 is the highest.)
no_of_ratings - string (total number of ratings for the product — may contain commas, e.g. "1,234")
discount_price - string (current/discounted price — may include currency symbol and commas)
actual_price - string (original price before discount — may include currency symbol and commas)

</schema>
Notes:
- discount_price, actual_price, and no_of_ratings are stored as TEXT, not numbers — they may contain
  currency symbols (₹) and commas (e.g. "₹1,299"). If you need to filter or sort by these numerically,
  strip non-numeric characters using REPLACE() before casting, e.g.:
  CAST(REPLACE(REPLACE(discount_price, '₹', ''), ',', '') AS REAL)
- MANDATORY RULE: whenever the question mentions a product type or keyword (e.g. "yoga mat", "running shoes")
  Never filter on sub_category alone. This is required even if sub_category looks like an exact match —
  many relevant products only mention the keyword in `name`, not in `sub_category`.
 - MANDATORY RULE: NEVER build multiple OR'd full-phrase variants to hedge against different wordings
  (e.g. do NOT do '%running shoes for women%' OR '%women running shoes%' OR '%women's running shoes%').
  Product names rarely contain exact multi-word phrases verbatim. Instead, break the request into its
  individual significant keywords and require each one to appear somewhere (name OR sub_category),
  combined with AND across keywords. Ignore filler words like "for", "options", "some", "a".
  Example — question "running shoes for women" MUST produce a WHERE clause shaped like:
  WHERE (name LIKE '%running%')
    AND (name LIKE '%shoes%')
    AND (name LIKE '%women%')
- Make sure any text search (product type, category, brand-like terms) uses %LIKE% and is case-insensitive.
  Never use "ILIKE".
Create a single SQL query for the question provided.
The query should have all the fields in SELECT clause (i.e. SELECT *)

Just the SQL query is needed, nothing more. Always provide the SQL in between the <SQL></SQL> tags."""

# ---- Prompt 2: natural language comprehension of results ----
comprehension_prompt = """You are an expert in understanding the context of the question and replying based on the data pertaining to the question provided. You will be provided with Question: and Data:. The data will be in the form of an array or a dataframe or dict. Reply based on only the data provided as Data for answering the question asked as Question. Do not write anything like 'Based on the data' or any other technical words. Just a plain simple natural language response.
The Data would always be in context to the question asked. For example is the question is “What is the average rating?” and data is “4.3”, then answer should be “The average rating for the product is 4.3”. So make sure the response is curated with the question and data. Make sure to note the column names to have some context, if needed, for your response.
There can also be cases where you are given an entire dataframe in the Data: field. Always remember that the data field contains the answer of the question asked. All you need to do is to always reply in the following format when asked about a product:
Product name, discount price, actual price, and rating, and then product link. Take care that all the products are listed in list format, one line after the other. Not as a paragraph.
For example:
1. Campus Women Running Shoes: ₹1104 (was ₹1699), Rating: 4.4 <link>
2. Campus Women Running Shoes: ₹1104 (was ₹1699), Rating: 4.4 <link>
3. Campus Women Running Shoes: ₹1104 (was ₹1699), Rating: 4.4 <link>"""


def call_groq(system_prompt: str, user_content: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def generate_sql(user_query: str) -> str:
    raw = call_groq(sql_prompt, user_query)

    match = re.search(r"<SQL>(.*?)</SQL>", raw, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not find <SQL> tags in model output:\n{raw}")

    return match.group(1).strip()


def is_safe_select(sql: str) -> bool:
    """Basic guard — only allow single SELECT statements."""
    normalized = sql.strip().rstrip(";").lower()
    if not normalized.startswith("select"):
        return False
    forbidden = ["insert", "update", "delete", "drop", "alter", "attach", "pragma", ";"]
    return not any(word in normalized for word in forbidden)


def run_sql(sql: str, db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def generate_comprehension(user_query: str, data: pd.DataFrame) -> str:
    data_str = data.to_dict(orient="records")
    user_content = f"Question: {user_query}\nData: {data_str}"
    return call_groq(comprehension_prompt, user_content)


def sql_chain(user_query: str) -> str:
    """Single entry point: NL question -> SQL -> execute -> NL answer."""
    sql = generate_sql(user_query)
    print(f"Generated SQL:\n{sql}\n")

    if not is_safe_select(sql):
        raise ValueError(f"Refusing to run unsafe/non-SELECT query: {sql}")

    data = run_sql(sql)

    if data.empty:
        return "I couldn't find any products matching that."

    return generate_comprehension(user_query, data)


if __name__ == "__main__":
    query = input("Ask a question about the products: ")
    answer = sql_chain(query)
    print(answer)
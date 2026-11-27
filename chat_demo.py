"""Midstone 4: a three-turn conversation with streaming and one tool call."""
import os
from pathlib import Path

from sqlalchemy import create_engine, text

from src.llm_client import Conversation, LLMClient, RunLog, tool_spec

LOG = "run_log.jsonl"
_default_db = Path(__file__).resolve().parent.parent / "Midstone 2" / "kaveri.db"
_database_url = os.getenv("DATABASE_URL", f"sqlite:///{_default_db.as_posix()}"
                          if _default_db.exists() else "sqlite:///kaveri.db")
engine = create_engine(_database_url)


def get_revenue(country: str, year: int) -> float:
    """Revenue from one country's customers in one year: ONE read-only SELECT."""
    sql = text("""SELECT SUM(orders.total_amount)
                  FROM orders
                  JOIN customers ON orders.customer_id = customers.customer_id
                  WHERE customers.country = :country
                    AND orders.order_date LIKE :year_pattern""")
    with engine.connect() as con:
        return float(con.execute(sql, {"country": country,
                                       "year_pattern": f"{int(year):04d}-%"}).scalar() or 0)


TOOLS = [tool_spec(
    "get_revenue",
    "Revenue in USD from Kaveri Exports customers in one country in one year.",
    {"country": {"type": "string", "description": "Country name, e.g. Germany"},
     "year": {"type": "integer", "description": "Four-digit year, e.g. 2024"}},
    ["country", "year"])]

TURNS = [
    "Hi! In one sentence, what can you help me with?",
    "What was our revenue from Germany in 2024? Use the tool.",
    "What was the revenue figure you just gave me, and was it greater than $80,000?",
]

if __name__ == "__main__":
    client = LLMClient(log=RunLog(LOG))
    chat = Conversation(client, system="You are the Kaveri Exports sales assistant. Be brief. "
                        "For revenue questions always call get_revenue; never guess numbers.",
                        tools=TOOLS, functions={"get_revenue": get_revenue})
    for question in TURNS:
        print(f"\nYOU: {question}\nAI:  ", end="", flush=True)
        chat.ask(question, on_token=lambda t: print(t, end="", flush=True))
        print()
    print(f"\nwrote {LOG}")

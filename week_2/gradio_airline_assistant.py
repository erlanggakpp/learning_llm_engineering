import os
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import json


# Load environment variables if available
load_dotenv()

# Database configuration
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticket_prices.db")

# Model and client configuration (referencing week_2/gradio_interface.py)
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Initialize client pointing to local Ollama server
openai = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

# ============================================================================
# SQLite Database Management Functions
# ============================================================================

def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def create_ticket_prices_table(db_path: str = DB_PATH) -> None:
    """
    Creates the 'ticket_prices' table with columns (city, price) if it doesn't already exist.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ticket_prices (
                city TEXT PRIMARY KEY,
                price REAL NOT NULL
            )
            """
        )
        conn.commit()


def insert_ticket_price_to_db(city: str, price: float, db_path: str = DB_PATH) -> None:
    """
    Inserts a new record into the 'ticket_prices' table.
    Updates the price if the city already exists in the table.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO ticket_prices (city, price)
            VALUES (?, ?)
            ON CONFLICT(city) DO UPDATE SET price = excluded.price
            """,
            (city.strip(), float(price)),
        )
        conn.commit()


def insert_ticket_prices(records: list[tuple[str, float]], db_path: str = DB_PATH) -> None:
    """
    Inserts multiple records into the 'ticket_prices' table in bulk.
    Updates the price if any city already exists in the table.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO ticket_prices (city, price)
            VALUES (?, ?)
            ON CONFLICT(city) DO UPDATE SET price = excluded.price
            """,
            [(city.strip(), float(price)) for city, price in records],
        )
        conn.commit()


def get_ticket_price_from_db(city: str, db_path: str = DB_PATH) -> float | None:
    """
    Retrieves the ticket price for a specified city (case-insensitive).
    Returns None if the city is not found.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT price FROM ticket_prices WHERE LOWER(city) = LOWER(?)",
            (city.strip(),),
        )
        row = cursor.fetchone()
        if row:
            return float(row["price"])
        return "Unknown destination. Please check the city name or try another destination."


def get_all_ticket_prices(db_path: str = DB_PATH) -> list[dict]:
    """
    Retrieves all records from the 'ticket_prices' table as a list of dicts.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT city, price FROM ticket_prices ORDER BY city ASC")
        rows = cursor.fetchall()
        return [{"city": row["city"], "price": row["price"]} for row in rows]


# Convenient aliases
create_table = create_ticket_prices_table
insert_record = insert_ticket_price_to_db

# ============================================================================
# TOOLS REGISTRY & FUNCTIONS
# ============================================================================

TOOL_REGISTRY = {}

def register_tool(name: str | None = None):
    """
    Decorator to register a function as an LLM tool in TOOL_REGISTRY.
    Supports usage with or without parentheses:
      @register_tool
      @register_tool("custom_name")
    """
    if callable(name):
        func = name
        TOOL_REGISTRY[func.__name__] = func
        return func

    def decorator(func):
        tool_name = name or func.__name__
        TOOL_REGISTRY[tool_name] = func
        return func

    return decorator


system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

@register_tool
def get_ticket_price(destination_city):
    print(f"Tool called for city {destination_city}")
    price = get_ticket_price_from_db(destination_city.lower())
    return f"The price of a ticket to {destination_city} is {price}"

@register_tool
def insert_ticket_price(destination_city, price):
    print(f"Tool called to insert price for {destination_city}: {price}")
    insert_ticket_price_to_db(destination_city.lower(), float(price))
    return f"The price of a ticket to {destination_city} has been set to ${float(price):.2f}"

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

insert_price_function = {
    "name": "insert_ticket_price",
    "description": "Insert or update the price of a ticket to a destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
            "price": {
                "type": "number",
                "description": "The price of the flight ticket",
            },
        },
        "required": ["destination_city", "price"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": price_function},
    {"type": "function", "function": insert_price_function},
]

# ============================================================================

def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        func_name = tool_call.function.name
        func = TOOL_REGISTRY.get(func_name)

        if func:
            try:
                arguments = json.loads(tool_call.function.arguments)
                content = func(**arguments)
            except Exception as e:
                content = f"Error executing tool '{func_name}': {e}"
        else:
            content = f"Error: Tool '{func_name}' is not recognized."

        responses.append({
            "role": "tool",
            "content": str(content),
            "tool_call_id": tool_call.id
        })
    return responses

def chat(message: str, history: list):
    """
    Handles multi-turn chat interactions with the Ollama model using the OpenAI client.
    Supports both dict-style and tuple-style history structures across Gradio versions.
    """
    messages = [{"role": "system", "content": system_message}]

    # Reconstruct multi-turn chat history
    for item in history:
        if isinstance(item, dict):
            # Gradio messages format: {"role": "user"|"assistant", "content": "..."}
            role = item.get("role")
            content = item.get("content")
            if role and content:
                messages.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            # Gradio tuple format: (user_msg, bot_msg)
            user_msg, bot_msg = item
            if user_msg:
                messages.append({"role": "user", "content": str(user_msg)})
            if bot_msg:
                messages.append({"role": "assistant", "content": str(bot_msg)})

    # Append the latest user prompt
    messages.append({"role": "user", "content": message})

    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        print(f"Response from Ollama atas: {response.choices[0]}")
        if response.choices[0].message.tool_calls:
            while response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                responses = handle_tool_calls(message)
                messages.append(message)
                messages.extend(responses)
                response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
                print(f"Response from Ollama bawah: {response.choices[0]}")
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ **Error communicating with Ollama:** {e}\n\nPlease verify that Ollama is running (`ollama serve`) and the model `{MODEL}` is pulled (`ollama pull {MODEL}`)."


# Set up Gradio ChatInterface
demo = gr.ChatInterface(
    fn=chat,
    title="Ollama AI Airline Assistant",
    description=f"Multi-turn streaming chatbot powered by local Ollama (`{MODEL}`) using the OpenAI Python SDK.",
    examples=[
        "What is the price of a flight to Los Angeles?",
        "What is the price of a flight to New York?",
        "What is the price of a flight to Chicago?",
        "Please set the ticket price to Miami to 250"
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    # Ensure database table exists and has initial data
    create_ticket_prices_table()
    if not get_all_ticket_prices():
        insert_ticket_prices([
            ("Los Angeles", 350.0),
            ("New York", 400.0),
            ("Chicago", 300.0),
        ])
    demo.launch()


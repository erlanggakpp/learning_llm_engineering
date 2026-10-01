import os
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import json
import base64
from io import BytesIO
from PIL import Image


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
    arguments = {}
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
    return responses, arguments

def chat(history: list):
    """
    Handles multi-turn chat interactions with the Ollama model using the OpenAI client.
    Receives updated history from chatbot component and returns (history, voice, image).
    """
    messages = [{"role": "system", "content": system_message}]
    cities = []

    # Reconstruct multi-turn chat history
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role and content:
                messages.append({"role": role, "content": content})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = item
            if user_msg:
                messages.append({"role": "user", "content": str(user_msg)})
            if bot_msg:
                messages.append({"role": "assistant", "content": str(bot_msg)})

    voice = None
    image = None

    try:
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
        )
        if response.choices[0].message.tool_calls:
            while response.choices[0].finish_reason == "tool_calls":
                msg = response.choices[0].message
                responses, arguments = handle_tool_calls(msg)
                city = arguments.get('destination_city') if arguments else None
                if city:
                    cities.append(city)
                messages.append(msg)
                messages.extend(responses)
                response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

        reply = response.choices[0].message.content or ""

        # Safely attempt TTS generation (requires OpenAI or a TTS provider)
        try:
            voice = talker(reply)
        except Exception as tts_err:
            print(f"TTS skipped/unavailable: {tts_err}")
            voice = None

        # Safely attempt Image generation if a city was requested
        if cities and cities[0]:
            try:
                image = artist(cities[0])
            except Exception as img_err:
                print(f"Image generation skipped/unavailable: {img_err}")
                image = None

        updated_history = history + [{"role": "assistant", "content": reply}]
        return updated_history, voice, image

    except Exception as e:
        error_msg = f"⚠️ **Error communicating with Ollama:** {e}\n\nPlease verify that Ollama is running (`ollama serve`) and the model `{MODEL}` is pulled (`ollama pull {MODEL}`)."
        updated_history = history + [{"role": "assistant", "content": error_msg}]
        return updated_history, None, None

def artist(city):
    print(f"Generating pop-art vacation image for {city}...")
    
    image_response = openai.images.generate(
        model="gemini-2.5-flash-image", # Google's image-supported model via OpenAI SDK
        prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
        size="1024x1024",
        n=1,
        response_format="b64_json"
    )
    
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

def talker(message):
    response = openai.audio.speech.create(
      model="gpt-4o-mini-tts",
      voice="onyx",    # Also, try replacing onyx with alloy or coral
      input=message
    )
    return response.content

def put_message_in_chatbot(message, history):
    return "", history + [{"role":"user", "content":message}]

if __name__ == "__main__":
    # Ensure database table exists and has initial data
    create_ticket_prices_table()
    if not get_all_ticket_prices():
        insert_ticket_prices([
            ("Los Angeles", 350.0),
            ("New York", 400.0),
            ("Chicago", 300.0),
        ])

    # UI definition
    with gr.Blocks() as ui:
        with gr.Row():
            chatbot = gr.Chatbot(height=500)
            image_output = gr.Image(height=500, interactive=False)
        with gr.Row():
            audio_output = gr.Audio(autoplay=True)
        with gr.Row():
            message = gr.Textbox(label="Chat with our AI Assistant:")

    # Hooking up events to callbacks

        message.submit(put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]).then(
            chat, inputs=chatbot, outputs=[chatbot, audio_output, image_output]
        )

    ui.launch(inbrowser=True)


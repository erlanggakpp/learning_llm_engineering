import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

# Load environment variables if available
load_dotenv()

# Model and client configuration (referencing week_2/gradio_interface.py)
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Initialize client pointing to local Ollama server
openai = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

system_message = "You are a helpful, thoughtful, and knowledgeable AI assistant."


def stream_chat(message: str, history: list):
    """
    Streams chatbot responses token-by-token from Ollama using the OpenAI client.
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
        stream = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
        )

        response_text = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            response_text += delta
            yield response_text

    except Exception as e:
        yield f"⚠️ **Error communicating with Ollama:** {e}\n\nPlease verify that Ollama is running (`ollama serve`) and the model `{MODEL}` is pulled (`ollama pull {MODEL}`)."


# Set up Gradio ChatInterface
demo = gr.ChatInterface(
    fn=stream_chat,
    title="Ollama Streaming Chatbot",
    description=f"Multi-turn streaming chatbot powered by local Ollama (`{MODEL}`) using the OpenAI Python SDK.",
    examples=[
        "Explain the Transformer architecture to a layperson",
        "Explain the Transformer architecture to an aspiring AI engineer",
        "What are the trade-offs of running LLMs locally vs in the cloud?",
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()

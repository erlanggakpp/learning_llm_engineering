import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live


MODEL = "qwen2.5"

# Initialize client (configured for local Ollama)
openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
console = Console()
display = console.print

# def shout(text):
#     print(f"Shout has been called with input {text}")
#     return text.upper()

# gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never").launch()

system_message = "You are a helpful assistant that responds in markdown without code blocks"

message_input = gr.Textbox(label="Your message:", info="Enter a message for Ollama", lines=7)
message_output = gr.Markdown(label="Response:")

def message_ollama(prompt):
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

# view = gr.Interface(
#     fn=message_ollama,
#     title="Ollama Chatbot", 
#     inputs=[message_input], 
#     outputs=[message_output], 
#     examples=[
#         "Explain the Transformer architecture to a layperson",
#         "Explain the Transformer architecture to an aspiring AI engineer",
#         ], 
#     flagging_mode="never"
#     )
# view.launch()

def stream_ollama(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result

message_input = gr.Textbox(label="Your message:", info="Enter a message for Ollama", lines=7)
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=stream_ollama,
    title="Ollama Chatbot", 
    inputs=[message_input], 
    outputs=[message_output], 
    examples=[
        "Explain the Transformer architecture to a layperson",
        "Explain the Transformer architecture to an aspiring AI engineer",
        ], 
    flagging_mode="never"
    )
view.launch()
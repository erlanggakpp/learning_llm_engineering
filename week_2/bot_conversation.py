import os
import json
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

MODEL = "qwen2.5"

# Initialize client (configured for local Ollama)
openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
console = Console()
display = console.print

snarky_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

polite_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."

absurdist_system = (
    "You are an eccentric, chaotic chatbot. You interpret everything literally"
    " or through surreal metaphors, derail debates with bizarre"
    " non-sequiturs, and treat mundane facts as astonishing revelations."
)

snarky_messages = ["Hi there"]
polite_messages = ["Hi"]
absurdist_messages = ["Hi"]

def call_snarky_system():
    messages = [{"role": "system", "content": snarky_system}]
    for snarky, polite in zip(snarky_messages, polite_messages):
        messages.append({"role": "assistant", "content": snarky})
        messages.append({"role": "user", "content": polite})
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

def call_absurdist_system():
    messages = [{"role": "system", "content": absurdist_system}]
    for absurdist, snarky in zip(absurdist_messages, snarky_messages):
        messages.append({"role": "assistant", "content": absurdist})
        messages.append({"role": "user", "content": snarky})
    messages.append({"role": "user", "content": snarky_messages[-1]})
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

# for i in range(5):
#     snarky_next = call_snarky_system()
#     display(Markdown(f"### Snarky:\n{snarky_next}\n"))
#     snarky_messages.append(snarky_next)

#     absurdist_next = call_absurdist_system()
#     display(Markdown(f"### Absurdist:\n{absurdist_next}\n"))
#     absurdist_messages.append(absurdist_next)

def call_polite_system():
    messages = [{"role": "system", "content": polite_system}]
    for polite, absurdist in zip(polite_messages, absurdist_messages):
        messages.append({"role": "assistant", "content": polite})
        messages.append({"role": "user", "content": absurdist})
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

hifdzi = "You are hifdzi, a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

rispo = "You are rispo, a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."

rigen = "You are rigen, a very sad and empathetic chatbot. You try to add sadness \
and empathy to the conversation as if these small moments of does not matter in the \
grand scheme of things."

conversation = [("hifdzi", "Hi there"), ("rispo", "Hi"), ("rigen", "Hello")]

def call_agent(agent_name, system_prompt, conversation):
    messages = [{"role": "system", "content": system_prompt}]
    convo_text = ""
    for name, msg in conversation:
        convo_text += f"{name}: {msg}\n"

    user_prompt = f"You are {agent_name}, in conversation with two other participants. \
    The conversation so far is as follows: {convo_text} \
    Now with this, respond with what you would like to say next, as {agent_name}."
    messages.append({"role": "user", "content": user_prompt})
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

for i in range(1):
    hifdzi = call_agent("hifdzi", hifdzi, conversation)
    display(Markdown(f"### hifdzi:\n{hifdzi}\n"))
    conversation.append(("hifdzi", hifdzi))

    rispo = call_agent("rispo", rispo, conversation)
    display(Markdown(f"### rispo:\n{rispo}\n"))
    conversation.append(("rispo", rispo))

    rigen = call_agent("rigen", rigen, conversation)
    display(Markdown(f"### rigen:\n{rigen}\n"))
    conversation.append(("rigen", rigen))

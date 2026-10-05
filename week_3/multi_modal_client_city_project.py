"""
Multimodal City Explorer - Gradio Client Application
================================================================================
Flow:
1. User types the name of a city.
2. Chat model generates an engaging overview of the city.
3. Text-to-Speech synthesizes the response into voice audio (autoplay).
4. If the city is a valid, recognized city, an image describing the city
   is generated using Stable Diffusion in your selected style ("Regular Photo" or "Pop Art").
   If invalid, no image is generated.

Prerequisites:
- Google Colab running `colab_multimodal_backend.ipynb` with ngrok active.
- `gradio`, `requests`, and `pillow` installed.
================================================================================
"""

import io
import os
import re
import requests
import gradio as gr
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Replace with your public ngrok URL from colab_multimodal_backend.ipynb
NGROK_URL = os.getenv("NGROK_URL", "https://plenty-panhandle-massive.ngrok-free.dev")

CHAT_ENDPOINT = f"{NGROK_URL}/chat"
TTS_ENDPOINT = f"{NGROK_URL}/tts"
IMAGE_ENDPOINT = f"{NGROK_URL}/generate-image"

# System prompt that instructs the LLM to provide travel details and flag validity
SYSTEM_PROMPT = """You are an enthusiastic and knowledgeable city travel guide.
When the user mentions or asks about a city:
1. If it is a real, recognized city or town:
   - Provide a vivid, engaging 2-3 sentence overview describing its most famous landmarks, food, culture, and vibe.
   - At the very end of your response, on a separate new line, output exactly:
     [VALID_CITY: YES | City Name, Country | Building: <Iconic Building/Landmark> | Food: <Famous Local Dish> | Culture: <Key Cultural Tradition/Element>]

2. If the user input is NOT a valid real city (e.g. gibberish, an animal, an everyday object, fictional place):
   - Politely explain that you only specialize in real cities around the world and ask them to name a valid city.
   - At the very end of your response, on a separate new line, output exactly:
     [VALID_CITY: NO]
"""


def parse_city_validity(raw_reply: str, user_text: str):
    """
    Parses the LLM reply to extract:
    1. The clean description text for display & TTS.
    2. A boolean indicating if it is a valid city.
    3. The recognized city name for the image prompt.
    4. A dictionary with extracted 'building', 'food', and 'culture'.
    """
    pattern = r"\[VALID_CITY:\s*([^\]]+)\]"
    match = re.search(pattern, raw_reply, re.IGNORECASE)
    is_valid = False
    city_name = user_text.strip()
    highlights = {"building": "", "food": "", "culture": ""}

    if match:
        content = match.group(1).strip()
        parts = [p.strip() for p in content.split("|")]
        status = parts[0].upper()
        if "YES" in status:
            is_valid = True
            if len(parts) > 1 and parts[1]:
                city_name = parts[1]
            for part in parts[2:]:
                if ":" in part:
                    k, v = part.split(":", 1)
                    k_clean = k.strip().lower()
                    if "building" in k_clean or "landmark" in k_clean:
                        highlights["building"] = v.strip()
                    elif "food" in k_clean or "culinary" in k_clean or "dish" in k_clean:
                        highlights["food"] = v.strip()
                    elif "culture" in k_clean or "tradition" in k_clean:
                        highlights["culture"] = v.strip()
        clean_reply = re.sub(pattern, "", raw_reply).strip()
    else:
        # Fallback heuristic if the structured tag was not produced
        clean_reply = raw_reply.strip()
        negative_signals = [
            "not a valid city",
            "not a real city",
            "not recognized as a city",
            "only specialize in real cities",
            "please name a valid city",
            "please provide a valid city",
        ]
        if any(sig in clean_reply.lower() for sig in negative_signals):
            is_valid = False
            city_name = ""
        else:
            is_valid = True
            city_name = user_text.strip()

    return clean_reply, is_valid, city_name, highlights


def extract_text(content):
    """Safely extracts plain string text from strings, Gradio 6 component lists, or dicts."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item and item["text"]:
                    parts.append(str(item["text"]).strip())
                elif "content" in item and item["content"]:
                    parts.append(str(item["content"]).strip())
            elif isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, (tuple, list)) and len(item) > 0:
                parts.append(str(item[0]).strip())
        return " ".join(parts).strip()
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "").strip()
    return str(content).strip()


def _normalize_history(chatbot):
    """Ensures chatbot history is always a list of {'role': str, 'content': str} dicts."""
    if not chatbot:
        return []
    normalized = []
    for item in chatbot:
        if isinstance(item, dict) and "role" in item and "content" in item:
            normalized.append({
                "role": str(item["role"]),
                "content": extract_text(item["content"]),
            })
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0] is not None:
                normalized.append({"role": "user", "content": extract_text(item[0])})
            if item[1] is not None:
                normalized.append({"role": "assistant", "content": extract_text(item[1])})
    return normalized


def put_message_in_chatbot(message, chatbot):
    """
    Appends the user's message to the chatbot history and clears the textbox.
    In Gradio messages format, each item must be a dict with 'role' and 'content'.
    """
    text = extract_text(message)
    if not text:
        return "", chatbot
    history = _normalize_history(chatbot)
    history.append({"role": "user", "content": text})
    return "", history


def chat(chatbot, image_style="Regular Photo"):
    """
    Main orchestration function:
    1. Sends query to Chat model (/chat).
    2. Synthesizes voice audio (/tts).
    3. If valid city, generates city image in chosen style (/generate-image).
    """
    history = _normalize_history(chatbot)
    if not history:
        return history, None, None

    # Retrieve user query from the most recent user message
    user_message = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            user_message = extract_text(msg.get("content", ""))
            if user_message:
                break

    if not user_message:
        return history, None, None

    # Guard check for ngrok placeholder
    if "YOUR_NGROK_URL" in NGROK_URL:
        err_msg = (
            "⚠️ NGROK_URL is not set! Please edit `multi_modal_client_city_project.py` "
            "and set NGROK_URL to your public Colab ngrok URL."
        )
        history.append({"role": "assistant", "content": err_msg})
        return history, None, None

    # 1. Chat Completion
    try:
        combined_prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_message}\nAssistant:"
        chat_payload = {
            "prompt": combined_prompt,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "message": user_message,
            "text": user_message,
            "max_tokens": 256,
            "temperature": 0.7,
        }
        res = requests.post(
            CHAT_ENDPOINT,
            json=chat_payload,
            timeout=60,
        )
        res.raise_for_status()
        data = res.json()
        if isinstance(data, dict):
            raw_reply = (
                data.get("reply")
                or data.get("response")
                or data.get("content")
                or data.get("text")
                or (data.get("choices") and data["choices"][0].get("message", {}).get("content"))
                or str(data)
            )
        elif isinstance(data, str):
            raw_reply = data
        else:
            raw_reply = str(data)
    except requests.exceptions.RequestException as e:
        server_detail = ""
        if hasattr(e, "response") and e.response is not None:
            server_detail = f" Server detail: {e.response.text}"
            print(f"[!] Server HTTP Error ({e.response.status_code}): {e.response.text}")
        err_msg = f"⚠️ Chat request failed ({e}).{server_detail}"
        history.append({"role": "assistant", "content": err_msg})
        return history, None, None
    except Exception as e:
        err_msg = f"⚠️ Unexpected error: {e}"
        history.append({"role": "assistant", "content": err_msg})
        return history, None, None

    # Parse validity, clean reply text, and extract city highlights
    clean_reply, is_valid, city_name, highlights = parse_city_validity(raw_reply, user_message)

    # Append assistant response
    history.append({"role": "assistant", "content": clean_reply})

    # 2. Text-to-Speech (TTS)
    audio_path = None
    try:
        tts_payload = {
            "text": clean_reply,
            "prompt": clean_reply,
            "input": clean_reply,
        }
        tts_res = requests.post(TTS_ENDPOINT, json=tts_payload, timeout=60)
        tts_res.raise_for_status()
        audio_path = "city_voice.wav"
        with open(audio_path, "wb") as f:
            f.write(tts_res.content)
    except Exception as e:
        print(f"[!] TTS generation failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"[!] TTS server detail: {e.response.text}")
        audio_path = None

    # 3. Image Generation (Only if city is valid)
    image_obj = None
    if is_valid and city_name:
        try:
            if image_style == "Pop Art":
                image_prompt = (
                    f"Vibrant pop art painting of {city_name}, iconic landmarks and skyline, "
                    f"bold contrasting colors, Roy Lichtenstein and Andy Warhol style, "
                    f"silkscreen print aesthetic, halftone dots, sharp outlines, masterpiece"
                )
            elif image_style == "Building, Food & Culture":
                bldg = highlights.get("building") or "iconic historic landmark and architectural monument"
                food = highlights.get("food") or "traditional authentic local food dishes"
                cult = highlights.get("culture") or "vibrant cultural traditions, festive street life"
                image_prompt = (
                    f"A detailed travel collage celebrating {city_name}: featuring {bldg} in the skyline, "
                    f"an appetizing plate of {food} in the foreground, and elements of {cult}, "
                    f"harmonious composition, vibrant colors, highly detailed, 8k resolution, cinematic lighting, masterpiece"
                )
            else:  # "Regular Photo"
                image_prompt = (
                    f"A stunning realistic photo of {city_name}, iconic landmarks and cityscape, "
                    f"high resolution 8k, natural lighting, beautiful architectural photography, "
                    f"photorealistic, sharp details, National Geographic style"
                )
            print(f"[*] Generating '{image_style}' image with prompt: \"{image_prompt}\"")
            img_payload = {
                "prompt": image_prompt,
                "text": image_prompt,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
            }
            img_res = requests.post(IMAGE_ENDPOINT, json=img_payload, timeout=120)
            img_res.raise_for_status()
            image_obj = Image.open(io.BytesIO(img_res.content))
            image_obj.save("city_image.png")
        except Exception as e:
            print(f"[!] Image generation failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"[!] Image generation server detail: {e.response.text}")
            image_obj = None
    else:
        print(f"[*] '{user_message}' is not a valid city. Skipping image generation.")

    return history, audio_path, image_obj


# =============================================================================
# Gradio Blocks UI Layout
# =============================================================================
with gr.Blocks(title="Multimodal City Explorer") as ui:
    gr.Markdown("# 🌆 Multimodal City Explorer")
    gr.Markdown(
        "Enter any city in the world to get a travel description, listen to the narrated voice, "
        "and view a generated image of the city in your preferred style!"
    )

    with gr.Row():
        chatbot = gr.Chatbot(height=500)
        image_output = gr.Image(height=500, interactive=False)
    with gr.Row():
        audio_output = gr.Audio(autoplay=True)
    with gr.Row():
        message = gr.Textbox(
            label="Chat with our AI Assistant:",
            placeholder="Enter a city name (e.g. Tokyo, Paris, Rome, New York)...",
        )
    with gr.Row():
        image_style = gr.Radio(
            label="Image Style",
            choices=["Regular Photo", "Pop Art", "Building, Food & Culture"],
            value="Regular Photo",
            info="Choose how you want the city to be visualized",
        )

    # Hooking up events to callbacks
    message.submit(
        put_message_in_chatbot,
        inputs=[message, chatbot],
        outputs=[message, chatbot],
    ).then(
        chat,
        inputs=[chatbot, image_style],
        outputs=[chatbot, audio_output, image_output],
    )

if __name__ == "__main__":
    ui.launch(inbrowser=True)

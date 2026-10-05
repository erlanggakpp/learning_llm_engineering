"""
Client script for Multimodal Remote Backend (Google Colab + ngrok).
Tests all 3 capabilities:
1. Chat Completions (/chat)
2. Text-to-Speech (/tts -> output.wav)
3. Image Generation (/generate-image -> output.png)

================================================================================
INSTRUCTIONS:
================================================================================
1. PREREQUISITES:
   Uses `requests` and `pillow`. Both are already available in your virtual environment.

2. GET THE NGROK URL:
   - Upload and run `colab_multimodal_backend.ipynb` in Google Colab (with T4 GPU).
   - Once Step 2 finishes loading, copy the public URL from the banner, e.g.:
     https://xxxx-xx-xx-xx-xx.ngrok-free.app

3. CONFIGURE THIS SCRIPT:
   - Update `NGROK_URL` below with your public ngrok URL.

4. RUN THIS SCRIPT:
   Execute in your terminal:
       python multimodal_client.py
   or with uv:
       uv run multimodal_client.py
================================================================================
"""

import io
import requests
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Replace with the public URL from the Colab notebook output banner:
NGROK_URL = "https://YOUR_NGROK_URL.ngrok-free.app"

CHAT_ENDPOINT = f"{NGROK_URL}/chat"
TTS_ENDPOINT = f"{NGROK_URL}/tts"
IMAGE_ENDPOINT = f"{NGROK_URL}/generate-image"


def test_chat():
    print("\n" + "=" * 60)
    print("1. TESTING CHAT COMPLETIONS (/chat)")
    print("=" * 60)
    payload = {
        "prompt": "Give me a one-sentence haiku or inspirational quote about artificial intelligence.",
        "max_tokens": 128,
        "temperature": 0.7,
    }
    print(f"[*] Prompt: \"{payload['prompt']}\"")
    try:
        response = requests.post(CHAT_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        print(f"[✓] Response from LLM:\n    {data.get('reply')}")
    except Exception as e:
        print(f"[!] Chat request failed: {e}")


def test_tts():
    print("\n" + "=" * 60)
    print("2. TESTING TEXT-TO-SPEECH (/tts)")
    print("=" * 60)
    tts_text = "Welcome to the world of open source AI. Your multimodal backend is fully operational."
    payload = {"text": tts_text}
    output_audio = "output.wav"
    print(f"[*] Text: \"{tts_text}\"")
    try:
        response = requests.post(TTS_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        with open(output_audio, "wb") as f:
            f.write(response.content)
        print(f"[✓] Audio successfully generated and saved to '{output_audio}'.")
    except Exception as e:
        print(f"[!] TTS request failed: {e}")


def test_image():
    print("\n" + "=" * 60)
    print("3. TESTING IMAGE GENERATION (/generate-image)")
    print("=" * 60)
    payload = {
        "prompt": "A futuristic floating cyberpunk city at sunset, neon lights reflections, cinematic lighting, 8k",
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
    }
    output_image = "output.png"
    print(f"[*] Prompt: \"{payload['prompt']}\"")
    try:
        response = requests.post(IMAGE_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        image.save(output_image)
        print(f"[✓] Image successfully generated and saved to '{output_image}'.")
    except Exception as e:
        print(f"[!] Image generation failed: {e}")


def main():
    if "YOUR_NGROK_URL" in NGROK_URL:
        print("[!] ERROR: Please update `NGROK_URL` with your actual ngrok public URL from Colab.")
        return

    test_chat()
    test_tts()
    test_image()
    print("\n[+] All tests completed!")


if __name__ == "__main__":
    main()

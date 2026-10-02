"""
Client script for Stable Diffusion 1.5 Remote Backend (Google Colab + ngrok).

================================================================================
INSTRUCTIONS:
================================================================================
1. PREREQUISITES:
   Ensure `requests` and `pillow` are installed in your environment.
   - If using the root virtual environment: they are already installed.
   - Alternatively: `pip install requests pillow` or `uv pip install requests pillow`.

2. GET THE NGROK URL:
   - Upload and run `colab_backend.ipynb` in Google Colab (with T4 GPU enabled).
   - Once Cell 2 starts, copy the public URL printed in the output, e.g.:
     https://xxxx-xx-xx-xx-xx.ngrok-free.app

3. CONFIGURE THIS SCRIPT:
   - Update `NGROK_URL` below with your copied ngrok URL.
   - Adjust `PROMPT` or inference parameters if desired.

4. RUN THIS SCRIPT:
   Execute in your terminal:
       python client.py
   or with uv:
       uv run client.py

5. VIEW OUTPUT:
   The generated image will be saved as `output.png` in the current folder.
================================================================================
"""

import io
import requests
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Replace with the public URL from the Colab notebook output:
NGROK_URL = "https://plenty-panhandle-massive.ngrok-free.dev"
ENDPOINT = f"{NGROK_URL}/generate"

# Prompt & Generation parameters
PAYLOAD = {
    "prompt": "A cozy, narrow street in Naples, Italy, focusing on a rustic open-air pizzeria with a wood-fired oven glowing in the background, fresh margherita pizzas on wooden paddles, bustling evening crowd, romantic lighting, oil painting style",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
}

OUTPUT_FILENAME = "output.png"


def main():
    if "YOUR_NGROK_URL" in NGROK_URL:
        print("[!] ERROR: Please update `NGROK_URL` with your actual ngrok public URL from Colab.")
        return

    print(f"[*] Sending prompt to: {ENDPOINT}")
    print(f"[*] Prompt: \"{PAYLOAD.get('prompt')}\"")

    try:
        response = requests.post(ENDPOINT, json=PAYLOAD, timeout=120)
        response.raise_for_status()

        # Read image bytes using Pillow
        image = Image.open(io.BytesIO(response.content))
        image.save(OUTPUT_FILENAME)
        print(f"[✓] Image successfully generated and saved to '{OUTPUT_FILENAME}'.")

    except requests.exceptions.RequestException as e:
        print(f"[!] Request failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"[!] Server details: {e.response.text}")
    except Exception as e:
        print(f"[!] An error occurred while processing the image: {e}")


if __name__ == "__main__":
    main()

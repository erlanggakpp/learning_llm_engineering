# Line-by-Line Code Explanation: Multimodal Colab Backend & Client

This document provides a line-by-line breakdown of both [`colab_multimodal_backend.ipynb`](colab_multimodal_backend.ipynb) and [`multimodal_client.py`](multimodal_client.py).

---

## Table of Contents
1. [`colab_multimodal_backend.ipynb` (Backend)](#1-colab_multimodal_backendipynb)
   - [Cell 1: Package Installations](#cell-1-package-installations)
   - [Cell 2: Core Server Implementation](#cell-2-core-server-implementation)
     - [Imports & Hardware Detection](#1-imports--hardware-detection)
     - [Loading the 3 AI Models](#2-loading-the-3-ai-models)
     - [FastAPI App & Pydantic Schemas](#3-fastapi-app--pydantic-schemas)
     - [Endpoint 1: Chat Completions (`/chat`)](#4-endpoint-1-chat-completions-chat)
     - [Endpoint 2: Text-to-Speech (`/tts`)](#5-endpoint-2-text-to-speech-tts)
     - [Endpoint 3: Image Generation (`/generate-image`)](#6-endpoint-3-image-generation-generate-image)
     - [Ngrok Tunnel & Uvicorn Runner](#7-ngrok-tunnel--uvicorn-runner)
2. [`multimodal_client.py` (Local Client)](#2-multimodal_clientpy)
   - [Imports & Target Configuration](#imports--target-configuration)
   - [`test_chat()` Function](#test_chat-function)
   - [`test_tts()` Function](#test_tts-function)
   - [`test_image()` Function](#test_image-function)
   - [`main()` Guard](#main-guard)
3. [Key Concepts & Architectural Decisions](#3-key-concepts--architectural-decisions)

---

## 1. `colab_multimodal_backend.ipynb`

### Cell 1: Package Installations

```bash
!pip install diffusers transformers accelerate fastapi uvicorn pyngrok nest-asyncio scipy
```
- `!` tells Jupyter/Colab to execute the line as a bash terminal command.
- `diffusers`: Hugging Face library providing the Stable Diffusion pipeline.
- `transformers`: Hugging Face library providing tokenizers and model architectures for Qwen (LLM) and MMS (TTS).
- `accelerate`: Optimizes model weight loading across CPU and GPU memory.
- `fastapi`: Modern, high-performance web framework for building Python APIs.
- `uvicorn`: ASGI web server implementation that executes the FastAPI application.
- `pyngrok`: Python wrapper around ngrok to create secure tunnels from Colab's private IP to the public web.
- `nest-asyncio`: Patches the running asyncio loop in Jupyter/Colab so Uvicorn can run without event loop conflicts.
- `scipy`: Used for audio processing (`scipy.io.wavfile`) to pack raw audio tensors into WAV bytes.

---

### Cell 2: Core Server Implementation

#### 1. Imports & Hardware Detection

```python
import io
```
- Standard Python library for handling in-memory byte streams (`io.BytesIO`) without needing to save temporary files to disk.

```python
from typing import List, Optional
```
- Provides type hints for data models (e.g., `Optional[str]`, `List[ChatMessage]`).

```python
import numpy as np
```
- NumPy is used for numeric calculations, such as converting audio waveforms into 16-bit PCM format.

```python
import scipy.io.wavfile as wavfile
```
- Handles encoding raw audio sample arrays into the standard `.wav` audio container format.

```python
import torch
```
- The PyTorch deep learning framework, which runs tensor calculations on the GPU.

```python
from diffusers import StableDiffusionPipeline
```
- High-level pipeline class that encapsulates the text encoder, UNet, VAE, and scheduler for Stable Diffusion 1.5.

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
```
- `FastAPI`: The application class.
- `HTTPException`: Standard way to return HTTP error status codes (like 400 or 500) with custom error messages.
- `Response`: Used to send raw binary data (image bytes, audio bytes) with custom MIME headers.

```python
from pydantic import BaseModel
```
- Base class for automatic request payload validation and JSON parsing.

```python
from pyngrok import ngrok
```
- Establishes the public tunnel to make the Colab server accessible over the internet.

```python
import nest_asyncio
import uvicorn
```
- `nest_asyncio` allows nested asynchronous event loops.
- `uvicorn` serves the FastAPI app.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, VitsModel
```
- `AutoModelForCausalLM`: Model class for text-generation LLMs (Qwen).
- `AutoTokenizer`: Converts raw strings into token IDs and vice versa.
- `VitsModel`: The neural network architecture used for the MMS Text-to-Speech model.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
- Automatically selects the NVIDIA GPU (`cuda`) if Colab runtime has a GPU enabled; otherwise falls back to `cpu`.

---

#### 2. Loading the 3 AI Models

##### Model A: Chat LLM (Qwen2.5-1.5B-Instruct)
```python
llm_id = "Qwen/Qwen2.5-1.5B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_id)
```
- Downloads and loads the tokenizer vocabulary and chat formatting template for Qwen 2.5.

```python
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
```
- Loads the neural network weights.
- `torch.float16`: Loads weights in half-precision (16-bit floats instead of 32-bit), cutting VRAM usage in half (~3 GB VRAM).
- `device_map="auto"`: Automatically places model layers onto available GPU memory.

##### Model B: Text-to-Speech (facebook/mms-tts-eng)
```python
tts_id = "facebook/mms-tts-eng"
tts_tokenizer = AutoTokenizer.from_pretrained(tts_id)
tts_model = VitsModel.from_pretrained(tts_id).to(device)
```
- Loads Meta's multilingual speech synthesizer (VITS architecture) trained on English.
- `.to(device)` transfers the model weights into GPU VRAM (uses only ~150 MB).

##### Model C: Image Generation (Stable Diffusion 1.5)
```python
sd_id = "runwayml/stable-diffusion-v1-5"
sd_pipe = StableDiffusionPipeline.from_pretrained(
    sd_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
sd_pipe = sd_pipe.to(device)
```
- Loads the pretrained Stable Diffusion 1.5 diffusion pipeline in 16-bit precision (~3.4 GB VRAM).
- Moves the entire text-to-image pipeline to the GPU.

---

#### 3. FastAPI App & Pydantic Schemas

```python
app = FastAPI(title="Multimodal AI Colab Backend")
```
- Instantiates the FastAPI application instance.

```python
class ChatMessage(BaseModel):
    role: str
    content: str
```
- Defines the structure of a single chat turn (e.g., `role="user"`, `content="Hello"`).

```python
class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    max_tokens: int = 512
    temperature: float = 0.7
```
- Accepts either a single `prompt` string or a multi-turn `messages` list.
- `max_tokens`: Maximum new tokens to generate.
- `temperature`: Controls randomness (higher = more creative, lower = more deterministic).

```python
class TTSRequest(BaseModel):
    text: str
```
- Schema for TTS: expects a single text string to synthesize.

```python
class ImageRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
```
- Schema for image generation:
  - `num_inference_steps`: Number of denoising iterations (30 is a great balance of speed and quality).
  - `guidance_scale`: How strongly the image adheres to the prompt (7.5 is standard).

---

#### 4. Endpoint 1: Chat Completions (`/chat`)

```python
@app.post("/chat")
@app.post("/v1/chat/completions")
def chat_endpoint(req: ChatRequest):
```
- Exposes both `/chat` and `/v1/chat/completions` (OpenAI format compatibility) for HTTP POST requests.

```python
    if req.messages:
        formatted_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    elif req.prompt:
        formatted_messages = [{"role": "user", "content": req.prompt}]
    else:
        raise HTTPException(status_code=400, detail="Provide either 'messages' or 'prompt'.")
```
- Normalizes the input into a standard list of role/content dictionaries. Rejects requests with neither.

```python
    prompt_text = llm_tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
```
- Formats the messages using the model's exact conversation template (e.g., `<|im_start|>user...<|im_end|><|im_start|>assistant`), preparing the model to reply.

```python
    inputs = llm_tokenizer([prompt_text], return_tensors="pt").to(device)
```
- Tokenizes the formatted text into PyTorch input ID tensors and moves them to the GPU.

```python
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature if req.temperature > 0 else None,
            do_sample=req.temperature > 0,
            pad_token_id=llm_tokenizer.pad_token_id or llm_tokenizer.eos_token_id,
        )
```
- `torch.no_grad()`: Disables gradient tracking during inference, saving significant memory and compute.
- `llm_model.generate()`: Generates the completion tokens.

```python
    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_len:]
    reply = llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
```
- Slices the output tensor so it contains **only** the newly generated tokens (excluding the prompt).
- Decodes the token IDs back into human-readable text and removes special control tokens.

```python
    return {
        "reply": reply,
        "choices": [{"message": {"role": "assistant", "content": reply}}],
    }
```
- Returns both a convenient `reply` field and an OpenAI-compatible `choices` array.

---

#### 5. Endpoint 2: Text-to-Speech (`/tts`)

```python
@app.post("/tts")
@app.post("/v1/audio/speech")
def tts_endpoint(req: TTSRequest):
```
- Exposes `/tts` and OpenAI-compatible `/v1/audio/speech`.

```python
    inputs = tts_tokenizer(req.text, return_tensors="pt").to(device)
    with torch.no_grad():
        waveform = tts_model(**inputs).waveform
```
- Tokenizes the input text into character phonemes and feeds them into the VITS model.
- `waveform`: A 1D tensor representing raw audio samples.

```python
    audio_data = waveform.squeeze().cpu().numpy()
    scaled = np.clip(audio_data, -1.0, 1.0)
    int16_pcm = (scaled * 32767).astype(np.int16)
```
- Removes extra dimensions (`squeeze()`) and copies data to CPU NumPy array.
- Scales floating-point audio in `[-1.0, 1.0]` to standard 16-bit signed PCM integers (`-32768` to `32767`) for universal audio player compatibility.

```python
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, rate=tts_model.config.sampling_rate, data=int16_pcm)
    wav_buffer.seek(0)
    return Response(content=wav_buffer.getvalue(), media_type="audio/wav")
```
- Writes the audio into an in-memory byte buffer with the model's native sampling rate (16,000 Hz).
- Rewinds the buffer (`seek(0)`) and sends raw WAV bytes with `media_type="audio/wav"`.

---

#### 6. Endpoint 3: Image Generation (`/generate-image`)

```python
@app.post("/generate-image")
@app.post("/generate")
def image_endpoint(req: ImageRequest):
```
- Exposes `/generate-image` (and `/generate` for backward compatibility).

```python
    image = sd_pipe(
        req.prompt,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
    ).images[0]
```
- Runs the diffusion process: text encoder -> latent UNet denoising loop -> VAE decoder to create a PIL Image object.

```python
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    return Response(content=img_buffer.getvalue(), media_type="image/png")
```
- Compresses the PIL image into PNG bytes in memory.
- Returns the bytes with `media_type="image/png"`.

---

#### 7. Ngrok Tunnel & Uvicorn Runner

```python
NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN_HERE"
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
```
- Authenticates your ngrok account so you can open public tunnels.

```python
public_url = ngrok.connect(8000).public_url
```
- Connects port 8000 to an internet-facing ngrok URL (e.g., `https://xxxx.ngrok-free.app`).

```python
nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)
```
- `nest_asyncio.apply()`: Patches the Jupyter notebook's active asyncio loop.
- `uvicorn.run(...)`: Starts listening for incoming HTTP requests on port 8000.

---

## 2. `multimodal_client.py`

### Imports & Target Configuration

```python
import io
import requests
from PIL import Image
```
- `requests`: Sends HTTP POST requests to the remote ngrok server.
- `PIL.Image`: Reads PNG byte streams and writes image files to disk.

```python
NGROK_URL = "https://YOUR_NGROK_URL.ngrok-free.app"
CHAT_ENDPOINT = f"{NGROK_URL}/chat"
TTS_ENDPOINT = f"{NGROK_URL}/tts"
IMAGE_ENDPOINT = f"{NGROK_URL}/generate-image"
```
- Defines the target URLs pointing to each respective capability on the backend.

---

### `test_chat()` Function

```python
def test_chat():
    payload = {
        "prompt": "Give me a one-sentence haiku or inspirational quote about artificial intelligence.",
        "max_tokens": 128,
        "temperature": 0.7,
    }
```
- Prepares the JSON payload requesting an LLM completion.

```python
    response = requests.post(CHAT_ENDPOINT, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    print(f"[✓] Response from LLM:\n    {data.get('reply')}")
```
- Posts the JSON payload.
- `raise_for_status()` raises an exception if the server returned an error (4xx/5xx).
- Decodes the JSON response and prints the assistant's answer.

---

### `test_tts()` Function

```python
def test_tts():
    tts_text = "Welcome to the world of open source AI. Your multimodal backend is fully operational."
    payload = {"text": tts_text}
    output_audio = "output.wav"
```
- Defines the sentence to be synthesized.

```python
    response = requests.post(TTS_ENDPOINT, json=payload, timeout=60)
    response.raise_for_status()
    with open(output_audio, "wb") as f:
        f.write(response.content)
```
- Posts the request to `/tts`.
- Reads `response.content` (raw binary WAV bytes) and writes them directly to `output.wav`.

---

### `test_image()` Function

```python
def test_image():
    payload = {
        "prompt": "A futuristic floating cyberpunk city at sunset, neon lights reflections, cinematic lighting, 8k",
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
    }
    output_image = "output.png"
```
- Configures the diffusion prompt and parameters.

```python
    response = requests.post(IMAGE_ENDPOINT, json=payload, timeout=120)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))
    image.save(output_image)
```
- Posts the prompt to `/generate-image`.
- Reads the raw bytes into a Pillow `Image` object via `io.BytesIO`.
- Saves the decoded image to `output.png`.

---

### `main()` Guard

```python
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
```
- Safety guard: Prevents failing network calls if the user forgot to replace the placeholder ngrok URL.
- Runs all three test suites sequentially.
- `if __name__ == "__main__":` ensures the code only executes when run directly as a script.

---

## 3. Key Concepts & Architectural Decisions

### Why Float16?
In `torch.float32`, each weight takes 4 bytes. In `torch.float16`, each weight takes 2 bytes. Loading Stable Diffusion (~1B params) and Qwen 2.5 (1.5B params) in `fp16` slashes memory consumption from ~10 GB to ~5 GB, allowing multiple heavy models to co-exist on a single 15 GB GPU.

### Why `io.BytesIO`?
Writing generated files to disk and then reading them back introduces unnecessary disk I/O latency. `io.BytesIO` creates an in-memory buffer that behaves like a file, allowing instant conversion from Python objects to HTTP network responses.

### Why `nest_asyncio`?
Jupyter notebooks run inside an active asyncio event loop. Calling `uvicorn.run()` directly inside Jupyter causes a `RuntimeError: This event loop is already running`. `nest_asyncio.apply()` enables re-entrant event loops, allowing Uvicorn to run seamlessly inside a Colab cell.

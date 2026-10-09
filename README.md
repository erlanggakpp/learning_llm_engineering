# Learning LLM Engineering

Personal learning and practice repository following the course **[LLM Engineering: Master AI and Large Language Models](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/)** by [Ed Donner](https://edwarddonner.com) on Udemy.

---

## 🎯 Repository Goals

- **Hands-on Implementation**: Reimplementing and extending the course lessons from Jupyter Notebooks (`.ipynb`) into standalone, modular Python (`.py`) applications and decoupled client-server systems.
- **Decoupled Architecture**: Running compute-intensive open-source models (e.g. `Qwen2.5-3B-Instruct`, `Whisper`, `Stable Diffusion 1.5`) remotely on free Google Colab GPUs (T4) with secure reverse tunnels (**`pyngrok`**), connected to local **`Gradio`** frontend clients.
- **Deep Quantization Benchmarks**: Rigorous performance profiling comparing unquantized **FP16** baselines against **4-Bit NormalFloat (NF4)** quantization (`bitsandbytes`), measuring VRAM footprints, peak activation overhead, latency, and token throughput.
- **Direct Tensor Generation vs. High-Level Pipelines**: Implementing low-level PyTorch tensor generation with manual prompt slicing (`outputs[0][prompt_len:]`) alongside Hugging Face `transformers.pipeline` abstractions.
- **Hybrid LLM Execution**: Running cloud APIs (OpenAI), local open-weight models via **Ollama** (e.g., `qwen2.5`, `phi3`, `deepseek-r1`) with automatic WSL2 host gateway discovery, and Hugging Face open weights.
- **Modern Python Tooling**: Using **`uv`** as a unified, blazingly fast package and virtual environment manager across all weekly modules.

---

## 📁 Project Structure

```text
learning_llm_engineering/
├── pyproject.toml                                # Root dependencies & environment configuration
├── uv.lock                                       # Pinned lockfile managed by uv
├── .venv/                                        # Shared virtual environment across all weeks
├── .env                                          # API keys & configuration (git-ignored)
├── README.md                                     # Main repository documentation
│
├── week_1/                                       # Foundations & Web Scraping
│   ├── day5.py                                   # Company brochure generator CLI (OpenAI + Rich)
│   ├── scraper.py                                # Web scraping utilities (Requests + BeautifulSoup)
│   └── day5_uv_guide.md                          # uv setup and CLI transition guide
│
├── week_2/                                       # Conversational AI, Tools & Streaming
│   ├── gradio_streaming_chatbot.py               # Real-time streaming chatbot with WSL2 Ollama resolution
│   ├── gradio_airline_assistant.py               # Airline assistant with SQLite function calling
│   ├── gradio_airline_assistant_multimodal.py    # Voice-enabled airline assistant with audio input
│   ├── gradio_interface.py                       # Basic Gradio chatbot interface
│   ├── bot_conversation.py                       # Automated multi-agent bot-to-bot dialogues
│   └── ticket_prices.db                          # SQLite flight database for tool execution
│
└── week_3/                                       # Open-Source Models, Multimodal & Quantization
    ├── runner_syntethic_data_client.py           # Gradio client for synthetic runner telemetry
    ├── runner_syntethic_data_notebook.ipynb      # Google Colab GPU backend for dual-model benchmarking
    ├── runner_syntethic_data_notebook.py         # Pure Python version of the runner telemetry backend
    ├── RUNNER_SYNTHETIC_DATA_NOTEBOOK_EXPLANATION.md # Line-by-line theoretical guide to notebook backend
    ├── RUNNER_SYNTHETIC_DATA_CLIENT_EXPLANATION.md   # Detailed guide to runner Gradio client
    ├── multi_modal_client_city_project.py        # Gradio client for multimodal city explorer
    ├── colab_multimodal_backend.ipynb            # Colab backend (Whisper + Qwen + Stable Diffusion)
    ├── COLAB_MULTIMODAL_BACKEND_EXPLANATION.md   # Architectural explanation for city explorer backend
    ├── presentation-EN.html / presentation-ID.html # Bilingual interactive slide decks
    ├── client.py                                 # Gradio client for audio meeting intelligence
    └── backend_pipeline.ipynb / backend_manual.ipynb # Whisper + Qwen meeting summarization pipelines
```

---

## 🚀 Weekly Module Highlights

### Week 1: Foundations & Website Brochure Generator
- **Focus**: OpenAI API essentials, prompt engineering, structured content extraction, and CLI formatting.
- **Highlights**:
  - Web scraping with `BeautifulSoup` to extract key company sections (About, Careers, Products).
  - Terminal-based company brochure generation styled with `rich` live markdown rendering.

### Week 2: Conversational AI, Streaming, Tool Calling & Multimodal Voice
- **Focus**: Interactive chatbots, function calling, state management, and multimodal audio input.
- **Highlights**:
  - **Dynamic WSL2 Gateway Discovery** in `gradio_streaming_chatbot.py`: Automatically inspects `/proc/sys/fs/binfmt_misc/WSLInterop` and resolves the Windows host gateway (`ip route | awk '/default/ {print $3}'`) to seamlessly connect to Ollama running on Windows from inside WSL2.
  - **SQLite Tool Calling** in `gradio_airline_assistant.py`: Equipped the LLM with structured tools to query flight schedules, check seat availability, and manage bookings from `ticket_prices.db`.
  - **Voice Airline Assistant**: Integrated audio transcription allowing users to speak their travel queries directly into the Gradio UI.

### Week 3: Open-Source Models, Multimodal Systems & Quantization Benchmarks
- **Focus**: Deploying open-weights causal LMs, quantizing models down to 4-bit, and building decoupled client-server systems.

#### 🏃 Featured Project: Synthetic Runner Telemetry Data Generator
A complete, decoupled two-file system comparing unquantized **FP16 Baseline** against **4-Bit NF4 Quantization** using `Qwen/Qwen2.5-3B-Instruct` on a free Google Colab T4 GPU:

1. **Google Colab Backend (`runner_syntethic_data_notebook.ipynb` / `.py`)**:
   - **Dual Model Loading**: Loads both FP16 (~6.18 GB VRAM) and 4-Bit NF4 (`BitsAndBytesConfig`, ~1.85 GB VRAM) models simultaneously on a single 16 GB T4 GPU.
   - **Static VRAM Savings**: Achieves an immediate **~70.1% reduction in weight memory**.
   - **Dual Inference Engine**:
     - *Normal Way (Direct Tensors)*: Explicit device placement (`.to(model.device)`), `torch.no_grad()`, `model.generate()`, prompt token slicing (`outputs[0][prompt_len:]`), and exact peak activation VRAM tracking.
     - *Pipeline Way (`transformers.pipeline`)*: High-level abstraction with an object-ID pipeline cache (`_pipeline_cache`).
     - Toggled seamlessly via the `USE_PIPELINE` flag or request payload.
   - **Biomechanical Sanity Engine**: Algorithmic physiological verification enforcing:
     - Heart rate cardinality (`max_heart_rate_bpm > avg_heart_rate_bpm`).
     - Cardiovascular biological boundaries (90–215 bpm).
     - Biomechanical cadence limits (140–205 spm).
     - Workout effort calibration across zones (Easy, Tempo, Interval, Long Run, Hill Repeats).
     - Kinematic motion consistency ($\text{Duration} \approx \text{Distance} \times \text{Pace}$).
   - **Kinematic Reconciliation & Multi-Line JSON Formatting**: Automatically corrects minor stochastic arithmetic rounding slips and serializes clean, indented JSON (`indent=2`).
   - **FastAPI + Pyngrok**: Exposes `/health` and `/generate` endpoints forwarded through an authenticated ngrok reverse tunnel and served via asynchronous `uvicorn`.

2. **Workstation Gradio Client (`runner_syntethic_data_client.py`)**:
   - **Interactive UI**: Dropdowns for workout type and athlete fitness profile, schema attribute checkboxes, record count slider, sampling temperature slider, and an **Inference Engine Mode** radio toggle.
   - **Real-Time Diagnostics**: Pings backend `/health` endpoint to display live VRAM allocation and active inference engine.
   - **Side-by-Side Presentation**: Displays outputs across tabs in formatted `gr.Code` blocks with syntax highlighting.
   - **Benchmark Comparison Table**: Real-time comparison table reporting static VRAM reduction %, peak generation VRAM savings %, latency, token counts, and dynamic throughput badges (`🚀 Speedup` vs `⏱️ Relative Throughput` explaining register dequantization overhead on T4 GPUs).
   - **Dataset File Export**: One-click download of generated datasets in `.json`, `.csv`, `.md`, plus the full benchmark report.
   - **Offline Demo Mode**: Built-in mock simulator allowing full UI exploration without an active Colab connection.

#### 🌆 Multimodal City Explorer (`multi_modal_client_city_project.py` & `colab_multimodal_backend.ipynb`)
- End-to-end multimodal pipeline:
  - Speech-to-text via **OpenAI Whisper**.
  - City historical knowledge synthesis via **Qwen2.5**.
  - Text-to-speech audio narration.
  - Text-to-image synthesis using **Stable Diffusion 1.5**.
- Interactive slide decks available in English (`presentation-EN.html`) and Indonesian (`presentation-ID.html`).

---

## 📊 Quantization Benchmark Findings (FP16 vs. 4-Bit NF4)

Testing conducted on a standard Nvidia T4 GPU (16 GB VRAM) using `Qwen/Qwen2.5-3B-Instruct`:

| Metric | FP16 Baseline (Unquantized) | 4-Bit NF4 Quantized (`bitsandbytes`) | Optimization / Impact |
| :--- | :---: | :---: | :--- |
| **Model Weight VRAM** | `~6,180 MB` | `~1,850 MB` | **🔻 70.1% Memory Reduction** |
| **Peak Generation VRAM** | `~6,800 MB` | `~2,400 MB` | **🔻 ~64.7% Peak Reduction** |
| **Throughput (Batch Size = 1)** | `~17.5 tok/s` | `~10.9 tok/s` | **⏱️ ~38% Slower** (Dequantization compute overhead) |
| **Physiological Accuracy** | 100% Passed | 100% Passed | **Zero loss in mathematical & domain fidelity** |
| **Hardware Capability** | Requires $\ge 8$ GB VRAM | Runs in $< 3$ GB VRAM | **Enables dual-model hosting on free T4 GPUs** |

> [!NOTE]
> **Why is 4-Bit NF4 slower than FP16 on a T4 GPU?**
> The Nvidia T4 possesses native FP16 Tensor Cores, but no native 4-bit floating-point compute engine. During each autoregressive decoding step, `bitsandbytes` must dynamically unpack and dequantize 4-bit weights into FP16 registers before multiplying them against activations. At batch size 1 on a 3B model, the arithmetic unpack overhead exceeds the memory bus transfer savings. Thus, 4-bit NF4 optimizes for **memory capacity**, while FP16 delivers higher raw throughput.

---

## 🛠️ Tech Stack & Dependencies

- **Runtime**: Python `>=3.12`
- **Environment & Package Manager**: [`uv`](https://docs.astral.sh/uv/)
- **Core Libraries**:
  - `gradio>=6.28.0`: High-performance reactive web interfaces.
  - `openai>=3.17.0`: Client for cloud APIs and local Ollama servers.
  - `requests>=2.34.2` & `beautifulsoup4>=4.15.0`: HTTP networking and web scraping.
  - `rich>=15.0.0`: Terminal styling, spinners, and live Markdown streaming.
  - `pillow>=12.3.0` & `reportlab>=5.0.1`: Image processing and PDF export.
  - `python-dotenv>=1.2.3`: Local `.env` secret management.
- **Backend Colab Stack**:
  - `torch`, `transformers`, `accelerate`, `bitsandbytes`
  - `fastapi`, `uvicorn`, `pyngrok`, `pydantic`

---

## 🚦 Quickstart Guide

### 1. Prerequisites
- Install **`uv`**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- *(Optional)* Install and start **[Ollama](https://ollama.com/)** for local models:
  ```bash
  ollama run qwen2.5
  ollama run phi3
  ```

### 2. Install Project Dependencies
Sync the virtual environment using `uv` from the repository root:
```bash
uv sync
```

### 3. Configure Environment Variables
Create a `.env` file in the repository root:
```bash
# Optional for Week 1 & Week 2 cloud inference:
OPENAI_API_KEY=sk-proj-your-key-here

# Optional for local Ollama overrides (auto-detected on WSL2):
# OLLAMA_BASE_URL=http://localhost:11434/v1
# OLLAMA_MODEL=qwen2.5
```

### 4. Running the Applications

#### Week 1: Website Brochure Generator
```bash
uv run python week_1/day5.py
```

#### Week 2: Real-Time Streaming Chatbot (with Ollama)
```bash
uv run python week_2/gradio_streaming_chatbot.py
```

#### Week 2: Airline Customer Assistant with SQLite Tools
```bash
uv run python week_2/gradio_airline_assistant.py
```

#### Week 3: Synthetic Runner Telemetry Client
1. Open and run all cells in [`runner_syntethic_data_notebook.ipynb`](file:///mnt/d/Software%20Engineering/learning_llm_engineering/week_3/runner_syntethic_data_notebook.ipynb) on a Google Colab T4 GPU.
2. Copy the public ngrok tunnel URL output by the final cell (e.g., `https://xxxx.ngrok-free.app`).
3. Launch the local Gradio client:
   ```bash
   uv run python week_3/runner_syntethic_data_client.py
   ```
4. Paste the ngrok URL into the interface, click **"Test Connection"**, and synthesize telemetry datasets with side-by-side quantization benchmarks.
*(Alternatively, enable **"Offline Demo Mode"** to test without a Colab backend).*

---

## 📖 In-Depth Technical Documentation

Detailed architectural and line-by-line theoretical guides are available in the repository:

- 📘 [Runner Synthetic Telemetry Backend Explanation](file:///mnt/d/Software%20Engineering/learning_llm_engineering/week_3/RUNNER_SYNTHETIC_DATA_NOTEBOOK_EXPLANATION.md)
- 📙 [Runner Synthetic Telemetry Client Explanation](file:///mnt/d/Software%20Engineering/learning_llm_engineering/week_3/RUNNER_SYNTHETIC_DATA_CLIENT_EXPLANATION.md)
- 📗 [Multimodal Colab Backend Explanation](file:///mnt/d/Software%20Engineering/learning_llm_engineering/week_3/COLAB_MULTIMODAL_BACKEND_EXPLANATION.md)
- 📕 [Audio Meeting Intelligence Client Explanation](file:///mnt/d/Software%20Engineering/learning_llm_engineering/week_3/client_explanation.md)
- 📓 [Meeting Intelligence Notebooks Guide](file:///mnt/d/Software%20Engineering/learning_llm_engineering/week_3/notebooks_explanation.md)

---

## 📚 Acknowledgments & Credits

- Course: **[LLM Engineering: Master AI and Large Language Models](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/)** by **Ed Donner** on Udemy.
- Course reference repository: [ed-donner/llm_engineering](https://github.com/ed-donner/llm_engineering).

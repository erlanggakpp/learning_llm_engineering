# Learning LLM Engineering

Personal learning and practice repository following the course **[LLM Engineering: Master AI and Large Language Models](https://www.udemy.com/course/llm-engineering-master-ai-and-large-language-models/)** by [Ed Donner](https://edwarddonner.com) on Udemy.

---

## 🎯 Repository Goals

- **Hands-on Implementation**: Reimplementing and extending the course lessons from Jupyter Notebooks (`.ipynb`) into standalone, modular Python (`.py`) applications.
- **Hybrid LLM Execution**: Running both cloud-based model APIs (OpenAI) and local open-weight models via **Ollama** (e.g., `qwen2.5`, `deepseek-r1`).
- **Modern Python Tooling**: Using **`uv`** as a unified, blazingly fast package and virtual environment manager across all weekly modules.
- **Enhanced Terminal Experience**: Using `rich` for live streaming, loading spinners, and styled terminal Markdown output.

---

## 📁 Project Structure

```
learning_llm_engineering/
├── pyproject.toml          # Root project dependencies & configuration
├── uv.lock                 # Pinned dependency lockfile
├── .venv/                  # Shared virtual environment across all weeks
├── .env                    # Environment variables & API keys (git-ignored)
├── week_1/                 # Week 1: Foundations & Website Brochure Generator
│   ├── day5.py             # Pure Python company brochure generator (CLI + Rich)
│   ├── scraper.py          # Web scraping utilities (Requests + BeautifulSoup)
│   └── day5_uv_guide.md    # Detailed uv setup & CLI transition guide
├── week_2/                 # (Upcoming weekly modules)
└── README.md
```

---

## 🛠️ Tech Stack & Dependencies

- **Runtime**: Python `>=3.12`
- **Package & Environment Manager**: [`uv`](https://docs.astral.sh/uv/)
- **Core Libraries**:
  - `openai`: Client interface for OpenAI-compatible APIs (OpenAI cloud & local Ollama).
  - `beautifulsoup4` & `requests`: Web scraping and HTML parsing.
  - `python-dotenv`: Environment configuration and secret management.
  - `rich`: Beautiful console output, live markdown streaming, and loading spinners.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Install **`uv`**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- *(Optional)* Install and start **[Ollama](https://ollama.com/)** for running local models:
  ```bash
  ollama run qwen2.5
  ```

### 2. Install Project Dependencies
Sync the virtual environment using `uv` from the repository root:
```bash
uv sync
```

### 3. Configure API Keys (If using cloud models)
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=sk-proj-your-key-here
```

### 4. Running Scripts

You can run scripts from anywhere in the repository using `uv run`:

```bash
# Run from repository root:
uv run python week_1/day5.py

# Or run from inside the weekly folder:
cd week_1
uv run python day5.py
```

---

## 📚 Acknowledgments & Credits

- Course: **LLM Engineering: Master AI and Large Language Models** by **Ed Donner** on Udemy.
- Course reference repository: [ed-donner/llm_engineering](https://github.com/ed-donner/llm_engineering).

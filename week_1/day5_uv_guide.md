# Project Setup Guide: Root `uv` Architecture for Multi-Week Projects

This guide walks you through setting up **`uv`** from scratch in the repository root (`learning_llm_engineering`). 

By initializing `uv` at the root, you create a single centralized environment that will seamlessly support **`week_1`**, **`week_2`**, **`week_3`**, and beyond without having to re-create virtual environments for every week.

---

## 1. Why Initialize `uv` in the Root Folder?

```
learning_llm_engineering/           <-- PROJECT ROOT (Initialized with `uv init`)
├── .venv/                          <-- Single shared virtual environment
├── pyproject.toml                  <-- Tracks all libraries across all weeks
├── uv.lock                         <-- Locks exact package versions
├── .env                            <-- Root API keys (OPENAI_API_KEY=...)
├── week_1/
│   ├── scraper.py                  <-- Copied helper from course materials
│   ├── day5.py                     <-- Your standard Python script
│   └── day5_uv_guide.md
├── week_2/                         <-- Automatically shares the same .venv
└── week_3/                         <-- Add new packages anytime with `uv add`
```

### Key Advantages:
1. **Zero Redundancy**: You don't need a separate `.venv` or package installations for each week.
2. **Directory Awareness**: When you are inside `week_1/` or `week_2/` and run `uv run python script.py`, `uv` automatically walks up the directory tree, finds the root `.venv`, and uses it.
3. **Effortless Expansion**: As upcoming weeks introduce new libraries (like LangChain, ChromaDB, Hugging Face, or Ollama), you just run `uv add <library>` at the root.

---

## 2. Step-by-Step Setup

### Step 1: Open Terminal in the Root Directory
Make sure your terminal is at the project root:
```bash
cd "/mnt/d/Software Engineering/learning_llm_engineering"
```

### Step 2: Initialize `uv` at the Root
Run the initialization command:
```bash
uv init --no-pin-python
```
*(Note: If `uv init` generates a default `hello.py` file in the root, you can safely delete it).*

This creates:
- `pyproject.toml`
- `.python-version`
- `.gitignore` (pre-configured for Python and `.venv`)

### Step 3: Add the Libraries for Day 5
Install the packages required for `day5.ipynb` and its helper `scraper.py`:

```bash
uv add openai python-dotenv requests beautifulsoup4
```

*(Optional Recommendation)* Install `rich` to render beautiful Markdown directly in the terminal:
```bash
uv add rich
```

Running `uv add` will automatically create the `.venv` folder and `uv.lock`.

### Step 4: Set Up Your `.env` File
Create a `.env` file in the root directory:
```bash
# In learning_llm_engineering/
echo "OPENAI_API_KEY=sk-proj-your-key-here" > .env
```
> [!IMPORTANT]
> Make sure your `.gitignore` contains `.env` so your API keys are never committed to git.

### Step 5: Copy `scraper.py` into `week_1/`
`day5` depends on `scraper.py` (which fetches and parses webpage HTML). Copy it from the course folder:
```bash
cp "/mnt/d/Software Engineering/llm_engineering/week1/scraper.py" "/mnt/d/Software Engineering/learning_llm_engineering/week_1/"
```

---

## 3. How to Run Your Python Code

Once you write your standard Python code in `week_1/day5.py`, you can run it from anywhere:

### From the Root Directory:
```bash
uv run python week_1/day5.py
```

### From Inside the `week_1/` Directory:
```bash
cd week_1
uv run python day5.py
```

> [!TIP]
> You **never** need to manually activate the virtual environment (`source .venv/bin/activate`). `uv run` handles environment activation automatically for every command!

---

## 4. How Future Weeks Will Work (`week_2`, `week_3`, etc.)

Whenever a future lesson introduces new packages:
1. Simply run `uv add <package>` in the project:
   ```bash
   # Example for future weeks:
   uv add langchain chromadb gradio
   ```
2. Create your folder `week_2/`, write your code, and execute:
   ```bash
   uv run python week_2/day1.py
   ```
All weeks share the central environment without conflicts or duplicate downloads.

---

## 5. Converting `day5.ipynb` to a Standard `.py` Script

When transferring logic from the notebook into `day5.py`:

### 1. Remove `IPython.display`
`IPython.display.Markdown` and `display()` only work inside Jupyter. In standard Python:
- **Standard Console Output**:
  ```python
  print(brochure_text)
  ```
- **Terminal Markdown Output (with `rich`)**:
  ```python
  from rich.console import Console
  from rich.markdown import Markdown

  console = Console()
  console.print(Markdown(brochure_text))
  ```

### 2. Streaming Responses in the Terminal
Instead of Jupyter's `update_display()`, stream tokens directly to standard output:
```python
stream = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": brochure_system_prompt},
        {"role": "user", "content": prompt}
    ],
    stream=True
)

for chunk in stream:
    content = chunk.choices[0].delta.content or ""
    print(content, end="", flush=True)

print()  # Add final newline
```

### 3. Load the `.env` from Root
At the top of your script:
```python
import os
from dotenv import load_dotenv
from openai import OpenAI

# Automatically searches for .env in current or parent directories
load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI()
```

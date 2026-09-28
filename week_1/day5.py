"""
Day 5: Automated Company Brochure Generator
Based on week1/day5.ipynb from LLM Engineering course.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Local helper module for web scraping (scraper.py in the same folder)
from scraper import fetch_website_links, fetch_website_contents

# Terminal formatting replacements for IPython.display
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

# ---------------------------------------------------------------------------
# 1. Environment & API Key Setup
# ---------------------------------------------------------------------------
# load_dotenv(override=True)
# api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------------
# 2. Client & Model Configuration
# ---------------------------------------------------------------------------
MODEL = "qwen2.5"

# Initialize client (configured for local Ollama)
openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
console = Console()
display = console.print  # Terminal replacement for notebook's display()


# ---------------------------------------------------------------------------
# 3. Prompts & Link Selection
# ---------------------------------------------------------------------------
def get_link_system_prompt():
    return """
    You are provided with a list of links found on a webpage.
    You are able to decide which of the links would be most relevant to include in a brochure about the company,
    such as links to an About page, or a Company page, or Careers/Jobs pages.
    You should respond in JSON as in this example:

    {
        "links": [
            {"type": "about page", "url": "https://full.url/goes/here/about"},
            {"type": "careers page", "url": "https://another.full.url/careers"}
        ]
    }
    """


def get_links_user_prompt(url):
    user_prompt = f"""
    Here is the list of links on the website {url} -
    Please decide which of these are relevant web links for a brochure about the company, 
    respond with the full https URL in JSON format.
    Do not include Terms of Service, Privacy, email links.

    Links (some might be relative links):

    """
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)
    return user_prompt


def select_relevant_links(url):
    console.print(f"[cyan]Selecting relevant links for {url} using {MODEL}...[/cyan]")
    get_links_user_prompt_value = get_links_user_prompt(url)
    get_link_system_prompt_value = get_link_system_prompt()
    # console.print(
    #     f"[green]✓ get_link_system_prompt: {get_link_system_prompt_value}.[/green]"
    # )
    # console.print(
    #     f"[green]✓ get_links_user_prompt: {get_links_user_prompt_value}.[/green]"
    # )
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": get_link_system_prompt_value},
            {"role": "user", "content": get_links_user_prompt_value},
        ],
        response_format={"type": "json_object"},
    )
    result = response.choices[0].message.content
    links = json.loads(result)
    console.print(
        f"[green]✓ Found {len(links.get('links', []))} relevant links.[/green]"
    )
    return links


def fetch_page_and_all_relevant_links(url):
    with console.status(
        f"[bold green]Scraping landing page and relevant links for {url}...[/bold green]"
    ):
        contents = fetch_website_contents(url)
        relevant_links = select_relevant_links(url)
        result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
        for link in relevant_links.get("links", []):
            result += f"\n\n### Link: {link['type']}\n"
            result += fetch_website_contents(link["url"])
    return result


# ---------------------------------------------------------------------------
# 4. Brochure Prompts & Generation (Console & Markdown)
# ---------------------------------------------------------------------------
brochure_system_prompt = """
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""


def get_brochure_user_prompt(company_name, url):
    user_prompt = f"""
You are looking at a company called: {company_name}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.

"""
    user_prompt += fetch_page_and_all_relevant_links(url)
    return user_prompt[:5000]  # Truncate if more than 5,000 characters


def create_brochure(company_name, url):
    """Generates brochure and renders it using Console and Markdown."""
    console.rule(f"[bold blue]Generating Brochure for {company_name}[/bold blue]")

    with console.status(
        f"[bold yellow]Analyzing website & writing brochure for {company_name}...[/bold yellow]"
    ):
        response = openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": brochure_system_prompt},
                {
                    "role": "user",
                    "content": get_brochure_user_prompt(company_name, url),
                },
            ],
        )
        result = response.choices[0].message.content

    # Using Console & Markdown to render styled headers, bullets, and bold text
    console.rule(f"[bold green]{company_name} Brochure[/bold green]")
    console.print(Markdown(result))


def stream_brochure(company_name, url):
    """Streams brochure live to the terminal, continuously rendering Markdown."""
    console.rule(f"[bold blue]Streaming Brochure for {company_name}[/bold blue]")

    prompt = get_brochure_user_prompt(company_name, url)

    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": prompt},
        ],
        # stream=True,
    )
    result = stream.choices[0].message.content
    display(Markdown(result))
    # console.rule(f"[bold green]{company_name} Live Stream[/bold green]")

    # 'Live' + 'Markdown' dynamically re-renders terminal Markdown as tokens arrive
    # full_response = ""
    # with Live(console=console, refresh_per_second=10) as live:
    #     for chunk in stream:
    #         token = chunk.choices[0].delta.content or ""
    #         full_response += token
    #         live.update(Markdown(full_response))


# ---------------------------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------------------------
def main():
    console.print(f"\n[bold]Initialized with model:[/bold] [cyan]{MODEL}[/cyan]\n")

    # Example 1: Demonstrate how console.print(Markdown(...)) renders
    demo_markdown = """
# Welcome to Day 5 Brochure Generator
This script replaces **Jupyter Notebook's** `display(Markdown(...))` with `rich.console.Console` and `rich.markdown.Markdown`.

### What was implemented:
- `console.print(Markdown(...))` for formatted markdown headers and bullet points
- `Live(console=console)` for real-time streaming markdown
- `console.status(...)` for interactive loading spinners
    """
    # console.print(select_relevant_links("https://edwarddonner.com"))

    # Example 2: To generate or stream a live company brochure, uncomment below:
    company = "Transfez"
    url = "https://www.transfez.com"
    stream_brochure(company, url)


if __name__ == "__main__":
    main()

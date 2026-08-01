"""
PDF Summary Tool — read a PDF and print a structured LLM summary.

Usage:
  python pdf_summary.py <path-to-pdf>
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def extract_text(pdf_path):
    """Extract text from each page of the PDF.  Returns (pages: list[str], page_count: int)."""
    try:
        import pdfplumber
    except ImportError:
        raise SystemExit(
            "pdfplumber is not installed.\n"
            "Run: pip install pdfplumber"
        )

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            print(f"Extracting page {i}/{total}...")
            text = page.extract_text()
            if text:
                pages.append(f"[Page {i}]\n{text.strip()}")
            else:
                pages.append(f"[Page {i}]\n(No extractable text on this page)")

    return pages, len(pdf.pages)


def build_prompt(pages, page_count):
    """Combine extracted pages into the user prompt."""
    numbered_text = "\n\n".join(pages)
    return f"""Here is the full text of a PDF document ({page_count} pages):

{numbered_text}

Summarise this document with exactly three sections:
1. Overview — a short paragraph covering the main topic
2. Key Points — exactly 3-5 bullets; every bullet MUST end with a [Page X] citation
3. Limitations — caveats, missing context, or extraction limits

Use ONLY information from the provided text. Do not invent facts."""


def ask_llm(prompt_text):
    """Send the prompt to OpenRouter and return the LLM response."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is missing. Add it to .env and try again.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model="qwen/qwen3.5-flash-02-23",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise research assistant. "
                    "Summarise ONLY from the provided text. "
                    "Provide exactly 3-5 Key Points. "
                    "Every Key Point MUST end with a [Page X] citation. "
                    "Do not invent facts or add outside knowledge."
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
    )

    return response.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python pdf_summary.py <path-to-pdf>")

    pdf_path = sys.argv[1]

    if not os.path.isfile(pdf_path):
        raise SystemExit(f"File not found: {pdf_path}")

    # Extract
    pages, page_count = extract_text(pdf_path)

    # Guard against fully unextractable PDFs
    all_empty = all("(No extractable text on this page)" in p for p in pages)
    if all_empty:
        raise SystemExit(
            "This PDF contains no extractable text (it may be a scanned document). "
            "Try a PDF with selectable text."
        )

    # Build prompt
    prompt_text = build_prompt(pages, page_count)

    # Call LLM
    print("正在生成摘要...\n")
    answer = ask_llm(prompt_text)
    print(answer)


if __name__ == "__main__":
    main()

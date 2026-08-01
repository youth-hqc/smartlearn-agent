"""
prompt_lab.py — compare three prompt quality levels on the same topic.
Sends Level A (vague), B (structured), C (precise) to OpenRouter and prints all three.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise SystemExit("OPENROUTER_API_KEY is missing. Add it to .env and try again.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


def ask(prompt: str) -> str:
    """Send one prompt to the AI and return the answer text."""
    response = client.chat.completions.create(
        model="qwen/qwen3.5-flash-02-23",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    return response.choices[0].message.content


# ============================================================
# Three levels of prompts — topic: Python Lists
# ============================================================

# Level A: Vague Prompt
prompt_a = "Explain Python lists"

# Level B: Structured Prompt (add a role + constraint)
prompt_b = (
    "You are a Python tutor for beginners. "
    "Explain Python lists in under 100 words."
)

# Level C: Precise Prompt (role + constraint + output format)
prompt_c = (
    "You are a Python tutor for beginners. Explain Python lists.\n"
    "Format:\n"
    "1) One-sentence definition\n"
    "2) Three common operations with code examples\n"
    "3) One common mistake to avoid"
)


# ============================================================
# Run the experiment and print the results
# ============================================================
if __name__ == "__main__":
    prompts = {
        "Level A (Vague)": prompt_a,
        "Level B (Structured)": prompt_b,
        "Level C (Precise)": prompt_c,
    }

    for level, prompt in prompts.items():
        if not prompt:
            print(f"\n{'='*60}")
            print(f"  {level}: (empty -- fill in the TODO above!)")
            print(f"{'='*60}")
            continue

        print(f"\n{'='*60}")
        print(f"  {level}")
        print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"{'='*60}")
        answer = ask(prompt)
        print(answer)

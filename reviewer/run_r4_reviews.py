#!/usr/bin/env python3
"""
Run R4 reviews via OpenAI (GPT-4o) and Gemini (2.5 Pro) APIs.
Keys are read from ../.env. Only used for this review task.
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

# Load .env
env_path = Path(__file__).parent.parent / '.env'
for line in env_path.read_text().strip().split('\n'):
    if '=' in line and not line.startswith('#'):
        k, v = line.split('=', 1)
        os.environ[k.strip()] = v.strip()

# Read paper
paper_path = Path(__file__).parent.parent / 'papers' / 'jsac' / 'main.tex'
paper_text = paper_path.read_text()

# Read prompts
chatgpt_prompt_path = Path(__file__).parent / 'PROMPT_chatgpt_R4_reviewer.md'
gemini_prompt_path = Path(__file__).parent / 'PROMPT_gemini_R4_reviewer.md'

chatgpt_system = chatgpt_prompt_path.read_text()
gemini_system = gemini_prompt_path.read_text()

# Strip the "[PASTE THE FULL main.tex CONTENT HERE]" placeholder
chatgpt_system = chatgpt_system.replace('[PASTE THE FULL main.tex CONTENT HERE]', '')
gemini_system = gemini_system.replace('[PASTE THE FULL main.tex CONTENT HERE]', '')

OUTPUT_DIR = Path(__file__).parent
today = datetime.now().strftime('%Y-%m-%d')


def run_chatgpt_review():
    """Run review via OpenAI GPT-4o."""
    print("[ChatGPT] Starting review with GPT-4o...")
    t0 = time.time()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": chatgpt_system},
                {"role": "user", "content": f"Here is the full paper (LaTeX source):\n\n{paper_text}"}
            ],
            temperature=0.3,
            max_tokens=8000,
        )

        review = response.choices[0].message.content
        outfile = OUTPUT_DIR / f'REVIEW_{today}_chatgpt_JSAC_round4.md'
        outfile.write_text(f"# JSAC R4 Review — ChatGPT (GPT-4o)\n\n"
                          f"**Date**: {today}\n"
                          f"**Model**: GPT-4o\n"
                          f"**Tokens**: {response.usage.total_tokens}\n\n"
                          f"---\n\n{review}\n")
        elapsed = time.time() - t0
        print(f"[ChatGPT] Done in {elapsed:.0f}s. Saved to {outfile.name}")
        print(f"[ChatGPT] Tokens used: {response.usage.total_tokens}")
        return review

    except Exception as e:
        elapsed = time.time() - t0
        print(f"[ChatGPT] FAILED after {elapsed:.0f}s: {e}")
        return None


def run_gemini_review():
    """Run review via Google Gemini 2.5 Pro."""
    print("[Gemini] Starting review with Gemini 2.5 Pro...")
    t0 = time.time()
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])

        model = genai.GenerativeModel('gemini-2.5-pro')

        full_prompt = gemini_system + "\n\n" + paper_text

        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=8000,
            ),
        )

        review = response.text
        outfile = OUTPUT_DIR / f'REVIEW_{today}_gemini_JSAC_round4.md'
        outfile.write_text(f"# JSAC R4 Review — Gemini 2.5 Pro\n\n"
                          f"**Date**: {today}\n"
                          f"**Model**: gemini-2.5-pro-preview-06-05\n\n"
                          f"---\n\n{review}\n")
        elapsed = time.time() - t0
        print(f"[Gemini] Done in {elapsed:.0f}s. Saved to {outfile.name}")
        return review

    except Exception as e:
        elapsed = time.time() - t0
        print(f"[Gemini] FAILED after {elapsed:.0f}s: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 60)
    print("R4 Review Runner — ChatGPT + Gemini in Parallel")
    print(f"Paper: {paper_path.name} ({len(paper_text)} chars)")
    print("=" * 60)

    results = {}
    threads = []

    def run_and_store(name, func):
        results[name] = func()

    t_gpt = threading.Thread(target=run_and_store, args=('chatgpt', run_chatgpt_review))
    t_gem = threading.Thread(target=run_and_store, args=('gemini', run_gemini_review))

    threads = [t_gpt, t_gem]
    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=300)  # 5 min max per review

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, review in results.items():
        if review:
            # Print first 500 chars as preview
            print(f"\n--- {name.upper()} (first 500 chars) ---")
            print(review[:500])
            print("...")
        else:
            print(f"\n--- {name.upper()}: FAILED ---")

    print(f"\nFull reviews saved in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Fresh-eyes Gemini review of papers — no anchoring from prior reviews.
Uses the REF_reviewer.md protocol for structured review.
"""

import json
import sys
import os
import urllib.request
import urllib.error
import time

API_KEY = "AIzaSyDz8DbncPthtLf17jAApxmaOKF24vSvEYo"
# Try gemini-2.5-pro first, fallback to gemini-2.0-flash
MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REVIEWER_DIR = os.path.join(BASE_DIR, "reviewer")

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def call_gemini(model, prompt, max_retries=2):
    """Call Gemini API with the given prompt."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "maxOutputTokens": 16384,
            "temperature": 0.7,
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                # Extract text from response
                if "candidates" in result and result["candidates"]:
                    parts = result["candidates"][0].get("content", {}).get("parts", [])
                    return "".join(p.get("text", "") for p in parts)
                else:
                    return f"ERROR: No candidates in response: {json.dumps(result, indent=2)}"
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 404:
                return None  # Model not found, try next
            if attempt < max_retries:
                print(f"  HTTP {e.code}, retrying in 5s... ({body[:200]})")
                time.sleep(5)
            else:
                return f"ERROR: HTTP {e.code}: {body[:500]}"
        except Exception as e:
            if attempt < max_retries:
                print(f"  Error: {e}, retrying in 5s...")
                time.sleep(5)
            else:
                return f"ERROR: {e}"

def find_working_model():
    """Find a Gemini model that works."""
    print("Finding available Gemini model...")
    for model in MODELS:
        print(f"  Trying {model}...")
        result = call_gemini(model, "Say 'OK' if you can read this.", max_retries=0)
        if result is not None and not result.startswith("ERROR"):
            print(f"  Using model: {model}")
            return model
        elif result is None:
            print(f"  Model {model} not found, trying next...")
        else:
            print(f"  Model {model} error: {result[:100]}")
    print("ERROR: No working Gemini model found!")
    sys.exit(1)

def build_review_prompt(paper_name, paper_content, review_protocol):
    """Build the review prompt combining protocol and paper."""
    return f"""You are an independent reviewer performing a fresh, unbiased review of an academic paper.
You have NEVER seen this paper before. You have NO prior reviews to anchor to. Start completely fresh.

## Review Protocol

{review_protocol}

## Important Notes

1. Be BRUTALLY honest. This paper has gone through many internal review rounds, and the authors are concerned about anchoring bias — each round only improved slightly. Your job is to be the fresh eyes that see what iterative reviewers missed.
2. Pay special attention to:
   - Fundamental logical flaws that might be hidden by good writing
   - Claims that don't match the actual data/experiments
   - Missing baselines or comparisons that would be obvious to a domain expert
   - Whether the contribution is truly novel or incremental
   - Whether the experimental methodology is rigorous enough for the target venue
3. Don't be gentle. If the paper has fatal flaws, say so. If it's solid, say so. But calibrate honestly.
4. Target venue context: This is targeting IEEE JSAC / INFOCOM / ICC — networking/communications venues, not ML venues.
5. Write your review in English.

## Paper to Review: {paper_name}

```latex
{paper_content}
```

Now write your complete, structured review following the protocol above. Be thorough and honest.
"""

def main():
    # Read review protocol
    protocol_path = os.path.join(BASE_DIR, "Tools", "REF_reviewer.md")
    review_protocol = read_file(protocol_path)

    # Papers to review
    papers = {
        "Paper-A (KV-Cache Compression)": os.path.join(BASE_DIR, "papers", "paper-A", "main.tex"),
        "Paper-B (Scout Protocol)": os.path.join(BASE_DIR, "papers", "paper-B", "main.tex"),
        "JSAC Merged Paper": os.path.join(BASE_DIR, "papers", "jsac", "main.tex"),
    }

    # Find working model
    model = find_working_model()

    # Process each paper
    os.makedirs(REVIEWER_DIR, exist_ok=True)

    for paper_name, paper_path in papers.items():
        print(f"\n{'='*60}")
        print(f"Reviewing: {paper_name}")
        print(f"{'='*60}")

        paper_content = read_file(paper_path)
        prompt = build_review_prompt(paper_name, paper_content, review_protocol)

        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  Calling Gemini ({model})...")

        start = time.time()
        review = call_gemini(model, prompt)
        elapsed = time.time() - start

        if review and not review.startswith("ERROR"):
            # Save review
            safe_name = paper_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
            filename = f"REVIEW_2026-02-26_gemini_{safe_name}.md"
            filepath = os.path.join(REVIEWER_DIR, filename)

            header = f"""# Gemini Fresh Review: {paper_name}

**Review Date**: 2026-02-26
**Reviewer**: Gemini ({model}) — fresh eyes, no prior review context
**Purpose**: Counter anchoring bias from iterative internal reviews
**Source**: {paper_path}

---

"""
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(header + review)

            print(f"  Done in {elapsed:.1f}s — saved to {filename}")
            print(f"  Review length: {len(review)} chars")
        else:
            print(f"  FAILED after {elapsed:.1f}s: {review[:200] if review else 'None'}")

    print(f"\n{'='*60}")
    print("All reviews complete!")
    print(f"Results in: {REVIEWER_DIR}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

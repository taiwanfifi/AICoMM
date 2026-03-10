# Cloud LLM API Reference

4 providers: OpenAI, Anthropic, Google Gemini, DeepSeek.

## Install

```bash
pip install openai anthropic google-genai requests python-dotenv
```

## API Keys (.env)

```env
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...
GEMINI_API_KEY=AIza...
DEEPSEEK_API_KEY=sk-...
HF_TOKEN=hf_...
```

```python
from dotenv import load_dotenv; load_dotenv()  # auto-loads .env
```

Key portals: [OpenAI](https://platform.openai.com/api-keys) | [Anthropic](https://console.anthropic.com/settings/keys) | [Google](https://aistudio.google.com/apikey) | [DeepSeek](https://platform.deepseek.com/api_keys) | [HuggingFace](https://huggingface.co/settings/tokens)

## Models

| Provider | Model | ID | Price (in/out per 1M tok) |
|----------|-------|----|---------------------------|
| OpenAI | GPT-4o | `gpt-4o` | $2.5 / $10 |
| OpenAI | GPT-4o mini | `gpt-4o-mini` | $0.15 / $0.6 |
| OpenAI | o1 | `o1` | $15 / $60 |
| Anthropic | Sonnet 4.5 | `claude-sonnet-4-5-20250929` | $3 / $15 |
| Anthropic | Haiku 4.5 | `claude-haiku-4-5-20251001` | $0.8 / $4 |
| Anthropic | Opus 4.6 | `claude-opus-4-6` | $15 / $75 |
| Google | Gemini 2.5 Flash | `gemini-2.5-flash` | free tier available |
| Google | Gemini 2.5 Pro | `gemini-2.5-pro` | paid |
| DeepSeek | Chat | `deepseek-chat` | ¥1 / ¥2 |
| DeepSeek | Reasoner | `deepseek-reasoner` | ¥2 / ¥8 |

## Usage Patterns

### OpenAI

```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY
r = client.chat.completions.create(
    model="gpt-4o", temperature=0.0, max_tokens=2048,
    messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}])
text = r.choices[0].message.content
```

### Anthropic

```python
import anthropic
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
r = client.messages.create(
    model="claude-sonnet-4-5-20250929", max_tokens=2048, temperature=0.0,
    system="...", messages=[{"role": "user", "content": "..."}])
text = r.content[0].text
```

### Google Gemini

```python
from google import genai
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
r = client.models.generate_content(
    model="gemini-2.5-flash", contents="...",
    config={"temperature": 0.0, "max_output_tokens": 4096})
text = r.text
```

### DeepSeek (OpenAI-compatible SDK)

```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
r = client.chat.completions.create(
    model="deepseek-chat", temperature=0.0, max_tokens=2048,
    messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}])
text = r.choices[0].message.content
```

### HuggingFace (Local/GPU Models)

```bash
pip install huggingface_hub transformers torch
huggingface-cli login --token $HF_TOKEN
```

**Gated model access:**

| Model | ID | Status |
|-------|----|--------|
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B` | Granted |

```python
import os
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
```

## Gotchas

- **Gemini**: silently truncates output -- always set `max_output_tokens=4096+`. Safety filters may also silently block responses; disable with `safety_settings` in config.
- **DeepSeek Reasoner**: response has both `reasoning_content` (CoT) and `content` (final answer). Server is in China, may be slow -- add timeout.
- **Timeout/retry**: `OpenAI(timeout=30.0, max_retries=3)` / `anthropic.Anthropic(timeout=30.0, max_retries=3)`
- **HuggingFace gated models**: Need to request access on model page + set `HF_TOKEN` env var. 401 errors mean token missing or access not granted.

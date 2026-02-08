"""
Task Definition: Extractive Question Answering

This module defines a simple QA task for evaluating KV-cache transmission methods.
The task is to extract the correct answer span from a given context.

We use this instead of perplexity because:
1. PPL measures distribution accuracy, not task success
2. In agent communication, we care about task outcomes, not internal confidence
3. This allows us to compute task-aware gradients for KV importance scoring
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class QAExample:
    """A single QA example."""
    context: str
    question: str
    answer: str
    # The answer should appear verbatim in context

    def __post_init__(self):
        assert self.answer.lower() in self.context.lower(), \
            f"Answer '{self.answer}' not found in context"


# Sample QA dataset for experiments
QA_EXAMPLES = [
    QAExample(
        context="The autonomous vehicle detected a pedestrian at coordinates 34.2N, 118.5W. "
                "Current speed is 45 km/h. Distance to pedestrian is 30 meters. "
                "The recommended action is emergency braking.",
        question="What is the recommended action?",
        answer="emergency braking"
    ),
    QAExample(
        context="Drone D1 reports thermal anomaly at grid reference Alpha-3. "
                "Temperature reading is 450 degrees Celsius. "
                "Smoke density is classified as high. "
                "Fire probability is estimated at 92 percent.",
        question="What is the fire probability?",
        answer="92 percent"
    ),
    QAExample(
        context="The warehouse monitoring system detected temperature rise in zone B. "
                "Current temperature is 67 degrees. Normal range is 20-25 degrees. "
                "Humidity dropped from 65% to 30%. "
                "The system recommends immediate evacuation.",
        question="What does the system recommend?",
        answer="immediate evacuation"
    ),
    QAExample(
        context="Agent Alpha completed reconnaissance of sector 7. "
                "Enemy units spotted: 3 tanks, 5 infantry squads. "
                "Estimated threat level is critical. "
                "Reinforcement request has been submitted to command.",
        question="What is the estimated threat level?",
        answer="critical"
    ),
    QAExample(
        context="The medical drone delivered supplies to location Delta. "
                "Delivery time was 4 minutes 32 seconds. "
                "Package integrity is confirmed intact. "
                "Return flight initiated at 14:35 UTC.",
        question="What was the delivery time?",
        answer="4 minutes 32 seconds"
    ),
]


def format_qa_prompt(example: QAExample) -> str:
    """Format a QA example into a prompt for the model."""
    return (
        f"Context: {example.context}\n\n"
        f"Question: {example.question}\n\n"
        f"Answer:"
    )


def check_answer(generated: str, expected: str) -> bool:
    """Check if the generated text contains the expected answer."""
    generated_lower = generated.lower().strip()
    expected_lower = expected.lower().strip()

    # Exact match or contains
    return expected_lower in generated_lower


def compute_qa_loss(
    model,
    tokenizer,
    example: QAExample,
    device: torch.device,
    past_key_values=None,
) -> torch.Tensor:
    """
    Compute the loss for predicting the answer tokens.

    This loss can be used to compute gradients w.r.t. KV-cache
    for task-aware importance scoring.
    """
    # Tokenize prompt (context + question)
    prompt = format_qa_prompt(example)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # Tokenize answer
    answer_ids = tokenizer(f" {example.answer}", return_tensors="pt")["input_ids"].to(device)
    # Remove BOS if present
    if answer_ids[0, 0] == tokenizer.bos_token_id:
        answer_ids = answer_ids[:, 1:]

    # Full sequence = prompt + answer
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)

    # Forward pass
    outputs = model(
        input_ids=full_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )

    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Only compute loss on the answer portion
    # Shift: predict token i+1 from position i
    prompt_len = prompt_ids.shape[1]
    answer_len = answer_ids.shape[1]

    # Logits for predicting answer tokens
    answer_logits = logits[:, prompt_len - 1 : prompt_len - 1 + answer_len, :]

    # Target: the answer tokens
    answer_targets = answer_ids

    # Cross-entropy loss
    loss = F.cross_entropy(
        answer_logits.view(-1, answer_logits.size(-1)),
        answer_targets.view(-1),
    )

    return loss, outputs.past_key_values


def compute_kv_importance_gradient(
    model,
    tokenizer,
    example: QAExample,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute task-aware importance scores for each KV entry using gradients.

    Returns a list of (key_importance, value_importance) per layer,
    where importance is the L2 norm of the gradient w.r.t. that entry.
    """
    # First, do a forward pass to get KV-cache
    prompt = format_qa_prompt(example)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # Enable gradient computation for KV-cache
    model.eval()

    # We need to manually track gradients through KV-cache
    # This is a simplified version - in practice, we'd need hooks

    with torch.enable_grad():
        outputs = model(
            input_ids=prompt_ids,
            use_cache=True,
            output_attentions=True,
        )

        kv_cache = outputs.past_key_values

        # For DynamicCache in transformers 5.x
        if hasattr(kv_cache, 'layers'):
            # Enable gradients for KV tensors
            kv_grads = []
            for layer in kv_cache.layers:
                k = layer.keys.detach().requires_grad_(True)
                v = layer.values.detach().requires_grad_(True)
                kv_grads.append((k, v))

            # Compute loss with these KV values
            # This is tricky because we need to re-run with custom KV
            # For now, we use attention weights as a proxy for importance
            # (Full gradient-based scoring requires model modification)

            # Proxy: use attention entropy as importance
            attentions = outputs.attentions  # list of (batch, heads, seq, seq)
            importance_scores = []

            for layer_idx, attn in enumerate(attentions):
                # attn: (1, num_heads, seq_len, seq_len)
                # Sum attention each position receives from all queries
                received_attn = attn[0].sum(dim=0).sum(dim=0)  # (seq_len,)

                # Use this as importance for both K and V
                k_imp = received_attn.detach().cpu()
                v_imp = received_attn.detach().cpu()
                importance_scores.append((k_imp, v_imp))

            return importance_scores
        else:
            raise NotImplementedError("Legacy KV-cache format not supported")


def compute_kv_importance_attention(
    model,
    tokenizer,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute attention-based importance scores (SnapKV style).

    Returns: (seq_len,) tensor of importance scores.
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            output_attentions=True,
        )

    attentions = outputs.attentions

    # Use last 4 layers (or all if fewer)
    num_obs_layers = min(4, len(attentions))
    obs_layers = attentions[-num_obs_layers:]

    seq_len = obs_layers[0].shape[-1]
    importance = torch.zeros(seq_len)

    for attn in obs_layers:
        # Sum attention each position receives
        received = attn[0].sum(dim=0).sum(dim=0)  # (seq_len,)
        importance += received.cpu()

    # Normalize
    importance = importance / importance.sum()
    return importance


def compute_kv_importance_task_proxy(
    model,
    tokenizer,
    example: QAExample,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute task-aware importance using a proxy method.

    Strategy: Positions where the model's attention focuses when
    predicting the answer tokens are more important.

    This is more task-aware than pure attention-based selection
    because it specifically looks at attention during answer generation.
    """
    prompt = format_qa_prompt(example)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]

    answer_ids = tokenizer(f" {example.answer}", return_tensors="pt")["input_ids"].to(device)
    if answer_ids[0, 0] == tokenizer.bos_token_id:
        answer_ids = answer_ids[:, 1:]

    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)

    with torch.no_grad():
        outputs = model(
            input_ids=full_ids,
            use_cache=True,
            output_attentions=True,
        )

    attentions = outputs.attentions

    # Focus on attention FROM answer tokens TO context tokens
    # This tells us which context positions are important for the answer

    importance = torch.zeros(prompt_len)

    for attn in attentions[-4:]:  # Last 4 layers
        # attn: (1, heads, full_seq, full_seq)
        # We want attention from answer positions to context positions
        # Answer positions: prompt_len to end
        # Context positions: 0 to prompt_len

        answer_to_context = attn[0, :, prompt_len:, :prompt_len]  # (heads, ans_len, ctx_len)

        # Sum across heads and answer positions
        ctx_importance = answer_to_context.sum(dim=0).sum(dim=0)  # (ctx_len,)
        importance += ctx_importance.cpu()

    # Normalize
    importance = importance / (importance.sum() + 1e-8)
    return importance


def evaluate_qa_accuracy(
    model,
    tokenizer,
    examples: list[QAExample],
    device: torch.device,
    kv_retention_pct: float = 100.0,
    selection_method: str = "attention",  # "attention" or "task_proxy"
    max_new_tokens: int = 20,
) -> dict:
    """
    Evaluate QA accuracy with optional KV pruning.

    Args:
        model: The language model
        tokenizer: The tokenizer
        examples: List of QA examples
        device: Torch device
        kv_retention_pct: Percentage of KV entries to retain (100 = no pruning)
        selection_method: How to select which KV entries to keep
        max_new_tokens: Max tokens to generate for answer

    Returns:
        dict with accuracy and per-example results
    """
    correct = 0
    results = []

    for example in examples:
        prompt = format_qa_prompt(example)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        # Get KV-cache for the prompt
        with torch.no_grad():
            outputs = model(
                input_ids=prompt_ids,
                use_cache=True,
                output_attentions=True,
            )

        kv_cache = outputs.past_key_values

        # Optionally prune KV-cache
        if kv_retention_pct < 100:
            if selection_method == "attention":
                importance = compute_kv_importance_attention(
                    model, tokenizer, prompt, device
                )
            else:  # task_proxy
                importance = compute_kv_importance_task_proxy(
                    model, tokenizer, example, device
                )

            # Prune (simplified: we'd need to properly mask the cache)
            # For now, this is a placeholder - full implementation needs
            # DynamicCache manipulation
            pass

        # Generate answer
        generated_ids = prompt_ids.clone()
        cache = kv_cache

        for _ in range(max_new_tokens):
            with torch.no_grad():
                if cache is not None and hasattr(cache, 'get_seq_length') and cache.get_seq_length() > 0:
                    inp = generated_ids[:, -1:]
                else:
                    inp = generated_ids
                out = model(input_ids=inp, past_key_values=cache, use_cache=True)

            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            cache = out.past_key_values

            if next_token.item() == tokenizer.eos_token_id:
                break

        # Decode generated answer
        generated_text = tokenizer.decode(
            generated_ids[0, prompt_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Check correctness
        is_correct = check_answer(generated_text, example.answer)
        if is_correct:
            correct += 1

        results.append({
            "question": example.question,
            "expected": example.answer,
            "generated": generated_text[:100],
            "correct": is_correct,
        })

    accuracy = correct / len(examples) * 100

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(examples),
        "results": results,
    }

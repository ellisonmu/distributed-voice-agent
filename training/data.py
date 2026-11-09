from __future__ import annotations

import json

import torch

IGNORE = -100


def encode_example(tokenizer, messages: list[dict], target: str, max_len: int) -> dict | None:
    """Prompt tokens are masked out of the loss; only the target (router token + answer or
    filler) is learned. Returns None if the example doesn't fit in max_len."""
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(target_ids) > max_len:
        return None
    return {"input_ids": prompt_ids + target_ids, "labels": [IGNORE] * len(prompt_ids) + target_ids}


def load_examples(path: str, tokenizer, max_len: int) -> tuple[list[dict], int]:
    examples, dropped = [], 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            ex = encode_example(tokenizer, row["messages"], row["target"], max_len)
            if ex is None:
                dropped += 1
            else:
                examples.append(ex)
    return examples, dropped


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        attention[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}

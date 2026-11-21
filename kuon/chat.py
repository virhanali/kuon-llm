"""Minimal chat formatting and loss-masking helpers for SFT.

Kuon has a single special token (`<endoftext>`), so chat markers are plain
text: `User:` / `Assistant:`. That is enough to teach the format; adding
dedicated special tokens would require retraining the tokenizer and is a
deliberate later step.
"""

from __future__ import annotations

USER_PREFIX = "User: "
ASSISTANT_PREFIX = "Assistant:"


def format_prompt(user: str) -> str:
    """The part of the conversation the model is given (never trained on).

    Ends at ':' on purpose: a trailing space would tokenize differently
    alone than when followed by the answer word (" Hello" is one chunk),
    which breaks generation."""
    return f"{USER_PREFIX}{user.strip()}\n{ASSISTANT_PREFIX}"


def format_example(user: str, assistant: str) -> str:
    """Full training text for one turn."""
    return f"{format_prompt(user)} {assistant.strip()}"


def build_example(tokenizer, user: str, assistant: str, block_size: int):
    """Tokenize one SFT turn into (input_ids, labels).

    Labels are the next-token targets; everything belonging to the prompt is
    masked with -100 so the loss only measures the assistant's answer.
    """
    prompt_ids = tokenizer.encode(format_prompt(user))
    full_ids = tokenizer.encode(format_example(user, assistant), add_eos=True)

    input_ids = full_ids[:-1]
    labels = full_ids[1:]

    n_mask = max(0, len(prompt_ids) - 1)  # last prompt token predicts the first answer token
    for i in range(min(n_mask, len(labels))):
        labels[i] = -100

    return input_ids[:block_size], labels[:block_size]

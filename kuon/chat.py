"""Minimal chat formatting and loss-masking helpers for SFT.

Kuon has a single special token (`<endoftext>`), so chat markers are plain
text: `User:` / `Assistant:`. That is enough to teach the format; adding
dedicated special tokens would require retraining the tokenizer and is a
deliberate later step.
"""

from __future__ import annotations

USER_PREFIX = "User: "
ASSISTANT_PREFIX = "Assistant:"
BUSINESS_INSTRUCTIONS = (
    "Use the user's stated facts, not guesses. If a key fact is missing, ask one question. "
    "Otherwise suggest one action and how to check it. Reply in the user's language.\n\n"
)


def format_prompt(user: str, *, evidence: str | None = None, business: bool = False) -> str:
    """The part of the conversation the model is given (never trained on).

    Ends at ':' on purpose: a trailing space would tokenize differently
    alone than when followed by the answer word (" Hello" is one chunk),
    which breaks generation."""
    context = f"\nEvidence:\n{evidence.strip()}" if evidence is not None else ""
    instructions = BUSINESS_INSTRUCTIONS if business else ""
    return f"{instructions}{USER_PREFIX}{user.strip()}{context}\n{ASSISTANT_PREFIX}"


def format_example(user: str, assistant: str, *, evidence: str | None = None,
                   business: bool = False) -> str:
    """Full training text for one turn."""
    return f"{format_prompt(user, evidence=evidence, business=business)} {assistant.strip()}"


def build_example(tokenizer, user: str, assistant: str, block_size: int,
                  *, evidence: str | None = None, business: bool = False):
    """Tokenize one SFT turn into (input_ids, labels).

    Labels are the next-token targets; everything belonging to the prompt is
    masked with -100 so the loss only measures the assistant's answer.
    """
    prompt_ids = tokenizer.encode(format_prompt(user, evidence=evidence, business=business))
    full_ids = tokenizer.encode(
        format_example(user, assistant, evidence=evidence, business=business), add_eos=True)
    if len(full_ids) > block_size + 1:
        return [], []  # do not teach a clipped answer without its stop token

    input_ids = full_ids[:-1]
    labels = full_ids[1:]

    n_mask = max(0, len(prompt_ids) - 1)  # last prompt token predicts the first answer token
    for i in range(min(n_mask, len(labels))):
        labels[i] = -100

    return input_ids, labels

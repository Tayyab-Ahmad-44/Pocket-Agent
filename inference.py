"""
Grader entry point. Exposes:

    def run(prompt: str, history: list[dict]) -> str

This loads the quantized GGUF model via llama-cpp-python (pure CPU) and emits
either a <tool_call>...</tool_call> block or a plain-text refusal.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Optional

from llama_cpp import Llama


# --------------------------------------------------------------------------- #
# System prompt — must match what the training data used verbatim. If you   #
# change the system prompt you'll degrade tool-call accuracy on adversarial  #
# paraphrases because the model learned to condition on this exact text.    #
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "You are Pocket-Agent, a compact on-device assistant.\n\n"
    "You have access to exactly five tools. When the user's request clearly "
    "maps to one of them, respond ONLY with a tool call wrapped in "
    "<tool_call>...</tool_call> tags containing valid JSON. When no tool fits, "
    "respond with plain natural language — no tags, no JSON.\n\n"
    "Available tools:\n"
    '{"tool": "weather",  "args": {"location": "string", "unit": "C|F"}}\n'
    '{"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}}\n'
    '{"tool": "convert",  "args": {"value": "number", "from_unit": "string", "to_unit": "string"}}\n'
    '{"tool": "currency", "args": {"amount": "number", "from": "ISO3", "to": "ISO3"}}\n'
    '{"tool": "sql",      "args": {"query": "string"}}\n\n'
    "Rules:\n"
    "- Emit EXACTLY ONE <tool_call>...</tool_call> block for tool requests, nothing else.\n"
    "- For calendar \"list\", title is optional and may be omitted.\n"
    "- For refusals, write a brief helpful plain-text reply.\n"
    "- In multi-turn conversations, resolve pronouns/references from prior turns.\n"
)


# --------------------------------------------------------------------------- #
# Model loader — cached at module level. First call pays the load cost;      #
# subsequent calls (and the 20 grading examples) reuse the same Llama.      #
# --------------------------------------------------------------------------- #

_MODEL: Optional[Llama] = None
_MODEL_PATH_ENV = "POCKET_AGENT_GGUF"
_DEFAULT_GGUF = "artifacts\\pocket-agent.gguf"


def _get_model() -> Llama:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_path = os.environ.get(_MODEL_PATH_ENV, str(_DEFAULT_GGUF))
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"GGUF not found at {model_path}. Set {_MODEL_PATH_ENV} or place "
            f"the file at {_DEFAULT_GGUF}."
        )

    # Tuning for the Colab CPU runtime latency gate (≤200 ms/turn):
    #   - n_ctx=2048: enough for system + 3 turns, avoids wasted KV allocation
    #   - n_threads = physical cores (Colab CPU runtime is typically 2)
    #   - n_batch=512: fast prompt-prefill on short prompts
    #   - use_mmap=True: lets the OS page-cache the weights across calls
    _MODEL = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=max(1, (os.cpu_count() or 2)),
        n_batch=512,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
        chat_format="chatml",   # Qwen3 uses ChatML; matches training
    )
    return _MODEL


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def _normalize_history(history: List[Dict]) -> List[Dict]:
    """
    Accept either OpenAI-style messages ({"role": "...", "content": "..."})
    or tuples, and strip any pre-existing system turn — we always prepend
    our own canonical SYSTEM_PROMPT so the model sees the format it trained on.
    """
    clean: List[Dict] = []
    for turn in history or []:
        if isinstance(turn, dict) and "role" in turn and "content" in turn:
            if turn["role"] == "system":
                continue  # we inject our own
            clean.append({"role": turn["role"], "content": str(turn["content"])})
    return clean


def _postprocess(text: str) -> str:
    """
    The model was trained to emit EXACTLY ONE <tool_call> block or plain text.
    In practice, small models occasionally:
      - add whitespace or newlines around the block
      - emit stray text after </tool_call>
      - emit the block at the very end of a longer reply (rare at inference)
    We strip to the first complete tool_call if present, else return trimmed text.
    """
    text = text.strip()
    start = text.find("<tool_call>")
    end = text.find("</tool_call>")
    if start != -1 and end != -1 and end > start:
        return text[start : end + len("</tool_call>")]
    return text


def run(prompt: str, history: List[Dict]) -> str:
    """
    Grader contract. `history` is prior turns (user + assistant alternating);
    `prompt` is the current user message. Returns the assistant's reply as a
    raw string.
    """
    model = _get_model()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_normalize_history(history))
    messages.append({"role": "user", "content": prompt})

    # Deterministic decoding for tool-call accuracy. Tool calls are a
    # structured-output problem — any sampling noise hurts arg fidelity.
    out = model.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        stop=["<|im_end|>"],
    )
    text = out["choices"][0]["message"]["content"] or ""
    return _postprocess(text)


# --------------------------------------------------------------------------- #
# Manual smoke test — `python inference.py` from the repo root.              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    tests = [
        ("What's the weather in Karachi in Celsius?", []),
        ("Convert 5 km to miles", []),
        (
            "Now convert that to meters",
            [
                {"role": "user", "content": "Convert 5 km to miles"},
                {
                    "role": "assistant",
                    "content": (
                        '<tool_call>{"tool":"convert","args":'
                        '{"value":5,"from_unit":"km","to_unit":"miles"}}</tool_call>'
                    ),
                },
            ],
        ),
        ("What's your favorite color?", []),
    ]
    for p, h in tests:
        print(f"\nUSER: {p}")
        print(f"BOT : {run(p, h)}")
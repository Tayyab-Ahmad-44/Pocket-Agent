# Pocket-Agent

Fine-tuned [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
for structured tool calling on-device. Trained with Unsloth (QLoRA), quantized
to GGUF, served via `llama-cpp-python` with zero network dependencies.

## TL;DR

```bash
pip install -r requirements.txt
# training + quantize runs in pocket-agent.ipynb (on Kaggle/Colab)
python demo.py                      # Gradio demo
```

---

## Design decisions

**Base model: `unsloth/Qwen3-0.6B`.** Unsloth mirror of the Qwen3 weights —
same params, but the Unsloth loader recognizes it as pre-registered and skips
the patching step.

**QLoRA over full fine-tune.** On a free T4 (15 GB VRAM), QLoRA on a 0.6B model
uses ~3-4 GB peak. Full fine-tune would waste VRAM budget.

**LoRA rank 32 with alpha 64.** Rank 16 is standard for small models but
tool-calling needs precision on structured output; 32 gave measurably better
arg-fidelity in preliminary tests. Alpha = 2×r is a conservative default.

**`/no_think` in system prompt.** Qwen3 has a built-in thinking chain that fires
by default. For tool-calling we don't need it — it burns tokens and adds latency.
Appending `/no_think` to the system prompt disables it at inference time.

**Response-only loss masking via `train_on_responses_only`.** The system prompt
is ~400 tokens and identical across every example. Training on all tokens means
the model spends most of its gradient steps memorizing the (already-correct)
system prompt instead of learning the tool calls. Masking the prompt halves
effective training time and improves args-correctness on adversarial examples.

**Markers for `train_on_responses_only`:**

- instruction: `"<|im_start|>user\n"`
- response: `"<|im_start|>assistant\n"`

These match what Qwen3's ChatML template emits. The notebook prints a sanity
check of the masked tokens on startup — if you see 0% supervised tokens, the
markers are wrong.

**Deterministic decoding at inference time (`temperature=0.0`).** Tool calls
are structured output — any sampling noise hurts arg fidelity.

**GGUF + `llama-cpp-python` for inference.** llama.cpp's CPU path is the
fastest option for sub-200ms on Colab CPU runtime and has no network deps —
passes the AST scan trivially.

---

## Repo layout

```text
stage 1/
├── inference.py              ← grader entry point  (def run(prompt, history))
├── requirements.txt
├── demo.py                   ← Gradio UI
├── dataset/
│   ├── tool_schemas.json
│   └── qwen_cleaned.jsonl
├── training/
│   ├── pocket-agent.ipynb
└── artifacts/
    ├── lora_adapter/         ← LoRA weights (~50 MB)
    ├── merged_16bit/         ← merged fp16 model (intermediate, large)
    └── pocket-agent.gguf     ← FINAL quantized model
```

## Error analysis

After running inference on `public_test.jsonl`, categorize failures:

- **`wrong_tool`** — model picked weather when the gold was convert, etc.
  Fix: generate more paraphrases for that tool.
- **`tool_ok_args_wrong`** — right tool, wrong args. Common cause: unit/ISO
  hallucination. Fix: add adversarial variants ("USDs", "dollars" → USD;
  "celsius"/"°C" → "C").
- **`malformed_json`** — model dropped the closing `}`. Check that
  `max_tokens` in `inference.py` is high enough (256 is plenty for all 5 tools).
- **`refused_but_called`** — model emitted a tool call when gold was refusal.
  Fix: train set needs more refusal examples.
- **`should_have_called`** — model refused when it should have called. Often
  code-switched user turn. Fix: add code-switched paraphrases to the train set.

"""
llm.py — Unified LLM Loader
Supports Unsloth fast-path (recommended) and plain HuggingFace as fallback.
Swap models by changing a single config field.
"""

from __future__ import annotations

import os
import torch
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Model config dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # ── Core ─────────────────────────────────────────────
    model_name: str         = "unsloth/gpt-oss-20b"
    max_seq_length: int     = 4096
    dtype: Optional[str]    = None          # None = auto-detect
    load_in_4bit: bool      = False         # BitsAndBytes 4-bit
    full_finetuning: bool   = False

    # ── LoRA (ignored when full_finetuning=True) ──────────
    lora_r: int             = 64
    lora_alpha: int         = 64
    lora_dropout: float     = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ── Misc ─────────────────────────────────────────────
    hf_token: Optional[str] = None          # for gated models
    use_gradient_checkpointing: str = "unsloth"


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(cfg: ModelConfig):
    """
    Load a model + tokenizer via Unsloth (preferred) or plain HF.
    Returns (model, tokenizer).
    """
    _print_load_banner(cfg)

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name              = cfg.model_name,
            max_seq_length          = cfg.max_seq_length,
            dtype                   = cfg.dtype,
            load_in_4bit            = cfg.load_in_4bit,
            full_finetuning         = cfg.full_finetuning,
            token                   = cfg.hf_token or os.getenv("HF_TOKEN"),
        )
        console.print("[bright_green]✓  Unsloth fast-path loaded.[/]")
    except ImportError:
        console.print("[yellow]⚠  Unsloth not found — falling back to HuggingFace.[/]")
        model, tokenizer = _load_hf(cfg)

    if not cfg.full_finetuning:
        model = _apply_lora(model, cfg)

    _print_model_info(model, tokenizer, cfg)
    return model, tokenizer


def _load_hf(cfg: ModelConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    bnb_config = None
    if cfg.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    dtype = getattr(torch, cfg.dtype) if isinstance(cfg.dtype, str) else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map="auto",
        token=cfg.hf_token or os.getenv("HF_TOKEN"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        token=cfg.hf_token or os.getenv("HF_TOKEN"),
    )
    return model, tokenizer


def _apply_lora(model, cfg: ModelConfig):
    try:
        from unsloth import FastLanguageModel
        model = FastLanguageModel.get_peft_model(
            model,
            r                       = cfg.lora_r,
            lora_alpha              = cfg.lora_alpha,
            lora_dropout            = cfg.lora_dropout,
            target_modules          = cfg.lora_target_modules,
            use_gradient_checkpointing = cfg.use_gradient_checkpointing,
            random_state            = 42,
            use_rslora              = False,
        )
        console.print(f"[bright_cyan]✓  LoRA applied  (r={cfg.lora_r}, α={cfg.lora_alpha})[/]")
    except ImportError:
        from peft import LoraConfig, get_peft_model, TaskType
        lora_config = LoraConfig(
            r              = cfg.lora_r,
            lora_alpha     = cfg.lora_alpha,
            lora_dropout   = cfg.lora_dropout,
            target_modules = cfg.lora_target_modules,
            task_type      = TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        console.print(f"[bright_cyan]✓  PEFT LoRA applied  (r={cfg.lora_r})[/]")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def generate_solution(
    model,
    tokenizer,
    problem: str,
    max_new_tokens: int = 2048,
    reasoning_effort: str = "high",   # low | medium | high
    stream: bool = True,
) -> str:
    """Run inference on a single olympiad problem."""
    from transformers import TextStreamer

    messages = [{"role": "user", "content": problem}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors        = "pt",
        return_dict           = True,
        reasoning_effort      = reasoning_effort,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    streamer = TextStreamer(tokenizer) if stream else None

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            streamer       = streamer,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def enable_fast_inference(model):
    """Switch model to fast inference mode (Unsloth 2× speedup)."""
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
        console.print("[bright_green]✓  Fast inference mode enabled.[/]")
    except Exception:
        model.eval()
        console.print("[yellow]⚠  Using standard eval() mode.[/]")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────
def _print_load_banner(cfg: ModelConfig):
    console.print(Panel(
        f"[bold bright_white]Loading:[/] [bright_cyan]{cfg.model_name}[/]\n"
        f"[dim]max_seq_len={cfg.max_seq_length}  "
        f"4bit={cfg.load_in_4bit}  "
        f"full_ft={cfg.full_finetuning}  "
        f"LoRA r={cfg.lora_r}[/]",
        title="[bold]🧠  Model Initialization[/]",
        border_style="bright_blue",
    ))


def _print_model_info(model, tokenizer, cfg: ModelConfig):
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = trainable / total * 100 if total > 0 else 0

    t = Table(box=box.SIMPLE, show_header=False, border_style="dim")
    t.add_column("Key",   style="dim")
    t.add_column("Value", style="bold bright_white")
    t.add_row("Total params",     f"{total/1e9:.2f} B")
    t.add_row("Trainable params", f"{trainable/1e6:.2f} M  ({pct:.3f}%)")
    t.add_row("Vocab size",       str(tokenizer.vocab_size))
    t.add_row("Device",           str(next(model.parameters()).device))
    t.add_row("Dtype",            str(next(model.parameters()).dtype))
    console.print(t)

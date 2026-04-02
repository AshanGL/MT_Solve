"""
train.py — Fine-Tuning Engine
Supports:
  • SFT  (Supervised Fine-Tuning)  via TRL SFTTrainer
  • GRPO (Group Relative Policy Optimization) reward training
Plugs into the MRVM weight-updater layer of the architecture.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    # ── Output ────────────────────────────────────────────
    output_dir: str          = "./checkpoints"
    run_name: str            = "olympiad-solver"

    # ── Core hyper-params ─────────────────────────────────
    num_train_epochs: int    = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size:  int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float     = 2e-4
    warmup_ratio: float      = 0.05
    lr_scheduler_type: str   = "cosine"
    weight_decay: float      = 0.01
    max_grad_norm: float     = 1.0

    # ── Sequence ──────────────────────────────────────────
    max_seq_length: int      = 4096
    packing: bool            = True          # faster training

    # ── Evaluation & saving ───────────────────────────────
    eval_strategy: str       = "steps"
    eval_steps: int          = 50
    save_strategy: str       = "steps"
    save_steps: int          = 100
    save_total_limit: int    = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # ── Logging ───────────────────────────────────────────
    logging_steps: int       = 10
    report_to: str           = "none"       # "wandb" | "tensorboard" | "none"

    # ── Precision ─────────────────────────────────────────
    bf16: bool               = True
    fp16: bool               = False

    # ── GRPO-specific ─────────────────────────────────────
    use_grpo: bool           = False
    grpo_num_generations: int = 4
    grpo_max_prompt_length: int = 1024
    grpo_max_completion_length: int = 3072

    # ── Misc ─────────────────────────────────────────────
    dataloader_num_workers: int = 2
    seed: int                = 42


# ─────────────────────────────────────────────────────────────────────────────
# SFT Training
# ─────────────────────────────────────────────────────────────────────────────
def train_sft(model, tokenizer, train_dataset, eval_dataset, cfg: TrainingConfig):
    """Supervised fine-tuning with TRL SFTTrainer."""
    from trl import SFTTrainer, SFTConfig

    _print_training_banner("SFT", cfg, len(train_dataset), len(eval_dataset))

    sft_cfg = SFTConfig(
        output_dir                  = cfg.output_dir,
        run_name                    = cfg.run_name,
        num_train_epochs            = cfg.num_train_epochs,
        per_device_train_batch_size = cfg.per_device_train_batch_size,
        per_device_eval_batch_size  = cfg.per_device_eval_batch_size,
        gradient_accumulation_steps = cfg.gradient_accumulation_steps,
        learning_rate               = cfg.learning_rate,
        warmup_ratio                = cfg.warmup_ratio,
        lr_scheduler_type           = cfg.lr_scheduler_type,
        weight_decay                = cfg.weight_decay,
        max_grad_norm               = cfg.max_grad_norm,
        max_seq_length              = cfg.max_seq_length,
        packing                     = cfg.packing,
        eval_strategy               = cfg.eval_strategy,
        eval_steps                  = cfg.eval_steps,
        save_strategy               = cfg.save_strategy,
        save_steps                  = cfg.save_steps,
        save_total_limit            = cfg.save_total_limit,
        load_best_model_at_end      = cfg.load_best_model_at_end,
        metric_for_best_model       = cfg.metric_for_best_model,
        logging_steps               = cfg.logging_steps,
        report_to                   = cfg.report_to,
        bf16                        = cfg.bf16,
        fp16                        = cfg.fp16,
        dataloader_num_workers      = cfg.dataloader_num_workers,
        seed                        = cfg.seed,
        dataset_text_field          = "text",
    )

    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_dataset,
        eval_dataset    = eval_dataset,
        args            = sft_cfg,
    )

    console.print("[bright_yellow]🚀  Starting SFT training…[/]")
    train_result = trainer.train()
    _print_train_result(train_result)
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# GRPO Reward Functions
# ─────────────────────────────────────────────────────────────────────────────
def _reward_exact_match(prompts, completions, answers, **kwargs) -> list[float]:
    """Reward +1.0 when the model's final number matches the expected answer."""
    rewards = []
    for completion, answer in zip(completions, answers):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        # Extract last number-like token from response
        nums = re.findall(r"-?\d+(?:[./]\d+)?", text.replace(",", ""))
        pred = nums[-1] if nums else ""
        rewards.append(1.0 if pred.strip() == str(answer).strip() else 0.0)
    return rewards


def _reward_format(prompts, completions, **kwargs) -> list[float]:
    """
    Soft reward for structured reasoning:
    +0.3  has numbered steps or 'Step N:' labels
    +0.3  mentions 'therefore', 'thus', 'hence', '∴'
    +0.4  ends with explicit 'answer' declaration
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0
        if re.search(r"(step \d+|^\d+\.)", text, re.I | re.M): score += 0.3
        if re.search(r"\b(therefore|thus|hence|∴)\b", text, re.I):  score += 0.3
        if re.search(r"\b(answer|final answer)\b.*[:=]", text, re.I): score += 0.4
        rewards.append(score)
    return rewards


def _reward_length_penalty(prompts, completions, **kwargs) -> list[float]:
    """Mild penalty for very short completions (< 100 chars)."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        rewards.append(0.0 if len(text) < 100 else 0.1)
    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# GRPO Training (MRVM weight updater)
# ─────────────────────────────────────────────────────────────────────────────
def train_grpo(model, tokenizer, train_dataset, eval_dataset, cfg: TrainingConfig):
    """
    GRPO fine-tuning — the MRVM weight-updater stage.
    Combines exact-match + format + length rewards.
    """
    from trl import GRPOTrainer, GRPOConfig

    _print_training_banner("GRPO", cfg, len(train_dataset), len(eval_dataset))

    grpo_cfg = GRPOConfig(
        output_dir                  = cfg.output_dir,
        run_name                    = cfg.run_name + "-grpo",
        num_train_epochs            = cfg.num_train_epochs,
        per_device_train_batch_size = cfg.per_device_train_batch_size,
        gradient_accumulation_steps = cfg.gradient_accumulation_steps,
        learning_rate               = cfg.learning_rate,
        warmup_ratio                = cfg.warmup_ratio,
        lr_scheduler_type           = cfg.lr_scheduler_type,
        weight_decay                = cfg.weight_decay,
        logging_steps               = cfg.logging_steps,
        save_steps                  = cfg.save_steps,
        bf16                        = cfg.bf16,
        fp16                        = cfg.fp16,
        num_generations             = cfg.grpo_num_generations,
        max_prompt_length           = cfg.grpo_max_prompt_length,
        max_completion_length       = cfg.grpo_max_completion_length,
        report_to                   = cfg.report_to,
        seed                        = cfg.seed,
    )

    trainer = GRPOTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_dataset,
        reward_funcs    = [
            _reward_exact_match,
            _reward_format,
            _reward_length_penalty,
        ],
        args            = grpo_cfg,
    )

    console.print("[bright_yellow]🚀  Starting GRPO training…[/]")
    train_result = trainer.train()
    _print_train_result(train_result)
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────
def _print_training_banner(mode: str, cfg: TrainingConfig, n_train: int, n_eval: int):
    mode_color = "bright_cyan" if mode == "SFT" else "bright_magenta"
    effective_batch = (
        cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    )
    steps_per_epoch = math.ceil(n_train / effective_batch)
    total_steps = steps_per_epoch * cfg.num_train_epochs

    console.print(Panel(
        f"[bold {mode_color}]Mode:[/] {mode}\n"
        f"[dim]Train={n_train}  Eval={n_eval}  "
        f"Epochs={cfg.num_train_epochs}  "
        f"EffBatch={effective_batch}  "
        f"LR={cfg.learning_rate:.0e}  "
        f"~{total_steps} steps[/]",
        title=f"[bold]🏋  Training Config — {mode}[/]",
        border_style=mode_color,
    ))


def _print_train_result(result):
    t = Table(box=box.SIMPLE, show_header=False, border_style="dim")
    t.add_column("Metric", style="dim")
    t.add_column("Value",  style="bold bright_green")
    metrics = result.metrics if hasattr(result, "metrics") else {}
    for k, v in metrics.items():
        t.add_row(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else str(v))
    if metrics:
        console.print(Panel(t, title="[bold]✅  Training Complete[/]", border_style="bright_green"))
    else:
        console.print("[bright_green]✅  Training complete.[/]")

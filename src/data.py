"""
data.py — OlymMATH Dataset Loader & Preprocessor
Handles loading, splitting, and formatting the OlymMATH JSONL dataset.
"""

import json
import random
from pathlib import Path
from typing import Optional
from collections import Counter

from datasets import Dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Chat template for reasoning-capable models (Unsloth / OpenAI-compatible)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an elite olympiad mathematics solver. "
    "Think step-by-step with rigorous logic. "
    "Show all intermediate work clearly. "
    "State your final answer explicitly at the end."
)

def format_example(problem: str, answer: str) -> list[dict]:
    """Format a single problem into a chat-style conversation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem},
        {"role": "assistant", "content": f"**Solution:**\n\n{answer}"},
    ]


def apply_chat_template(tokenizer, example: dict) -> dict:
    """Apply tokenizer chat template to a formatted example."""
    messages = format_example(example["problem"], example["answer"])
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────
def load_jsonl(path: str | Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_dataset_split(
    jsonl_path: str | Path,
    train_pct: float = 0.80,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> tuple[Dataset, Dataset]:
    """
    Load OlymMATH JSONL, shuffle, split into train / eval.

    Args:
        jsonl_path  : Path to the .jsonl file.
        train_pct   : Fraction of data used for training  (0.0 – 1.0).
        seed        : Random seed for reproducibility.
        max_samples : Cap total samples (useful for quick debugging runs).

    Returns:
        (train_dataset, eval_dataset)
    """
    records = load_jsonl(jsonl_path)

    random.seed(seed)
    random.shuffle(records)

    if max_samples:
        records = records[:max_samples]

    split_idx = int(len(records) * train_pct)
    train_records = records[:split_idx]
    eval_records  = records[split_idx:]

    train_ds = Dataset.from_list(train_records)
    eval_ds  = Dataset.from_list(eval_records)

    _print_dataset_summary(records, train_records, eval_records)
    return train_ds, eval_ds


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────
def _print_dataset_summary(all_rec, train_rec, eval_rec):
    subjects = Counter(r.get("subject", "Unknown") for r in all_rec)
    subj_colors = {
        "Algebra":       "bright_cyan",
        "Geometry":      "bright_green",
        "Combinatorics": "bright_yellow",
        "Number Theory": "bright_magenta",
    }

    table = Table(
        title="📊  OlymMATH Dataset Summary",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold bright_white on dark_blue",
        show_footer=True,
    )
    table.add_column("Split",   style="bold",  footer="Total")
    table.add_column("Samples", justify="right", footer=str(len(all_rec)))
    table.add_column("% of Data", justify="right", footer="100%")

    table.add_row("Train", str(len(train_rec)), f"{len(train_rec)/len(all_rec)*100:.1f}%", style="bright_green")
    table.add_row("Eval",  str(len(eval_rec)),  f"{len(eval_rec)/len(all_rec)*100:.1f}%",  style="bright_yellow")
    console.print(table)

    subj_table = Table(
        title="🔬  Subject Distribution",
        box=box.SIMPLE_HEAD,
        border_style="dim",
        header_style="bold",
    )
    subj_table.add_column("Subject")
    subj_table.add_column("Count", justify="right")
    subj_table.add_column("Bar")

    total = len(all_rec)
    for subj, cnt in subjects.most_common():
        color = subj_colors.get(subj, "white")
        bar_len = int(cnt / total * 30)
        bar = f"[{color}]{'█' * bar_len}[/]"
        subj_table.add_row(f"[{color}]{subj}[/]", str(cnt), bar)

    console.print(subj_table)

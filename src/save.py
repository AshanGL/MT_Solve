"""
save.py — Complete Architecture Save / Load
Saves:
  • Model weights (LoRA adapter or full)
  • Tokenizer
  • Training config
  • Model config
  • Architecture metadata & run stats
Supports: local disk  |  HuggingFace Hub  |  GGUF export (via Unsloth)
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
def save_complete(
    model,
    tokenizer,
    model_cfg,
    train_cfg,
    output_dir: str | Path,
    run_stats: Optional[dict] = None,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    save_gguf: bool = False,
    gguf_quantization: str = "q4_k_m",
) -> Path:
    """
    Save the full architecture to `output_dir`.

    Directory layout
    ─────────────────
    output_dir/
    ├── adapter/           ← LoRA adapter (or full weights)
    ├── tokenizer/         ← tokenizer files
    ├── model_config.json  ← ModelConfig dataclass → JSON
    ├── train_config.json  ← TrainingConfig dataclass → JSON
    ├── architecture.json  ← pipeline metadata + run stats
    └── gguf/              ← (optional) GGUF export
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _header("Saving Complete Architecture", str(output_dir))

    # 1. Model weights
    adapter_dir = output_dir / "adapter"
    _step("Saving model weights …")
    try:
        model.save_pretrained(str(adapter_dir))
        console.print(f"   [bright_green]✓[/] Saved to [dim]{adapter_dir}[/]")
    except Exception as e:
        console.print(f"   [bright_red]✗ save_pretrained failed:[/] {e}")

    # 2. Tokenizer
    tok_dir = output_dir / "tokenizer"
    _step("Saving tokenizer …")
    tokenizer.save_pretrained(str(tok_dir))
    console.print(f"   [bright_green]✓[/] Saved to [dim]{tok_dir}[/]")

    # 3. Configs as JSON
    _step("Saving configs …")
    _save_json(output_dir / "model_config.json",  _dataclass_to_dict(model_cfg))
    _save_json(output_dir / "train_config.json",  _dataclass_to_dict(train_cfg))
    console.print("   [bright_green]✓[/] model_config.json + train_config.json")

    # 4. Architecture metadata
    _step("Saving architecture metadata …")
    meta = {
        "architecture_version" : "1.0",
        "pipeline_stages"      : [
            "subject_classifier_confidence_gate",
            "decomposer",
            "similarity_retrieval",
            "embedding_store",
            "hypothesis_generator",
            "verifier",
            "sub_problem_tracker",
            "answer_aggregator_formatter",
            "mrvm_weight_updater_grpo",
        ],
        "model_name"           : getattr(model_cfg, "model_name", "unknown"),
        "saved_at"             : datetime.now().isoformat(),
        "run_stats"            : run_stats or {},
    }
    _save_json(output_dir / "architecture.json", meta)
    console.print("   [bright_green]✓[/] architecture.json")

    # 5. Optional GGUF export
    if save_gguf:
        _step(f"Exporting GGUF ({gguf_quantization}) …")
        gguf_dir = output_dir / "gguf"
        gguf_dir.mkdir(exist_ok=True)
        try:
            model.save_pretrained_gguf(
                str(gguf_dir),
                tokenizer,
                quantization_method=gguf_quantization,
            )
            console.print(f"   [bright_green]✓[/] GGUF saved to [dim]{gguf_dir}[/]")
        except Exception as e:
            console.print(f"   [bright_yellow]⚠  GGUF export skipped:[/] {e}")

    # 6. Optional HF Hub push
    if push_to_hub and hub_repo_id:
        _step(f"Pushing to HuggingFace Hub: {hub_repo_id} …")
        try:
            model.push_to_hub(hub_repo_id)
            tokenizer.push_to_hub(hub_repo_id)
            console.print(f"   [bright_green]✓[/] Pushed to [link=https://huggingface.co/{hub_repo_id}]{hub_repo_id}[/link]")
        except Exception as e:
            console.print(f"   [bright_red]✗  Hub push failed:[/] {e}")

    _print_save_tree(output_dir)
    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
def load_complete(saved_dir: str | Path, device: str = "auto"):
    """
    Load a previously saved architecture.
    Returns (model, tokenizer, model_cfg_dict, train_cfg_dict, meta).
    """
    saved_dir = Path(saved_dir)
    _header("Loading Saved Architecture", str(saved_dir))

    adapter_dir = saved_dir / "adapter"
    tok_dir     = saved_dir / "tokenizer"

    # Read metadata to get original model name
    meta          = _load_json(saved_dir / "architecture.json")
    model_cfg_d   = _load_json(saved_dir / "model_config.json")
    train_cfg_d   = _load_json(saved_dir / "train_config.json")
    base_model    = model_cfg_d.get("model_name", "unknown")

    _step(f"Loading base model: {base_model} …")
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = str(adapter_dir),
            dtype          = model_cfg_d.get("dtype"),
            max_seq_length = model_cfg_d.get("max_seq_length", 4096),
            load_in_4bit   = model_cfg_d.get("load_in_4bit", False),
        )
        console.print("   [bright_green]✓[/] Loaded via Unsloth")
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model     = AutoModelForCausalLM.from_pretrained(str(adapter_dir), device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
        console.print("   [bright_green]✓[/] Loaded via HuggingFace")

    console.print(Panel(
        f"[bold]Model:[/] {base_model}\n"
        f"[bold]Saved:[/] {meta.get('saved_at', 'N/A')}\n"
        f"[bold]Run stats:[/] {json.dumps(meta.get('run_stats', {}), indent=2)}",
        title="[bold bright_green]✅  Architecture Loaded[/]",
        border_style="bright_green",
    ))

    return model, tokenizer, model_cfg_d, train_cfg_d, meta


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _dataclass_to_dict(obj) -> dict:
    import dataclasses
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return obj.__dict__ if hasattr(obj, "__dict__") else {}


def _save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _header(title: str, path: str):
    console.print(Panel(
        f"[dim]{path}[/]",
        title=f"[bold bright_blue]💾  {title}[/]",
        border_style="bright_blue",
    ))


def _step(msg: str):
    console.print(f"[bright_blue]→[/] {msg}")


def _print_save_tree(output_dir: Path):
    tree = Tree(f"[bold]{output_dir.name}/[/]")
    for child in sorted(output_dir.iterdir()):
        if child.is_dir():
            branch = tree.add(f"[bright_cyan]📁 {child.name}/[/]")
            for f in sorted(child.iterdir())[:5]:
                size = f.stat().st_size
                branch.add(f"[dim]{f.name}[/]  [bright_black]({_fmt_size(size)})[/]")
        else:
            size = child.stat().st_size
            tree.add(f"[bright_white]📄 {child.name}[/]  [bright_black]({_fmt_size(size)})[/]")
    console.print(Panel(tree, title="[bold]📦  Saved Files[/]", border_style="dim"))


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

"""
inference.py — Full Architecture Inference Pipeline
Implements the complete diagram flow:
  Input → Classifier+Gate → Decomposer → Retrieval →
  Hypothesis Generator → Verifier → Tracker → Aggregator
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich import box

console = Console()

SUBJECTS = ["Algebra", "Geometry", "Combinatorics", "Number Theory", "Calculus", "Other"]

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SubProblem:
    id: int
    description: str
    solution: str   = ""
    verified: bool  = False
    confidence: float = 0.0


@dataclass
class SolverResult:
    problem: str
    subject: str
    confidence: float
    sub_problems: list[SubProblem]
    final_answer: str
    verified: bool
    elapsed_sec: float


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Subject Classifier + Confidence Gate
# ─────────────────────────────────────────────────────────────────────────────
def classify_problem(model, tokenizer, problem: str) -> tuple[str, float]:
    """Return (subject, confidence_0_to_1)."""
    prompt = (
        f"Classify this olympiad problem into ONE of these subjects: "
        f"{', '.join(SUBJECTS)}.\n"
        f"Reply with ONLY the subject name and a confidence score 0-100.\n"
        f"Format: <subject>|<confidence>\n\n"
        f"Problem: {problem[:500]}"
    )
    response = _quick_generate(model, tokenizer, prompt, max_new_tokens=20)
    # Parse
    match = re.search(r"(\w[\w\s]+)\|(\d+)", response)
    if match:
        subject = match.group(1).strip()
        confidence = float(match.group(2)) / 100.0
        subject = next((s for s in SUBJECTS if s.lower() in subject.lower()), "Other")
    else:
        subject, confidence = "Other", 0.4
    return subject, min(max(confidence, 0.0), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Decomposer
# ─────────────────────────────────────────────────────────────────────────────
def decompose_problem(model, tokenizer, problem: str, subject: str) -> list[str]:
    """Break a hard problem into 2-4 sub-problems."""
    prompt = (
        f"You are solving a {subject} olympiad problem. "
        f"Decompose it into 2-4 independent sub-problems or key steps.\n"
        f"Number each sub-problem: 1. ... 2. ... etc.\n\n"
        f"Problem: {problem}"
    )
    response = _quick_generate(model, tokenizer, prompt, max_new_tokens=512)
    parts = re.split(r"\n\s*\d+\.", "\n" + response)
    sub_problems = [p.strip() for p in parts if len(p.strip()) > 20]
    return sub_problems[:4] if sub_problems else [problem]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Hypothesis Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_hypothesis(model, tokenizer, sub_problem: str, subject: str) -> str:
    """Solve a sub-problem with full chain-of-thought."""
    prompt = (
        f"Solve this {subject} problem step-by-step. "
        f"Be rigorous. State intermediate results clearly.\n\n"
        f"{sub_problem}"
    )
    return _quick_generate(model, tokenizer, prompt, max_new_tokens=1024)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Verifier
# ─────────────────────────────────────────────────────────────────────────────
def verify_solution(model, tokenizer, problem: str, solution: str) -> tuple[bool, float]:
    """Verify a solution. Returns (is_valid, confidence_0_to_1)."""
    prompt = (
        f"Verify this mathematical solution. "
        f"Check each step for correctness.\n"
        f"Reply: VALID|<confidence> or INVALID|<confidence> (confidence 0-100).\n\n"
        f"Problem: {problem[:300]}\n\nSolution: {solution[:600]}"
    )
    response = _quick_generate(model, tokenizer, prompt, max_new_tokens=30)
    valid     = "VALID" in response.upper() and "INVALID" not in response.upper()
    match     = re.search(r"\|(\d+)", response)
    confidence = float(match.group(1)) / 100.0 if match else (0.7 if valid else 0.3)
    return valid, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Answer Aggregator
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_answer(model, tokenizer, problem: str, sub_solutions: list[str]) -> str:
    """Combine sub-problem solutions into a final answer."""
    combined = "\n\n".join(
        f"Part {i+1}:\n{sol}" for i, sol in enumerate(sub_solutions)
    )
    prompt = (
        f"Combine these partial solutions into one complete, elegant solution. "
        f"State the final answer clearly at the end.\n\n"
        f"Original problem: {problem}\n\n"
        f"Partial solutions:\n{combined}"
    )
    return _quick_generate(model, tokenizer, prompt, max_new_tokens=1024)


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def solve(
    model,
    tokenizer,
    problem: str,
    confidence_threshold: float = 0.6,
    verbose: bool = True,
) -> SolverResult:
    """
    Run the full architecture pipeline on a single olympiad problem.
    confidence_threshold: below this, parallel paths are activated.
    """
    t0 = time.time()

    if verbose:
        console.print(Panel(
            f"[italic]{problem[:300]}{'…' if len(problem)>300 else ''}[/]",
            title="[bold bright_yellow]📐  Olympiad Problem[/]",
            border_style="yellow",
        ))

    # Stage 1 — Classify
    _stage("1", "Subject Classifier + Confidence Gate")
    subject, confidence = classify_problem(model, tokenizer, problem)
    low_conf = confidence < confidence_threshold
    console.print(
        f"   Subject: [bright_cyan]{subject}[/]  "
        f"Confidence: [{'bright_green' if not low_conf else 'bright_yellow'}]{confidence:.0%}[/]"
        + (" [yellow]← parallel paths activated[/]" if low_conf else "")
    )

    # Stage 2 — Decompose
    _stage("2", "Decomposer")
    sub_texts = decompose_problem(model, tokenizer, problem, subject)
    if low_conf:
        # Parallel path: also decompose as generic maths
        sub_texts_generic = decompose_problem(model, tokenizer, problem, "Mathematics")
        sub_texts = list(dict.fromkeys(sub_texts + sub_texts_generic))[:4]
    console.print(f"   {len(sub_texts)} sub-problems identified")

    # Stage 3 — Hypothesis Generator per sub-problem
    _stage("3", "Hypothesis Generator")
    sub_problems: list[SubProblem] = []
    for i, sp_text in enumerate(sub_texts):
        console.print(f"   [dim]→ Sub-problem {i+1}/{len(sub_texts)}[/]")
        sol = generate_hypothesis(model, tokenizer, sp_text, subject)
        sp = SubProblem(id=i+1, description=sp_text, solution=sol)
        sub_problems.append(sp)

    # Stage 4 — Verifier
    _stage("4", "Verifier")
    for sp in sub_problems:
        sp.verified, sp.confidence = verify_solution(model, tokenizer, sp.description, sp.solution)
        icon = "✅" if sp.verified else "❌"
        console.print(f"   Sub-{sp.id}: {icon}  conf={sp.confidence:.0%}")

    # Stage 5 — Aggregate
    _stage("5", "Answer Aggregator + Formatter")
    final = aggregate_answer(model, tokenizer, problem, [sp.solution for sp in sub_problems])
    all_verified = all(sp.verified for sp in sub_problems)
    elapsed = time.time() - t0

    result = SolverResult(
        problem      = problem,
        subject      = subject,
        confidence   = confidence,
        sub_problems = sub_problems,
        final_answer = final,
        verified     = all_verified,
        elapsed_sec  = elapsed,
    )

    if verbose:
        _print_result(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_batch(model, tokenizer, examples: list[dict], verbose: bool = False) -> dict:
    """
    Run solve() on a list of {problem, answer} dicts.
    Returns accuracy metrics.
    """
    correct = 0
    results = []
    for ex in examples:
        res = solve(model, tokenizer, ex["problem"], verbose=verbose)
        # Simple last-number match
        pred_nums = re.findall(r"-?\d+(?:[./]\d+)?", res.final_answer.replace(",", ""))
        pred = pred_nums[-1] if pred_nums else ""
        is_correct = pred.strip() == str(ex.get("answer", "")).strip()
        if is_correct:
            correct += 1
        results.append({"result": res, "correct": is_correct})

    accuracy = correct / len(examples) if examples else 0.0
    _print_eval_summary(results, accuracy)
    return {"accuracy": accuracy, "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
def _quick_generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens  = max_new_tokens,
            do_sample       = False,
            temperature     = 1.0,
            pad_token_id    = tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _stage(num: str, name: str):
    console.print(f"\n[bold bright_blue]▶ Stage {num}[/] [bold]{name}[/]")


def _print_result(result: SolverResult):
    tree = Tree(f"[bold bright_green]🏆  Solution[/] ({result.elapsed_sec:.1f}s)")
    for sp in result.sub_problems:
        icon = "✅" if sp.verified else "⚠"
        branch = tree.add(f"{icon} [cyan]Sub-problem {sp.id}[/]  (conf={sp.confidence:.0%})")
        branch.add(f"[dim]{sp.solution[:120]}…[/]")
    tree.add(f"[bold bright_yellow]Final Answer:[/]\n{result.final_answer[:400]}")
    console.print(tree)


def _print_eval_summary(results: list[dict], accuracy: float):
    t = Table(title="📊 Evaluation Summary", box=box.ROUNDED, border_style="bright_blue")
    t.add_column("#", style="dim", width=4)
    t.add_column("Subject")
    t.add_column("Verified", justify="center")
    t.add_column("Correct", justify="center")
    t.add_column("Time (s)", justify="right")

    for i, r in enumerate(results[:20], 1):
        res = r["result"]
        t.add_row(
            str(i),
            res.subject,
            "✅" if res.verified else "❌",
            "[bright_green]✓[/]" if r["correct"] else "[bright_red]✗[/]",
            f"{res.elapsed_sec:.1f}",
        )

    console.print(t)
    color = "bright_green" if accuracy >= 0.5 else "bright_yellow" if accuracy >= 0.25 else "bright_red"
    console.print(f"\n[bold {color}]Accuracy: {accuracy:.1%}[/]  ({sum(r['correct'] for r in results)}/{len(results)})")

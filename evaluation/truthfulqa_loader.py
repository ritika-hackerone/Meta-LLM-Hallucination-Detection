"""
evaluation/truthfulqa_loader.py
────────────────────────────────
Loads the TruthfulQA benchmark dataset via HuggingFace `datasets`.

Dataset card: https://huggingface.co/datasets/truthful_qa
Paper:        https://arxiv.org/abs/2109.07958

https://huggingface.co/datasets/domenicrosati/TruthfulQA
The dataset has two configs:
  • "generation"  — open-ended answers + best/correct/incorrect answers
  • "multiple_choice" — MC format

We use "generation" and derive binary labels:
  CORRECT      → response matches a best/correct answer
  HALLUCINATED → response matches an incorrect answer

For evaluation purposes we sample from both correct and incorrect answers
to create a balanced labelled test set.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from datasets import load_dataset
from loguru import logger


def load_truthfulqa(
    n_samples:  int  = 817,
    seed:       int  = 42,
    config_name: str = "generation",
) -> List[dict]:
    """
    Load TruthfulQA and return a list of dicts with keys:
      question, response, true_label

    Parameters
    ----------
    n_samples  : total samples to return (balanced correct / hallucinated)
    seed       : random seed for reproducibility
    config_name: "generation" or "multiple_choice"
    """
    random.seed(seed)

    logger.info(f"Loading TruthfulQA ({config_name}) from HuggingFace…")
    # Dataset: https://huggingface.co/datasets/truthful_qa
    ds = load_dataset("truthful_qa", config_name, split="validation")
    

    logger.info(f"Loaded {len(ds)} TruthfulQA examples.")

    samples: List[dict] = []

    for row in ds:
        question = row["question"]

        if config_name == "generation":
            correct_answers   = row.get("correct_answers", [])
            incorrect_answers = row.get("incorrect_answers", [])

            # Add correct examples
            for ans in correct_answers[:1]:        # 1 correct per question
                samples.append({
                    "question":   question,
                    "response":   ans,
                    "true_label": "CORRECT",
                })

            # Add incorrect (hallucinated) examples
            for ans in incorrect_answers[:1]:      # 1 incorrect per question
                samples.append({
                    "question":   question,
                    "response":   ans,
                    "true_label": "HALLUCINATED",
                })

        elif config_name == "multiple_choice":
            mc1 = row.get("mc1_targets", {})
            choices = mc1.get("choices", [])
            labels  = mc1.get("labels",  [])

            for choice, label in zip(choices, labels):
                samples.append({
                    "question":   question,
                    "response":   choice,
                    "true_label": "CORRECT" if label == 1 else "HALLUCINATED",
                })

    random.shuffle(samples)
    selected = samples[:n_samples]

    correct_count = sum(1 for s in selected if s["true_label"] == "CORRECT")
    logger.info(
        f"Selected {len(selected)} samples "
        f"({correct_count} correct, {len(selected)-correct_count} hallucinated)"
    )
    return selected


def quick_sample(n: int = 10) -> List[dict]:
    """Convenience wrapper for quick testing."""
    return load_truthfulqa(n_samples=n)

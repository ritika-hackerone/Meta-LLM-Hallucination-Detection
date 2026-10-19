"""
pipeline/nli_classifier.py
──────────────────────────
Natural Language Inference (NLI) module.

Uses a cross-encoder NLI model (DeBERTa-v3-base by default) to classify
whether retrieved evidence ENTAILS, CONTRADICTS, or is NEUTRAL to a claim.

Aggregation strategy (across multiple evidence passages):
  - If ANY passage contradicts   → HALLUCINATED
  - If ANY passage entails       → CORRECT
  - Otherwise                    → UNCERTAIN
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from transformers import pipeline as hf_pipeline
from loguru import logger

import config


@dataclass
class NLIResult:
    label:              str                    # CORRECT | HALLUCINATED | UNCERTAIN
    entailment_score:   float   = 0.0
    contradiction_score:float   = 0.0
    neutral_score:      float   = 0.0
    evidence_used:      str     = ""
    per_passage:        List[dict] = field(default_factory=list)


class NLIClassifier:
    """
    Wraps a HuggingFace cross-encoder NLI model.

    Model downloaded automatically on first run (~500 MB).
    Model card: https://huggingface.co/cross-encoder/nli-deberta-v3-base
    """

    # HuggingFace label map (model-specific)
    _LABEL_MAP = {
        "entailment":    "entailment",
        "contradiction": "contradiction",
        "neutral":       "neutral",
        # Some models use LABEL_0/1/2
        "LABEL_0":       "contradiction",
        "LABEL_1":       "entailment",
        "LABEL_2":       "neutral",
    }

    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading NLI model: {config.NLI_MODEL}  (device={'cuda' if device==0 else 'cpu'})")

        self._pipe = hf_pipeline(
            "text-classification",
            model=config.NLI_MODEL,
            device=device,
            top_k=None,         # return all scores
            truncation=True,
            max_length=512,
        )
        logger.info("NLI model loaded.")

    # ──────────────────────────────────────────────────────────

    def classify(self, claim: str, passages: List[str]) -> NLIResult:
        """
        Classify *claim* against a list of evidence *passages*.

        Returns
        -------
        NLIResult with aggregated label and scores.
        """
        if not passages:
            return NLIResult(
                label=config.LABEL_UNCERTAIN,
                evidence_used="No evidence retrieved.",
            )

        per_passage: List[dict] = []
        max_entail     = 0.0
        max_contradict = 0.0
        best_evidence  = ""

        for passage in passages:
            # Model input: premise = passage, hypothesis = claim
            scores_raw = self._pipe(f"{passage} [SEP] {claim}")
            scores     = self._parse_scores(scores_raw)

            per_passage.append({
                "passage":          passage[:200] + "..." if len(passage) > 200 else passage,
                "entailment":       scores["entailment"],
                "contradiction":    scores["contradiction"],
                "neutral":          scores["neutral"],
            })

            if scores["entailment"] > max_entail:
                max_entail    = scores["entailment"]
                best_evidence = passage

            if scores["contradiction"] > max_contradict:
                max_contradict = scores["contradiction"]
                if not best_evidence:
                    best_evidence = passage

        # ── Aggregate decision ────────────────────────────────
        label = self._aggregate(max_entail, max_contradict)

        return NLIResult(
            label=label,
            entailment_score=max_entail,
            contradiction_score=max_contradict,
            neutral_score=1.0 - max_entail - max_contradict,
            evidence_used=best_evidence,
            per_passage=per_passage,
        )

    # ──────────────────────────────────────────────────────────

    def _parse_scores(self, raw) -> dict:
        """Normalise raw HuggingFace output into a score dict."""
        scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        items  = raw[0] if isinstance(raw[0], list) else raw  # handle nested list

        for item in items:
            canonical = self._LABEL_MAP.get(item["label"], "neutral")
            scores[canonical] = float(item["score"])

        return scores

    @staticmethod
    def _aggregate(entail: float, contradict: float) -> str:
        # Contradiction wins if strong enough
        if contradict >= config.CONTRADICTION_THRESHOLD and contradict > entail:
            return config.LABEL_HALLUCINATED
        if entail >= config.ENTAILMENT_THRESHOLD:
            return config.LABEL_CORRECT
        return config.LABEL_UNCERTAIN

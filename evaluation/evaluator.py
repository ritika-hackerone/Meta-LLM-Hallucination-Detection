"""
evaluation/evaluator.py
────────────────────────
Runs the VerifierPipeline on a labelled dataset and computes:
  • Per-class Precision, Recall, F1
  • Macro / Weighted averages
  • Confusion matrix
  • Full results DataFrame (saved to CSV)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from loguru import logger

from pipeline.verifier_pipeline import VerifierPipeline
import config


LABELS = [config.LABEL_CORRECT, config.LABEL_HALLUCINATED, config.LABEL_UNCERTAIN]


class Evaluator:
    """
    Runs end-to-end evaluation on a list of labelled samples.

    Parameters
    ----------
    pipeline     : VerifierPipeline instance
    output_dir   : directory to save results (CSV + JSON report)
    """

    def __init__(
        self,
        pipeline:   VerifierPipeline,
        output_dir: str = "results",
    ):
        self.pipeline   = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────

    def evaluate(self, samples: List[dict], desc: str = "Evaluating") -> dict:
        """
        Evaluate the pipeline on *samples*.

        Each sample must have: question, response, true_label.

        Returns
        -------
        report : dict with precision, recall, F1 and all raw results
        """
        rows          = []
        y_true        = []
        y_pred        = []

        for sample in tqdm(samples, desc=desc):
            q       = sample["question"]
            resp    = sample["response"]
            gt      = sample["true_label"]

            t0 = time.time()
            try:
                result = self.pipeline.verify(question=q, response=resp)
                pred   = result.final_label
            except Exception as exc:
                logger.error(f"Pipeline error on '{q[:60]}': {exc}")
                pred = config.LABEL_UNCERTAIN

            elapsed = time.time() - t0

            y_true.append(gt)
            y_pred.append(pred)

            rows.append({
                "question":       q,
                "response":       resp[:200],
                "true_label":     gt,
                "predicted":      pred,
                "correct":        gt == pred,
                "confidence":     result.confidence if "result" in dir() else 0,
                "explanation":    result.explanation[:200] if "result" in dir() else "",
                "latency_s":      round(elapsed, 2),
            })

        # ── Metrics ───────────────────────────────────────────
        report = self._compute_metrics(y_true, y_pred, rows)
        self._save_results(rows, report)
        return report

    # ──────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
        rows:   List[dict],
    ) -> dict:
        # sklearn report
        sk_report = classification_report(
            y_true, y_pred,
            labels=LABELS,
            zero_division=0,
            output_dict=True,
        )

        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred,
            labels=LABELS,
            zero_division=0,
            average=None,
        )

        cm = confusion_matrix(y_true, y_pred, labels=LABELS)

        report = {
            "num_samples":    len(y_true),
            "accuracy":       round(sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), 4),
            "per_class":      {},
            "macro_avg":      sk_report.get("macro avg", {}),
            "weighted_avg":   sk_report.get("weighted avg", {}),
            "confusion_matrix": cm.tolist(),
            "labels":         LABELS,
        }

        for i, label in enumerate(LABELS):
            report["per_class"][label] = {
                "precision": round(float(prec[i]), 4),
                "recall":    round(float(rec[i]),  4),
                "f1":        round(float(f1[i]),   4),
                "support":   int(support[i]),
            }

        # Pretty print
        logger.info("\n" + "="*60)
        logger.info(f"  EVALUATION RESULTS  ({report['num_samples']} samples)")
        logger.info("="*60)
        logger.info(f"  Accuracy: {report['accuracy']*100:.1f}%")
        logger.info("")
        logger.info(f"  {'Label':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        logger.info(f"  {'-'*55}")
        for label in LABELS:
            m = report["per_class"][label]
            logger.info(f"  {label:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")
        logger.info("")
        ma = report["macro_avg"]
        logger.info(f"  {'Macro avg':<15} {ma.get('precision',0):>10.4f} {ma.get('recall',0):>10.4f} {ma.get('f1-score',0):>10.4f}")
        logger.info("="*60 + "\n")

        return report

    def _save_results(self, rows: List[dict], report: dict):
        # CSV of all predictions
        df_path = self.output_dir / "predictions.csv"
        pd.DataFrame(rows).to_csv(df_path, index=False)
        logger.info(f"Predictions saved → {df_path}")

        # JSON report
        rpt_path = self.output_dir / "metrics_report.json"
        with open(rpt_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Metrics report saved → {rpt_path}")

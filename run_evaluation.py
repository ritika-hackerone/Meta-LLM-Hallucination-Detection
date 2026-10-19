"""
run_evaluation.py
──────────────────
CLI script to run the full TruthfulQA evaluation.

Usage
-----
    python run_evaluation.py                  # default: 100 samples
    python run_evaluation.py --n 50           # fewer samples (faster)
    python run_evaluation.py --no-judge       # skip Judge LLM (NLI-only, no API cost)
    python run_evaluation.py --output my_dir  # custom output directory
"""

import argparse
from loguru import logger

from pipeline.verifier_pipeline import VerifierPipeline
from evaluation.truthfulqa_loader import load_truthfulqa
from evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="TruthfulQA Evaluation Runner")
    parser.add_argument("--n",        type=int, default=100, help="Number of samples")
    parser.add_argument("--seed",     type=int, default=42,  help="Random seed")
    parser.add_argument("--output",   type=str, default="results", help="Output directory")
    parser.add_argument("--no-judge", action="store_true",   help="Skip Judge LLM (NLI only)")
    args = parser.parse_args()

    use_judge = not args.no_judge
    logger.info(f"Starting evaluation | n={args.n} | judge={use_judge}")

    # Load dataset
    samples = load_truthfulqa(n_samples=args.n, seed=args.seed)

    # Build pipeline
    pipe = VerifierPipeline(use_judge=use_judge)

    # Run evaluation
    evaluator = Evaluator(pipeline=pipe, output_dir=args.output)
    report    = evaluator.evaluate(samples, desc="TruthfulQA Eval")

    # Final summary
    print("\n✅  Evaluation complete!")
    print(f"   Accuracy : {report['accuracy']*100:.1f}%")
    ma = report["macro_avg"]
    print(f"   Macro F1 : {ma.get('f1-score',0):.4f}")
    print(f"   Results  : {args.output}/")


if __name__ == "__main__":
    main()

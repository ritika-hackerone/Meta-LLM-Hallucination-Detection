"""
test_setup.py — Environment smoke test
Run:  python test_setup.py
"""

import sys
import importlib

PASS = "  [OK] "
FAIL = "  [XX] "
WARN = "  [!!] "

failures = 0

print("\n" + "="*50)
print("  Meta-LLM Setup Verification")
print("="*50)

# ── 1. Package imports ───────────────────────────────
print("\n[1] Package imports")

packages = [
    ("torch",                "torch"),
    ("transformers",         "transformers"),
    ("sentence_transformers","sentence_transformers"),
    ("faiss",                "faiss"),
    ("datasets",             "datasets"),
    ("anthropic",            "anthropic"),
    ("wikipedia-api",        "wikipediaapi"),
    ("streamlit",            "streamlit"),
    ("plotly",               "plotly"),
    ("sklearn",              "sklearn"),
    ("pandas",               "pandas"),
    ("loguru",               "loguru"),
    ("dotenv",               "dotenv"),
    ("numpy",                "numpy"),
    ("tqdm",                 "tqdm"),
]

for display_name, import_name in packages:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "ok")
        print(f"{PASS}{display_name}  ({ver})")
    except ImportError as e:
        print(f"{FAIL}{display_name}  → NOT INSTALLED: {e}")
        failures += 1

# ── 2. .env file ─────────────────────────────────────
print("\n[2] Environment (.env file)")

from pathlib import Path

if Path(".env").exists():
    print(f"{PASS}.env file found")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        import os
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if key and not key.startswith("sk-ant-your"):
            print(f"{PASS}ANTHROPIC_API_KEY is set  ({key[:16]}...)")
        else:
            print(f"{WARN}ANTHROPIC_API_KEY not set — demo mode only")
    except Exception as e:
        print(f"{FAIL}Could not read .env: {e}")
else:
    print(f"{WARN}.env not found — copy .env.example to .env and add your key")

# ── 3. Project structure ─────────────────────────────
print("\n[3] Project structure")

required_files = [
    "config.py",
    "app.py",
    "run_evaluation.py",
    "pipeline/__init__.py",
    "pipeline/rag_retriever.py",
    "pipeline/nli_classifier.py",
    "pipeline/meta_llm.py",
    "pipeline/verifier_pipeline.py",
    "evaluation/__init__.py",
    "evaluation/truthfulqa_loader.py",
    "evaluation/evaluator.py",
]

for f in required_files:
    if Path(f).exists():
        print(f"{PASS}{f}")
    else:
        print(f"{FAIL}{f}  → FILE MISSING")
        failures += 1

# ── 4. Wikipedia API ─────────────────────────────────
print("\n[4] Wikipedia API (network test)")

try:
    import wikipediaapi
    wiki = wikipediaapi.Wikipedia(
        user_agent="MetaLLMTest/1.0",
        language="en"
    )
    page = wiki.page("Artificial intelligence")
    if page.exists():
        print(f"{PASS}Wikipedia fetch OK  (page length: {len(page.summary)} chars)")
    else:
        print(f"{WARN}Wikipedia page not found — check internet connection")
except Exception as e:
    print(f"{FAIL}Wikipedia API error: {e}")
    failures += 1

# ── 5. Embeddings ────────────────────────────────────
print("\n[5] Sentence-Transformers embeddings")

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode(["Hello world"], normalize_embeddings=True)
    assert vec.shape == (1, 384), f"Wrong shape: {vec.shape}"
    print(f"{PASS}all-MiniLM-L6-v2 loaded  (output shape: {vec.shape})")
except Exception as e:
    print(f"{FAIL}Embedding error: {e}")
    failures += 1

# ── 6. FAISS ─────────────────────────────────────────
print("\n[6] FAISS vector index")

try:
    import faiss
    import numpy as np
    idx = faiss.IndexFlatIP(384)
    dummy = np.random.rand(5, 384).astype("float32")
    idx.add(dummy)
    _, I = idx.search(dummy[:1], 3)
    assert I.shape == (1, 3)
    print(f"{PASS}FAISS IndexFlatIP  (add + search OK)")
except Exception as e:
    print(f"{FAIL}FAISS error: {e}")
    failures += 1

# ── 7. NLI model (optional — slow first run) ─────────
print("\n[7] NLI Model — DeBERTa-v3")
print(f"{WARN}Skipping live NLI forward pass (downloads ~500MB on first run)")
print(f"      Run app.py and it will download automatically on first use.")

# ── 8. Anthropic API ─────────────────────────────────
print("\n[8] Anthropic API key")

try:
    from dotenv import load_dotenv
    load_dotenv()
    import os
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if key and not key.startswith("sk-ant-your"):
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say: OK"}]
        )
        print(f"{PASS}Anthropic API live call OK  → '{msg.content[0].text.strip()}'")
    else:
        print(f"{WARN}Skipped — no valid ANTHROPIC_API_KEY in .env")
        print(f"      App will run in demo mode without a key.")
except Exception as e:
    print(f"{FAIL}Anthropic API error: {e}")
    failures += 1

# ── Summary ───────────────────────────────────────────
print("\n" + "="*50)
if failures == 0:
    print("  ALL CHECKS PASSED")
    print("  Run:  streamlit run app.py")
else:
    print(f"  {failures} CHECK(S) FAILED")
    print("  Fix the errors above, then re-run: python test_setup.py")
print("="*50 + "\n")

sys.exit(failures)

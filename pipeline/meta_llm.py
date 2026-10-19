# """
# pipeline/meta_llm.py
# ─────────────────────
# Meta-LLM: uses Claude (Anthropic) to:
#   1. Generate an initial answer to a question (the "student" LLM)
#   2. Produce a structured verification verdict given evidence + NLI result (the "judge" LLM)

# Both roles use the same underlying model but with different system prompts.
# """

# from __future__ import annotations

# import anthropic
# from loguru import logger

# import config

# _CLIENT = None


# # def _get_client() -> anthropic.Anthropic:
# #     global _CLIENT
# #     if _CLIENT is None:
# #         if not config.ANTHROPIC_API_KEY:
# #             raise EnvironmentError(
# #                 "ANTHROPIC_API_KEY is not set. "
# #                 "Copy .env.example → .env and add your key."
# #             )
# #         _CLIENT = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
# #     return _CLIENT
# def _get_client() -> anthropic.Anthropic:
#     global _CLIENT
#     if _CLIENT is None:
#         if not config.ANTHROPIC_API_KEY:
#             raise EnvironmentError(
#                 "ANTHROPIC_API_KEY is not set. "
#                 "Copy .env.example → .env and add your key."
#             )
#         _CLIENT = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
#     return _CLIENT


# # ─────────────────────────────────────────────────────────────────────────────
# # Role 1: Student LLM — answers questions (may hallucinate)
# # ─────────────────────────────────────────────────────────────────────────────

# STUDENT_SYSTEM = """You are a knowledgeable assistant. Answer questions clearly and concisely.
# Do not say "I don't know" — give your best factual answer in 1-3 sentences."""


# def generate_answer(question: str) -> str:
#     """
#     Generate a direct answer to *question* using Claude.
#     This answer will then be fact-checked by the pipeline.
#     """
#     client = _get_client()
#     logger.info(f"[MetaLLM-Student] Generating answer for: '{question[:80]}'")

#     message = client.messages.create(
#         model="claude-sonnet-4-20250514",
#         max_tokens=256,
#         system=STUDENT_SYSTEM,
#         messages=[{"role": "user", "content": question}],
#     )
#     answer = message.content[0].text.strip()
#     logger.debug(f"[MetaLLM-Student] Answer: {answer[:120]}")
#     return answer


# # ─────────────────────────────────────────────────────────────────────────────
# # Role 2: Judge LLM — synthesises NLI result + evidence into a verdict
# # ─────────────────────────────────────────────────────────────────────────────

# JUDGE_SYSTEM = """You are a rigorous fact-checker. You are given:
# - A QUESTION
# - A RESPONSE to evaluate
# - EVIDENCE retrieved from Wikipedia
# - An NLI SIGNAL (entailment/contradiction/neutral scores)

# Your task: output a concise VERDICT with:
#   • Final label: CORRECT | HALLUCINATED | UNCERTAIN
#   • A 1-2 sentence explanation citing the evidence
#   • Confidence (0-100%)

# Format your response EXACTLY as:
# LABEL: <label>
# CONFIDENCE: <0-100>%
# EXPLANATION: <explanation>"""


# def judge_response(
#     question:   str,
#     response:   str,
#     evidence:   str,
#     nli_label:  str,
#     entail_score: float,
#     contra_score: float,
# ) -> dict:
#     """
#     Ask the judge LLM to synthesise everything into a final verdict.

#     Returns a dict with keys: label, confidence, explanation
#     """
#     client = _get_client()

#     user_msg = f"""QUESTION: {question}

# RESPONSE TO EVALUATE: {response}

# EVIDENCE (from Wikipedia):
# {evidence if evidence else "No relevant evidence found."}

# NLI SIGNAL:
#   - NLI label:          {nli_label}
#   - Entailment score:   {entail_score:.2f}
#   - Contradiction score:{contra_score:.2f}

# Please output your verdict."""

#     logger.info("[MetaLLM-Judge] Generating verdict...")

#     message = client.messages.create(
#         model="claude-sonnet-4-20250514",
#         max_tokens=200,
#         system=JUDGE_SYSTEM,
#         messages=[{"role": "user", "content": user_msg}],
#     )

#     raw = message.content[0].text.strip()
#     return _parse_verdict(raw, nli_label)


# def _parse_verdict(raw: str, fallback_label: str) -> dict:
#     """Parse the structured verdict from the judge LLM."""
#     result = {
#         "label":       fallback_label,
#         "confidence":  50,
#         "explanation": raw,
#         "raw":         raw,
#     }

#     for line in raw.splitlines():
#         line = line.strip()
#         if line.startswith("LABEL:"):
#             val = line.replace("LABEL:", "").strip().upper()
#             if val in (config.LABEL_CORRECT, config.LABEL_HALLUCINATED, config.LABEL_UNCERTAIN):
#                 result["label"] = val
#         elif line.startswith("CONFIDENCE:"):
#             try:
#                 result["confidence"] = int(line.replace("CONFIDENCE:", "").replace("%", "").strip())
#             except ValueError:
#                 pass
#         elif line.startswith("EXPLANATION:"):
#             result["explanation"] = line.replace("EXPLANATION:", "").strip()

#     return result



















# """
# pipeline/meta_llm.py
# ─────────────────────
# Meta-LLM module — supports multiple LLM backends:

#   1. Anthropic  (Claude)   — set ANTHROPIC_API_KEY  in .env
#   2. Gemini     (Google)   — set GEMINI_API_KEY      in .env  ← FREE
#   3. Groq       (Llama3)   — set GROQ_API_KEY        in .env  ← FREE
#   4. Ollama     (Local)    — set OLLAMA_MODEL        in .env  ← FREE + offline

# Priority order: Anthropic → Gemini → Groq → Ollama → Demo mode
# The first key found in .env is used automatically.
# """

# from __future__ import annotations
# import os
# from dotenv import load_dotenv
# from loguru import logger

# load_dotenv()

# # ── Detect which backend is available ────────────────────────
# def _detect_backend() -> str:
#     if os.getenv("ANTHROPIC_API_KEY", "").startswith("sk-ant") and \
#        not os.getenv("ANTHROPIC_API_KEY","").endswith("your-key-here"):
#         return "anthropic"
#     if os.getenv("GEMINI_API_KEY", "") and \
#        not os.getenv("GEMINI_API_KEY","").endswith("your-key-here"):
#         return "gemini"
#     if os.getenv("GROQ_API_KEY", "") and \
#        not os.getenv("GROQ_API_KEY","").endswith("your-key-here"):
#         return "groq"
#     if os.getenv("OLLAMA_MODEL", ""):
#         return "ollama"
#     return "demo"

# BACKEND = _detect_backend()
# logger.info(f"[MetaLLM] Backend selected: {BACKEND.upper()}")


# # ══════════════════════════════════════════════════════════════
# # SYSTEM PROMPTS  (shared across all backends)
# # ══════════════════════════════════════════════════════════════

# STUDENT_SYSTEM = """You are a knowledgeable assistant. Answer questions clearly and concisely.
# Give your best factual answer in 1-3 sentences. Do not say 'I don't know'."""

# JUDGE_SYSTEM = """You are a rigorous fact-checker. You are given:
# - A QUESTION
# - A RESPONSE to evaluate
# - EVIDENCE retrieved from Wikipedia
# - An NLI SIGNAL (entailment/contradiction/neutral scores)

# Output your verdict EXACTLY in this format (no extra text):
# LABEL: <CORRECT or HALLUCINATED or UNCERTAIN>
# CONFIDENCE: <number 0-100>%
# EXPLANATION: <one sentence citing the evidence>"""


# # ══════════════════════════════════════════════════════════════
# # UNIFIED CALL FUNCTION
# # ══════════════════════════════════════════════════════════════

# def _call_llm(system_prompt: str, user_message: str, max_tokens: int = 300) -> str:
#     """Route to the correct backend and return the text response."""

#     # if BACKEND == "anthropic":
#     #     return _call_anthropic(system_prompt, user_message, max_tokens)
#     if BACKEND == "gemini":
#         return _call_gemini(system_prompt, user_message, max_tokens)
#     # elif BACKEND == "groq":
#     #     return _call_groq(system_prompt, user_message, max_tokens)
#     # elif BACKEND == "ollama":
#     #     return _call_ollama(system_prompt, user_message, max_tokens)
#     else:
#         return _demo_response(user_message)


# # ══════════════════════════════════════════════════════════════
# # BACKEND IMPLEMENTATIONS
# # ══════════════════════════════════════════════════════════════

# # ── 1. Anthropic (Claude) ─────────────────────────────────────
# # def _call_anthropic(system: str, user: str, max_tokens: int) -> str:
# #     import anthropic
# #     client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# #     msg = client.messages.create(
# #         model="claude-sonnet-4-20250514",
# #         max_tokens=max_tokens,
# #         system=system,
# #         messages=[{"role": "user", "content": user}],
# #     )
# #     return msg.content[0].text.strip()


# # ── 2. Google Gemini (FREE) ───────────────────────────────────
# def _call_gemini(system: str, user: str, max_tokens: int) -> str:
#     import google.generativeai as genai
#     genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#     model = genai.GenerativeModel(
#         model_name="gemini-1.5-flash",   # free tier model
#         system_instruction=system,
#     )
#     resp = model.generate_content(
#         user,
#         generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
#     )
#     return resp.text.strip()


# # # ── 3. Groq (FREE — Llama3) ───────────────────────────────────
# # def _call_groq(system: str, user: str, max_tokens: int) -> str:
# #     from groq import Groq
# #     client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# #     chat = client.chat.completions.create(
# #         model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),  # free model
# #         max_tokens=max_tokens,
# #         messages=[
# #             {"role": "system",  "content": system},
# #             {"role": "user",    "content": user},
# #         ],
# #     )
# #     return chat.choices[0].message.content.strip()


# # # ── 4. Ollama (LOCAL — no API key) ────────────────────────────
# # def _call_ollama(system: str, user: str, max_tokens: int) -> str:
# #     import requests, json
# #     model = os.getenv("OLLAMA_MODEL", "llama3.2")
# #     url   = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
# #     payload = {
# #         "model": model,
# #         "stream": False,
# #         "options": {"num_predict": max_tokens},
# #         "messages": [
# #             {"role": "system", "content": system},
# #             {"role": "user",   "content": user},
# #         ],
# #     }
# #     resp = requests.post(url, json=payload, timeout=120)
# #     resp.raise_for_status()
# #     return resp.json()["message"]["content"].strip()


# # ── 5. Demo mode (no key needed) ──────────────────────────────
# def _demo_response(user: str) -> str:
#     """Returns a plausible canned response when no API key is available."""
#     import random, hashlib
#     seed = int(hashlib.md5(user.encode()).hexdigest(), 16) % 9999
#     random.seed(seed)
#     labels = ["CORRECT", "HALLUCINATED", "UNCERTAIN"]
#     lbl    = random.choices(labels, [0.44, 0.37, 0.19])[0]
#     conf   = random.randint(62, 95) if lbl != "UNCERTAIN" else random.randint(35, 58)
#     exps   = {
#         "CORRECT":      "The response aligns with retrieved Wikipedia evidence.",
#         "HALLUCINATED": "The response contradicts retrieved evidence.",
#         "UNCERTAIN":    "Evidence is insufficient to confirm or deny the response.",
#     }
#     return f"LABEL: {lbl}\nCONFIDENCE: {conf}%\nEXPLANATION: {exps[lbl]}"


# # ══════════════════════════════════════════════════════════════
# # PUBLIC API  (called by verifier_pipeline.py)
# # ══════════════════════════════════════════════════════════════

# def generate_answer(question: str) -> str:
#     """Student LLM: generate an answer to fact-check."""
#     logger.info(f"[MetaLLM-Student/{BACKEND}] Answering: '{question[:70]}'")
#     return _call_llm(STUDENT_SYSTEM, question, max_tokens=200)


# def judge_response(
#     question:     str,
#     response:     str,
#     evidence:     str,
#     nli_label:    str,
#     entail_score: float,
#     contra_score: float,
# ) -> dict:
#     """Judge LLM: verify the response against evidence and return a verdict dict."""
#     logger.info(f"[MetaLLM-Judge/{BACKEND}] Judging response…")

#     user_msg = f"""QUESTION: {question}

# RESPONSE TO EVALUATE: {response}

# EVIDENCE (from Wikipedia):
# {evidence if evidence else "No relevant evidence found."}

# NLI SIGNAL:
#   NLI label:           {nli_label}
#   Entailment score:    {entail_score:.2f}
#   Contradiction score: {contra_score:.2f}

# Output your verdict now."""

#     raw = _call_llm(JUDGE_SYSTEM, user_msg, max_tokens=200)
#     return _parse_verdict(raw, nli_label)


# def _parse_verdict(raw: str, fallback_label: str) -> dict:
#     """Parse structured verdict from judge LLM output."""
#     result = {
#         "label":       fallback_label,
#         "confidence":  50,
#         "explanation": raw,
#         "raw":         raw,
#     }
#     for line in raw.splitlines():
#         line = line.strip()
#         if line.upper().startswith("LABEL:"):
#             val = line.split(":", 1)[1].strip().upper()
#             if val in ("CORRECT", "HALLUCINATED", "UNCERTAIN"):
#                 result["label"] = val
#         elif line.upper().startswith("CONFIDENCE:"):
#             try:
#                 result["confidence"] = int(
#                     line.split(":", 1)[1].strip().replace("%", "").strip()
#                 )
#             except ValueError:
#                 pass
#         elif line.upper().startswith("EXPLANATION:"):
#             result["explanation"] = line.split(":", 1)[1].strip()
#     return result


# def get_active_backend() -> str:
#     """Returns the name of the currently active backend."""
#     return BACKEND


# 3
"""
pipeline/meta_llm.py
─────────────────────
Meta-LLM module — Gemini (Google) backend.

Uses the NEW google-genai SDK (GA since May 2025).
  pip install google-genai

Get a free API key at: https://aistudio.google.com/app/apikey
Free tier: gemini-2.5-flash → 10 RPM / 250 requests per day
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

BACKEND = "gemini"
_MODEL  = "gemini-2.5-flash"   # free tier; 10 RPM, 250 req/day
logger.info(f"[MetaLLM] Backend: {BACKEND.upper()} | Model: {_MODEL}")


# ══════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════

STUDENT_SYSTEM = """You are a knowledgeable assistant. Answer questions clearly and concisely.
Give your best factual answer in 1-3 sentences. Do not say 'I don't know'."""

JUDGE_SYSTEM = """You are a rigorous fact-checker. You are given:
- A QUESTION
- A RESPONSE to evaluate
- EVIDENCE retrieved from Wikipedia
- An NLI SIGNAL (entailment/contradiction/neutral scores)

Output your verdict EXACTLY in this format (no extra text):
LABEL: <CORRECT or HALLUCINATED or UNCERTAIN>
CONFIDENCE: <number 0-100>%
EXPLANATION: <one sentence citing the evidence>"""


# ══════════════════════════════════════════════════════════════
# GEMINI CLIENT  (new google-genai SDK — from google import genai)
# ══════════════════════════════════════════════════════════════

_CLIENT = None

def _get_client():
    """Lazy-initialised singleton Gemini client."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    from google import genai

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key.endswith("your-key-here"):
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.\n"
            "  1. Go to https://aistudio.google.com/app/apikey\n"
            "  2. Click 'Create API key'\n"
            "  3. Add to your .env:  GEMINI_API_KEY=AIza...\n"
        )

    _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


# ══════════════════════════════════════════════════════════════
# CORE CALL FUNCTION
# ══════════════════════════════════════════════════════════════

def _call_llm(system_prompt: str, user_message: str, max_tokens: int = 300) -> str:
    """Route to Gemini and return the text response."""
    from google.genai import types

    client = _get_client()
    try:
        response = client.models.generate_content(
            model=_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"[MetaLLM/Gemini] API call failed: {e}")
        raise


# ══════════════════════════════════════════════════════════════
# PUBLIC API  (called by verifier_pipeline.py)
# ══════════════════════════════════════════════════════════════

def generate_answer(question: str) -> str:
    """Student LLM: generate an answer to fact-check."""
    logger.info(f"[MetaLLM-Student/Gemini] Answering: '{question[:70]}'")
    return _call_llm(STUDENT_SYSTEM, question, max_tokens=200)


def judge_response(
    question:     str,
    response:     str,
    evidence:     str,
    nli_label:    str,
    entail_score: float,
    contra_score: float,
) -> dict:
    """Judge LLM: verify the response against evidence and return a verdict dict."""
    logger.info("[MetaLLM-Judge/Gemini] Judging response…")

    user_msg = f"""QUESTION: {question}

RESPONSE TO EVALUATE: {response}

EVIDENCE (from Wikipedia):
{evidence if evidence else "No relevant evidence found."}

NLI SIGNAL:
  NLI label:           {nli_label}
  Entailment score:    {entail_score:.2f}
  Contradiction score: {contra_score:.2f}

Output your verdict now."""

    raw = _call_llm(JUDGE_SYSTEM, user_msg, max_tokens=200)
    return _parse_verdict(raw, nli_label)


def _parse_verdict(raw: str, fallback_label: str) -> dict:
    """Parse structured verdict from judge LLM output."""
    result = {
        "label":       fallback_label,
        "confidence":  50,
        "explanation": raw,
        "raw":         raw,
    }
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("LABEL:"):
            val = line.split(":", 1)[1].strip().upper()
            if val in ("CORRECT", "HALLUCINATED", "UNCERTAIN"):
                result["label"] = val
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                result["confidence"] = int(
                    line.split(":", 1)[1].strip().replace("%", "").strip()
                )
            except ValueError:
                pass
        elif line.upper().startswith("EXPLANATION:"):
            result["explanation"] = line.split(":", 1)[1].strip()
    return result


def get_active_backend() -> str:
    """Returns the name of the currently active backend."""
    return BACKEND
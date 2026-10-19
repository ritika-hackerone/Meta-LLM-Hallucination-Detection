<div align="center">

# 🧠 Meta-LLM Hallucination Detection System

### End-to-end factual verification — RAG · NLI · LLM-as-Judge · TruthfulQA

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A prototype Meta-LLM pipeline that detects hallucinations in LLM-generated responses  
> by combining Retrieval-Augmented Generation (RAG), Natural Language Inference (NLI),  
> and an LLM-as-Judge architecture — evaluated on the TruthfulQA benchmark.

---

<!-- Add your screenshot here -->
![Dashboard Preview](assets/banner.png)

</div>

---

## 📌 What is Hallucination?

LLMs sometimes generate **confident but factually wrong** statements — called *hallucinations*.  
This system **detects** them by:

1. 🤖 **Generating** an answer with Gemini / Claude
2. 🗃️ **Retrieving** ground-truth facts from Wikipedia via FAISS semantic search (RAG)
3. 🔬 **Classifying** whether the answer agrees with facts using DeBERTa NLI
4. ⚖️ **Judging** with an LLM-as-Judge to produce a final verdict + explanation

---

## 🏗️ Pipeline Architecture

```
User Question
      │
      ▼
┌─────────────────────┐
│  Student LLM        │  → Generates an answer
│  (Gemini / Claude)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  RAG — Wikipedia    │  → Retrieves Top-K evidence passages
│  FAISS + MiniLM     │     via semantic similarity search
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  DeBERTa-v3 NLI     │  → Entailment / Contradiction / Neutral
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Judge LLM          │  → Final Verdict + Confidence + Explanation
│  (Gemini / Claude)  │
└─────────────────────┘
           │
           ▼
  ✅ CORRECT / ❌ HALLUCINATED / ⚠️ UNCERTAIN
```

---

## ⚙️ Tech Stack

| Component | Tool / Model |
|---|---|
| 🗂️ Dataset | [TruthfulQA](https://huggingface.co/datasets/truthful_qa) |
| 🤖 LLM | Google Gemini 1.5 Flash / Anthropic Claude |
| 🔢 Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| 🗃️ Vector DB | [FAISS](https://faiss.ai) |
| 🔬 NLI Model | [DeBERTa-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) |
| 📊 Dashboard | Streamlit + Plotly |
| 📏 Metrics | Scikit-learn |

---

## 📸 Screenshots

| Verify Query | Evaluation Results |
|---|---|
| ![verify](assets/verify.png) | ![eval](assets/eval.png) |

| Category Comparison | Architecture |
|---|---|
| ![category](assets/category.png) | ![arch](assets/architecture.png) |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ritika-hackerone/Meta-LLM-Hallucination-Detection.git
cd Meta-LLM-Hallucination-Detection

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
python setup.py

# 4. Launch the app
streamlit run app.py
```

---

## 🔑 Get Free API Key

| Provider | Link |
|---|---|
| Google Gemini | https://aistudio.google.com |
| Anthropic Claude | https://console.anthropic.com |

---

## 📊 Evaluation Results (40 samples)

| Metric | Score |
|---|---|
| Accuracy | 2.5% *(prototype — Wikipedia RAG mismatch with TruthfulQA)* |
| Macro Precision | 0.111 |
| Macro Recall | 0.015 |
| Macro F1-Score | 0.027 |

> **Note:** Low scores are expected at prototype stage — TruthfulQA questions are  
> opinion/myth-based and Wikipedia retrieval often returns no matching passage,  
> causing the model to default to UNCERTAIN. Improving retrieval source is the key next step.

---


## 🔮 Future Improvements

- [ ] Swap Wikipedia RAG → TruthfulQA-specific knowledge base
- [ ] Fine-tune NLI thresholds per question category
- [ ] Add more LLM providers (GPT-4, Mistral)
- [ ] Deploy on Streamlit Cloud / Hugging Face Spaces
- [ ] Improve confidence calibration

---

## 👩‍💻 Author

**Ritika Solanki** — M.Tech AI/ML, JECRC University  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-ritikasolankii-blue?logo=linkedin)](https://linkedin.com/in/ritikasolankii)
[![GitHub](https://img.shields.io/badge/GitHub-ritika--hackerone-black?logo=github)](https://github.com/ritika-hackerone)

---

<div align="center">
⭐ Star this repo if you found it useful!
</div>




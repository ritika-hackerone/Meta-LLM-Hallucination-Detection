"""
app.py ─ Meta-LLM Hallucination Detection Dashboard
─────────────────────────────────────────────────────
Run:  streamlit run app.py

Outputs per verification:
  ① Answer   — generated or user-supplied response
  ② Verdict  — CORRECT | HALLUCINATED | UNCERTAIN
  ③ Confidence Score (0-100 %)
  ④ Evidence — retrieved Wikipedia facts

Evaluation tab:
  • Accuracy, Precision, Recall, F1-Score on TruthfulQA
  • Hallucination category comparison + NLI scatter

Category Comparison tab:
  • Donut distribution, avg confidence, scatter plot
"""

import json, time, random
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meta-LLM · Hallucination Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,300&family=DM+Sans:wght@400;500;700&display=swap');

:root{
  --bg:#f5f2eb;--surface:#faf8f3;--ink:#1a1410;--muted:#7a6f62;
  --border:#d9d2c5;--accent:#c8401a;
  --correct:#1a7a42;--halluc:#c8401a;--uncert:#b07c10;
  --correct-bg:#eef7f2;--halluc-bg:#fdf0ec;--uncert-bg:#fdf6e8;
  --card:#ffffff;--shadow:rgba(26,20,16,.08);
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{
  background:var(--bg)!important;font-family:'DM Sans',sans-serif;color:var(--ink);}
[data-testid="stSidebar"]{background:var(--ink)!important;border-right:none;}
[data-testid="stSidebar"] *{color:#f5f2eb!important;}
[data-testid="stSidebar"] a{color:#f0a97a!important;}
[data-testid="stSidebar"] hr{border-color:#3a3028!important;}
[data-testid="stSidebar"] label{color:#c8b8a8!important;font-size:.8rem!important;}
h1,h2,h3{font-family:'Fraunces',serif;font-weight:700;color:var(--ink);}
.stTextArea textarea,.stTextInput>div>div>input{
  background:var(--card)!important;border:1.5px solid var(--border)!important;
  border-radius:6px!important;color:var(--ink)!important;
  font-family:'DM Sans',sans-serif!important;font-size:.93rem!important;}
.stTextArea textarea:focus,.stTextInput>div>div>input:focus{
  border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(200,64,26,.12)!important;}
div[data-testid="stButton"]>button{
  background:var(--ink)!important;color:#f5f2eb!important;border:none!important;
  border-radius:6px!important;font-family:'IBM Plex Mono',monospace!important;
  font-size:.78rem!important;font-weight:600!important;letter-spacing:.06em!important;
  text-transform:uppercase!important;padding:.55rem 1.5rem!important;
  transition:background .15s,transform .1s!important;}
div[data-testid="stButton"]>button:hover{background:var(--accent)!important;transform:translateY(-1px)!important;}
[data-testid="stTabs"] [role="tablist"]{border-bottom:2px solid var(--border);gap:0;}
[data-testid="stTabs"] button[role="tab"]{
  font-family:'IBM Plex Mono',monospace;font-size:.73rem;text-transform:uppercase;
  letter-spacing:.09em;color:var(--muted)!important;padding:.5rem 1.1rem!important;
  border-bottom:2px solid transparent!important;margin-bottom:-2px;}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
  color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;font-weight:700!important;}
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);
  border-radius:8px;padding:.8rem 1rem!important;}
[data-testid="stMetricLabel"]{font-size:.7rem!important;color:var(--muted)!important;
  font-family:'IBM Plex Mono',monospace!important;text-transform:uppercase!important;}
[data-testid="stMetricValue"]{font-family:'Fraunces',serif!important;font-size:1.65rem!important;color:var(--ink)!important;}
[data-testid="stExpander"]{background:var(--card);border:1px solid var(--border)!important;border-radius:8px!important;}
hr{border-color:var(--border)!important;margin:1.1rem 0!important;}

.hero{padding:2.2rem 0 1.4rem;border-bottom:2px solid var(--ink);margin-bottom:1.8rem;}
.hero-eyebrow{font-family:'IBM Plex Mono',monospace;font-size:.7rem;letter-spacing:.18em;
  text-transform:uppercase;color:var(--accent);margin-bottom:.35rem;}
.hero-title{font-family:'Fraunces',serif;font-size:2.8rem;font-weight:700;line-height:1.06;color:var(--ink);}
.hero-sub{font-size:.93rem;color:var(--muted);margin-top:.5rem;max-width:520px;}

.result-shell{border:2px solid var(--border);border-radius:12px;overflow:hidden;
  margin:1.1rem 0;background:var(--card);box-shadow:0 2px 16px var(--shadow);}
.result-header{padding:.9rem 1.3rem;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:.65rem;}
.verdict-label{font-family:'IBM Plex Mono',monospace;font-size:.7rem;font-weight:700;
  letter-spacing:.13em;text-transform:uppercase;padding:.22rem .65rem;border-radius:4px;}
.verdict-CORRECT{background:var(--correct-bg);color:var(--correct);border:1px solid #a3d4b7;}
.verdict-HALLUCINATED{background:var(--halluc-bg);color:var(--halluc);border:1px solid #f0b8a8;}
.verdict-UNCERTAIN{background:var(--uncert-bg);color:var(--uncert);border:1px solid #e8d09a;}
.result-section{padding:.85rem 1.3rem;border-bottom:1px solid var(--border);}
.result-section:last-child{border-bottom:none;}
.section-tag{font-family:'IBM Plex Mono',monospace;font-size:.64rem;letter-spacing:.15em;
  text-transform:uppercase;color:var(--muted);margin-bottom:.32rem;}
.answer-text{font-family:'Fraunces',serif;font-size:1.03rem;line-height:1.56;color:var(--ink);}
.conf-bar-track{height:8px;background:var(--border);border-radius:4px;overflow:hidden;margin-top:.4rem;}
.conf-bar-fill{height:100%;border-radius:4px;}

.evidence-card{background:var(--bg);border:1px solid var(--border);
  border-left:3px solid var(--accent);border-radius:0 6px 6px 0;
  padding:.65rem .95rem;margin:.4rem 0;font-size:.83rem;line-height:1.54;color:var(--ink);}
.evidence-source{font-family:'IBM Plex Mono',monospace;font-size:.63rem;
  color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.22rem;}

.eval-card{background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:1.1rem 1.3rem;text-align:center;}
.eval-val{font-family:'Fraunces',serif;font-size:2.1rem;font-weight:700;color:var(--ink);}
.eval-lbl{font-family:'IBM Plex Mono',monospace;font-size:.67rem;text-transform:uppercase;
  letter-spacing:.12em;color:var(--muted);margin-top:.2rem;}

.hist-item{background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:.7rem .95rem;margin:.38rem 0;font-size:.83rem;display:flex;align-items:center;gap:.7rem;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading NLI model & RAG retriever…")
def load_pipeline(use_judge: bool):
    from pipeline.verifier_pipeline import VerifierPipeline
    return VerifierPipeline(use_judge=use_judge)


def _demo_result(question, response):
    import hashlib
    seed = int(hashlib.md5((question + response).encode()).hexdigest(), 16) % 9999
    random.seed(seed)
    lbl  = random.choices(["CORRECT","HALLUCINATED","UNCERTAIN"], [.44,.37,.19])[0]
    conf = random.randint(63,96) if lbl != "UNCERTAIN" else random.randint(38,59)
    ent  = round(random.uniform(.57,.93) if lbl=="CORRECT"      else random.uniform(.10,.38), 3)
    con  = round(random.uniform(.57,.90) if lbl=="HALLUCINATED" else random.uniform(.05,.28), 3)
    exps = {
        "CORRECT":      "The response is well-supported by retrieved Wikipedia evidence. Key facts align.",
        "HALLUCINATED": "The response contradicts retrieved evidence. Specific claims could not be verified.",
        "UNCERTAIN":    "Evidence is ambiguous. Passages neither confirm nor deny the response.",
    }
    return {
        "question":question,"response":response or f"[Auto-answer for: {question}]",
        "final_label":lbl,"confidence":conf,"explanation":exps[lbl],
        "nli_label":lbl,"entailment_score":ent,"contradiction_score":con,
        "retrieved_passages":[
            f"Historical records show that {question.rstrip('?')} is documented in academic literature.",
            "Multiple encyclopaedic sources provide detailed information on this subject.",
            "Peer-reviewed references confirm the established consensus on this topic.",
        ],
        "retrieved_sources":["Wikipedia (Demo)","Encyclopedia Britannica (Demo)","Academic DB (Demo)"],
        "auto_generated":not bool(response.strip()),"demo_mode":True,
    }


def run_verify(question, response, auto_gen, use_judge):
    import config as cfg
    if not cfg.GEMINI_API_KEY:
        return _demo_result(question, response)
    try:
        pipe = load_pipeline(use_judge)
        r    = pipe.verify(question=question, response=response, generate_answer=auto_gen)
        return r.to_dict()
    except Exception as e:
        st.warning(f"Pipeline error ({e}) — showing demo result.")
        return _demo_result(question, response)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state["history"] = []


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style='font-family:"IBM Plex Mono",monospace;font-size:.63rem;letter-spacing:.18em;
  text-transform:uppercase;color:#f0a97a;margin-bottom:.3rem;'>Meta-LLM System</div>
<div style='font-family:"Fraunces",serif;font-size:1.25rem;font-weight:700;
  line-height:1.2;margin-bottom:1.2rem;'>Hallucination<br>Detector</div>
""", unsafe_allow_html=True)

    st.markdown("#### ⚙️ Settings")
    use_judge = st.toggle("Judge LLM ", value=True,
        help="Disable for NLI-only mode — no API key needed.")

    st.markdown("---")
    st.markdown("#### 🧪 TruthfulQA Evaluation")
    n_eval = st.slider("Samples", 10, 150, 40, step=10)
    run_eval_btn = st.button("▶ Run Evaluation", use_container_width=True)

    st.markdown("---")
    st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;font-size:.69rem;
line-height:2;color:#c8b8a8;'>
<b style='color:#f5f2eb;'>Dataset</b><br>
<a href='https://huggingface.co/datasets/truthful_qa'>TruthfulQA ↗</a><br>
<b style='color:#f5f2eb;'>NLI Model</b><br>
<a href='https://huggingface.co/cross-encoder/nli-deberta-v3-base'>DeBERTa-v3-base ↗</a><br>
<b style='color:#f5f2eb;'>Embeddings</b><br>
<a href='https://www.sbert.net/'>sentence-transformers ↗</a><br>
<b style='color:#f5f2eb;'>Vector DB</b><br>
<a href='https://github.com/facebookresearch/faiss'>FAISS ↗</a><br>
<b style='color:#f5f2eb;'>LLM API</b><br>
<a href='https://aistudio.google.com/'>Google Gemini ↗</a>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">RAG · NLI · LLM-as-Judge · TruthfulQA</div>
  <div class="hero-title">Meta-LLM Hallucination<br>Detection System</div>
  <div class="hero-sub">
    End-to-end factual verification — retrieves Wikipedia evidence, runs NLI,
    and delivers a verdict with confidence score and cited facts.
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_v, tab_e, tab_c, tab_a = st.tabs([
    "🔬 Verify Query",
    "📊 Evaluation Results",
    "📂 Category Comparison",
    "🏗 Architecture",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 · VERIFY
# ─────────────────────────────────────────────────────────────────────────────
with tab_v:
    ci, cr = st.columns([1, 1], gap="large")

    with ci:
        st.markdown("#### ① Enter Query")
        question = st.text_area("Question", height=100,
            value="Who invented the World Wide Web and when?",
            placeholder="Enter a factual question…")

        st.markdown("**② Response to verify**")
        mode = st.radio("src", ["Type manually", "Auto-generate with Gemini"],
                        horizontal=True, label_visibility="collapsed")
        auto_gen = (mode == "Auto-generate with Gemini")

        response_text = ""
        if not auto_gen:
            response_text = st.text_area("Response", height=105, label_visibility="collapsed",
                value="Tim Berners-Lee invented the World Wide Web in 1989 at CERN in Geneva.",
                placeholder="Paste LLM response…")
        else:
            st.info("🤖 Gemini generates answers")

        go_btn = st.button("🔍 Verify Now", use_container_width=True)

    with cr:
        st.markdown("#### Verification Output")

        if go_btn:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Running: RAG → NLI → Judge…"):
                    t0  = time.time()
                    res = run_verify(question, response_text, auto_gen, use_judge)
                    elapsed = round(time.time()-t0, 2)

                res["elapsed"] = elapsed
                st.session_state["history"].insert(0, res)

                lbl   = res["final_label"]
                conf  = res["confidence"]
                emoji = {"CORRECT":"✅","HALLUCINATED":"🚨","UNCERTAIN":"❓"}[lbl]
                clr   = {"CORRECT":"#1a7a42","HALLUCINATED":"#c8401a","UNCERTAIN":"#b07c10"}[lbl]

                # ── Structured result card ──
                demo_note = (
                    "<div style='margin-top:.3rem;font-size:.73rem;color:var(--accent);"
                    "font-family:IBM Plex Mono,monospace;'>⚠ Demo mode — add GEMINI_API_KEY to .env</div>"
                    if res.get("demo_mode") else ""
                )
                auto_note = (
                    "<div style='margin-top:.25rem;font-size:.73rem;color:var(--muted);"
                    "font-family:IBM Plex Mono,monospace;'>★ Auto-generated by Gemini</div>"
                    if res.get("auto_generated") else ""
                )

                st.markdown(f"""
<div class="result-shell">
  <div class="result-header">
    <span style="font-size:1.35rem">{emoji}</span>
    <span class="verdict-label verdict-{lbl}">{lbl}</span>
    <span style="flex:1"></span>
    <span style="font-family:'IBM Plex Mono',monospace;font-size:.76rem;color:var(--muted)">⏱ {elapsed}s</span>
  </div>

  <div class="result-section">
    <div class="section-tag">① Answer</div>
    <div class="answer-text">{res["response"]}</div>
    {auto_note}{demo_note}
  </div>

  <div class="result-section">
    <div class="section-tag">② Verdict</div>
    <div style="display:flex;align-items:center;gap:.6rem;margin-top:.25rem;">
      <span class="verdict-label verdict-{lbl}" style="font-size:.82rem;padding:.3rem .9rem;">{emoji} {lbl}</span>
      <span style="font-size:.83rem;color:var(--muted);">{res.get("explanation","")}</span>
    </div>
  </div>

  <div class="result-section">
    <div class="section-tag">③ Confidence Score</div>
    <div style="display:flex;align-items:center;gap:.75rem;margin-top:.3rem;">
      <div style="font-family:'Fraunces',serif;font-size:2rem;font-weight:700;color:{clr};">{conf}%</div>
      <div style="flex:1;">
        <div class="conf-bar-track">
          <div class="conf-bar-fill" style="width:{conf}%;background:{clr};"></div>
        </div>
      </div>
    </div>
    <div style="display:flex;gap:1.5rem;margin-top:.55rem;font-family:'IBM Plex Mono',monospace;font-size:.78rem;">
      <div><div style="color:var(--muted);font-size:.62rem;margin-bottom:.08rem;">ENTAILMENT</div>
           <div style="color:var(--correct);font-weight:600;">{res.get("entailment_score",0):.3f}</div></div>
      <div><div style="color:var(--muted);font-size:.62rem;margin-bottom:.08rem;">CONTRADICTION</div>
           <div style="color:var(--halluc);font-weight:600;">{res.get("contradiction_score",0):.3f}</div></div>
      <div><div style="color:var(--muted);font-size:.62rem;margin-bottom:.08rem;">NLI LABEL</div>
           <div style="font-weight:600;">{res.get("nli_label","—")}</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

                # ── Evidence (④) ──
                st.markdown("##### ④ Evidence (Retrieved Facts)")
                passages = res.get("retrieved_passages", [])
                sources  = res.get("retrieved_sources",  [])
                if passages:
                    for p, src in zip(passages[:4], sources[:4]):
                        snippet = p[:340] + ("…" if len(p)>340 else "")
                        st.markdown(f"""
<div class="evidence-card">
  <div class="evidence-source">📖 {src}</div>
  {snippet}
</div>""", unsafe_allow_html=True)
                else:
                    st.info("No Wikipedia passages retrieved.")

        else:
            st.markdown("""
<div style="background:var(--card);border:1.5px dashed var(--border);border-radius:12px;
  padding:2.8rem 2rem;text-align:center;color:var(--muted);">
  <div style="font-size:2.3rem;margin-bottom:.6rem;">🧠</div>
  <div style="font-family:'Fraunces',serif;font-size:1.05rem;color:var(--ink);">
    Enter a question &amp; click <b>Verify Now</b>
  </div>
  <div style="font-size:.83rem;margin-top:.35rem;">
    Returns: Answer · Verdict · Confidence Score · Evidence
  </div>
</div>""", unsafe_allow_html=True)

    # ── History ──
    if st.session_state["history"]:
        st.markdown("---")
        st.markdown("#### 🕓 Session History")
        for h in st.session_state["history"][:8]:
            lbl   = h["final_label"]
            emoji = {"CORRECT":"✅","HALLUCINATED":"🚨","UNCERTAIN":"❓"}.get(lbl,"❓")
            clr   = {"CORRECT":"var(--correct)","HALLUCINATED":"var(--halluc)","UNCERTAIN":"var(--uncert)"}.get(lbl,"")
            st.markdown(f"""
<div class="hist-item">
  <span style="font-size:1.05rem">{emoji}</span>
  <div style="flex:1;overflow:hidden;">
    <div style="font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{h["question"][:80]}</div>
    <div style="font-size:.74rem;color:var(--muted);font-family:'IBM Plex Mono',monospace;">{h["response"][:60]}…</div>
  </div>
  <span style="font-family:'IBM Plex Mono',monospace;font-size:.71rem;font-weight:700;color:{clr};">
    {lbl} · {h["confidence"]}%
  </span>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 · EVALUATION RESULTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_e:
    if run_eval_btn:
        st.session_state["trigger_eval"] = n_eval

    if st.session_state.get("trigger_eval"):
        n = st.session_state.pop("trigger_eval")
        st.markdown(f"### Running TruthfulQA Evaluation — {n} samples")
        try:
            from evaluation.truthfulqa_loader import load_truthfulqa
            from evaluation.evaluator import Evaluator
            samples = load_truthfulqa(n_samples=n)
            pipe    = load_pipeline(use_judge)
            ev      = Evaluator(pipeline=pipe, output_dir="results")
            prog    = st.progress(0, text="Starting…")
            rows_l, y_true_l, y_pred_l = [], [], []
            for i, s in enumerate(samples):
                prog.progress((i+1)/len(samples), text=f"Sample {i+1}/{len(samples)}: {s['question'][:55]}…")
                r = pipe.verify(question=s["question"], response=s["response"])
                rows_l.append({"Question":s["question"][:65],"Response":s["response"][:60],
                                "True":s["true_label"],"Predicted":r.final_label,
                                "Conf%":r.confidence,"Match":"✅" if s["true_label"]==r.final_label else "❌"})
                y_true_l.append(s["true_label"]); y_pred_l.append(r.final_label)
            prog.empty()
            report = ev._compute_metrics(y_true_l, y_pred_l, rows_l)
            st.session_state["eval_report"] = report
            st.session_state["eval_rows"]   = rows_l
        except Exception as e:
            st.error(f"Evaluation failed: {e}"); st.exception(e)

    # Load report
    if "eval_report" in st.session_state:
        report = st.session_state["eval_report"]
        rows   = st.session_state.get("eval_rows", [])
    elif Path("results/metrics_report.json").exists():
        report = json.loads(Path("results/metrics_report.json").read_text())
        rows   = pd.read_csv("results/predictions.csv").to_dict("records") if Path("results/predictions.csv").exists() else []
    else:
        report = {"num_samples":100,"accuracy":.743,
                  "per_class":{
                      "CORRECT":     {"precision":.812,"recall":.791,"f1":.801,"support":52},
                      "HALLUCINATED":{"precision":.718,"recall":.745,"f1":.731,"support":38},
                      "UNCERTAIN":   {"precision":.624,"recall":.590,"f1":.607,"support":10},
                  },
                  "macro_avg":{"precision":.718,"recall":.709,"f1-score":.713},
                  "weighted_avg":{"precision":.758,"recall":.743,"f1-score":.750},
                  "confusion_matrix":[[41,8,3],[7,28,3],[2,2,6]],
                  "labels":["CORRECT","HALLUCINATED","UNCERTAIN"],"demo":True}
        rows = []
        st.info("📊 Demo results shown. Click **▶ Run Evaluation** in the sidebar for live TruthfulQA metrics.")

    # ── Top metrics ──
    st.markdown("### 📊 TruthfulQA Evaluation Metrics")
    if report.get("demo"): st.caption("⚠ Demo metrics")

    ma = report.get("macro_avg", {}); wa = report.get("weighted_avg",{})
    def pct(v): return f"{float(v)*100:.1f}%"

    m1,m2,m3,m4,m5 = st.columns(5)
    for col, val, lbl in [
        (m1, pct(report.get("accuracy",0)), "Accuracy"),
        (m2, f"{float(ma.get('precision',0)):.3f}", "Macro Precision"),
        (m3, f"{float(ma.get('recall',0)):.3f}",    "Macro Recall"),
        (m4, f"{float(ma.get('f1-score',0)):.3f}",  "Macro F1-Score"),
        (m5, str(report.get("num_samples",0)),       "Total Samples"),
    ]:
        col.markdown(f'<div class="eval-card"><div class="eval-val">{val}</div>'
                     f'<div class="eval-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([1,1.2], gap="large")
    LABELS_E = ["CORRECT","HALLUCINATED","UNCERTAIN"]
    COLOURS_E = {"CORRECT":"#1a7a42","HALLUCINATED":"#c8401a","UNCERTAIN":"#b07c10"}

    with col_l:
        st.markdown("#### Per-Class Precision / Recall / F1")
        tbl = [{"Category":l,
                "Precision": f"{report['per_class'].get(l,{}).get('precision',0):.4f}",
                "Recall":    f"{report['per_class'].get(l,{}).get('recall',0):.4f}",
                "F1-Score":  f"{report['per_class'].get(l,{}).get('f1',0):.4f}",
                "Support":    report['per_class'].get(l,{}).get('support','—')}
               for l in LABELS_E]
        st.dataframe(pd.DataFrame(tbl).set_index("Category"), width="stretch")

        st.markdown("#### Weighted Averages")
        st.dataframe(pd.DataFrame([
            {"Metric":"Precision","Score":f"{float(wa.get('precision',0)):.4f}"},
            {"Metric":"Recall",   "Score":f"{float(wa.get('recall',0)):.4f}"},
            {"Metric":"F1-Score", "Score":f"{float(wa.get('f1-score',0)):.4f}"},
        ]).set_index("Metric"), width='stretch')

    with col_r:
        st.markdown("#### Grouped Bar — Precision / Recall / F1")
        fig_bar = go.Figure()
        for lbl in LABELS_E:
            m = report["per_class"].get(lbl,{})
            vals = [m.get("precision",0), m.get("recall",0), m.get("f1",0)]
            fig_bar.add_trace(go.Bar(
                name=lbl, x=["Precision","Recall","F1-Score"], y=vals,
                marker_color=COLOURS_E[lbl],
                text=[f"{v:.3f}" for v in vals], textposition="outside",
                textfont=dict(size=9,family="IBM Plex Mono"),
            ))
        fig_bar.update_layout(
            barmode="group",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans",color="#1a1410"),
            legend=dict(orientation="h",y=-0.18,x=0),
            yaxis=dict(range=[0,1.18],gridcolor="#d9d2c5",tickformat=".2f"),
            xaxis=dict(gridcolor="#d9d2c5"),
            height=285,margin=dict(t=15,b=65,l=10,r=10),
        )
        st.plotly_chart(fig_bar, width='stretch')

    # ── Confusion matrix ──
    st.markdown("#### Confusion Matrix")
    import numpy as np
    cm_arr = np.array(report.get("confusion_matrix",[[0,0,0],[0,0,0],[0,0,0]]))
    lbls_cm = report.get("labels", LABELS_E)
    fig_cm = go.Figure(go.Heatmap(
        z=cm_arr, x=lbls_cm, y=lbls_cm,
        colorscale=[[0,"#f5f2eb"],[.5,"#f0a97a"],[1,"#c8401a"]],
        text=cm_arr, texttemplate="<b>%{text}</b>",
        textfont=dict(size=14,family="IBM Plex Mono"), showscale=True,
    ))
    fig_cm.update_layout(
        xaxis_title="Predicted", yaxis_title="True",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans",color="#1a1410"),
        height=310,margin=dict(t=15,b=50,l=80,r=20),
    )
    st.plotly_chart(fig_cm, width='stretch')

    if rows:
        st.markdown("#### Prediction Details")
        st.dataframe(pd.DataFrame(rows), width='stretch', height=270)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 · CATEGORY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab_c:
    st.markdown("### 📂 Hallucination Category Comparison")

    hist = st.session_state.get("history", [])
    CMAP = {"CORRECT":"#1a7a42","HALLUCINATED":"#c8401a","UNCERTAIN":"#b07c10"}

    if len(hist) >= 3:
        df_h   = pd.DataFrame(hist)
        counts = df_h["final_label"].value_counts().reset_index(); counts.columns=["Category","Count"]
        avgc   = df_h.groupby("final_label")["confidence"].mean().reset_index(); avgc.columns=["Category","Avg Conf"]
    else:
        counts = pd.DataFrame({"Category":["CORRECT","HALLUCINATED","UNCERTAIN"],"Count":[44,37,19]})
        avgc   = pd.DataFrame({"Category":["CORRECT","HALLUCINATED","UNCERTAIN"],"Avg Conf":[83.4,78.1,51.6]})
        st.info("Showing demo data. Verify more queries to see live category stats.")

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown("#### Verdict Distribution")
        colours = [CMAP.get(c,"#888") for c in counts["Category"]]
        fig_pie = go.Figure(go.Pie(
            labels=counts["Category"], values=counts["Count"], hole=.54,
            marker_colors=colours,
            textinfo="label+percent", textfont=dict(family="IBM Plex Mono",size=11),
            pull=[.04,.04,.04],
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans",color="#1a1410"),
            legend=dict(orientation="h",y=-0.1),
            height=295,margin=dict(t=15,b=40,l=15,r=15),
            annotations=[dict(text=f"<b>{counts['Count'].sum()}</b><br>Queries",
                              font=dict(size=13,family="Fraunces",color="#1a1410"),showarrow=False)],
        )
        st.plotly_chart(fig_pie, width='stretch')

    with c_right:
        st.markdown("#### Avg Confidence by Category")
        fig_conf = go.Figure(go.Bar(
            x=avgc["Category"], y=avgc["Avg Conf"],
            marker_color=[CMAP.get(c,"#888") for c in avgc["Category"]],
            text=[f"{v:.1f}%" for v in avgc["Avg Conf"]], textposition="outside",
            textfont=dict(size=11,family="IBM Plex Mono"),
        ))
        fig_conf.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans",color="#1a1410"),
            yaxis=dict(range=[0,110],gridcolor="#d9d2c5",title="Avg Confidence %"),
            xaxis=dict(gridcolor="#d9d2c5"),
            height=295,margin=dict(t=15,b=15,l=25,r=15),
        )
        st.plotly_chart(fig_conf, width='stretch')

    # ── Category definition cards ──
    st.markdown("#### Category Definitions")
    cc1,cc2,cc3 = st.columns(3)
    for col,(lbl,em,fg,bg,bd,desc) in zip([cc1,cc2,cc3],[
        ("CORRECT",     "✅","#1a7a42","#eef7f2","#a3d4b7",
         "NLI entailment ≥ 0.6 for at least one passage. The Judge LLM confirms the response aligns with retrieved Wikipedia facts."),
        ("HALLUCINATED","🚨","#c8401a","#fdf0ec","#f0b8a8",
         "NLI contradiction ≥ 0.6 and exceeds entailment. The model invented or distorted facts not supported by evidence."),
        ("UNCERTAIN",   "❓","#b07c10","#fdf6e8","#e8d09a",
         "Neither entailment nor contradiction threshold met. Evidence is insufficient to confirm or deny the claim."),
    ]):
        with col:
            st.markdown(f"""
<div style="background:{bg};border:1.5px solid {bd};border-radius:10px;padding:1.1rem;">
  <div style="font-size:1.45rem;margin-bottom:.35rem;">{em}</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:.69rem;font-weight:700;
    text-transform:uppercase;letter-spacing:.12em;color:{fg};margin-bottom:.45rem;">{lbl}</div>
  <div style="font-size:.83rem;line-height:1.58;color:#1a1410;">{desc}</div>
</div>""", unsafe_allow_html=True)

    # ── NLI scatter ──
    st.markdown("---")
    st.markdown("#### NLI Score Space — Entailment vs Contradiction")

    if len(hist) >= 3:
        df_sc = pd.DataFrame([{
            "Category":res["final_label"],
            "Entailment":res.get("entailment_score",0),
            "Contradiction":res.get("contradiction_score",0),
        } for res in hist])
    else:
        random.seed(42)
        sc_rows = []
        for lbl, er, cr_ in [("CORRECT",(.55,.95),(.02,.28)),
                              ("HALLUCINATED",(.05,.35),(.55,.92)),
                              ("UNCERTAIN",(.20,.54),(.20,.50))]:
            for _ in range(22):
                sc_rows.append({"Category":lbl,
                                 "Entailment":random.uniform(*er),
                                 "Contradiction":random.uniform(*cr_)})
        df_sc = pd.DataFrame(sc_rows)

    fig_sc = px.scatter(df_sc, x="Entailment", y="Contradiction", color="Category",
                        color_discrete_map=CMAP, opacity=.78, size_max=10)
    fig_sc.add_hline(y=.6, line_dash="dash", line_color="#7a6f62",
                     annotation_text="Contradiction threshold", annotation_position="right")
    fig_sc.add_vline(x=.6, line_dash="dash", line_color="#7a6f62",
                     annotation_text="Entailment threshold", annotation_position="top left")
    fig_sc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#1a1410"),
        legend=dict(orientation="h",y=-0.2),
        xaxis=dict(range=[0,1],gridcolor="#d9d2c5",title="Entailment Score"),
        yaxis=dict(range=[0,1],gridcolor="#d9d2c5",title="Contradiction Score"),
        height=360, margin=dict(t=20,b=80,l=20,r=20),
    )
    st.plotly_chart(fig_sc, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 · ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
with tab_a:
    st.markdown("### 🏗 End-to-End Pipeline Architecture")
    st.markdown("""
```
INPUT: Question (+ optional response)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 0 · STUDENT LLM                                                   │
│  Generates an initial answer → OUTPUT ①  Answer                         │
└────────────────────────┬────────────────────────────────────────────────┘
                         │ response to verify
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1 · RAG RETRIEVAL                                                 │
│  Query → Wikipedia API → sentence-transformers (all-MiniLM-L6-v2)      │
│  → FAISS IndexFlatIP cosine search → Top-K passages                    │
│                                   → OUTPUT ④  Evidence                 │
└────────────────────────┬────────────────────────────────────────────────┘
                         │ evidence passages
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2 · NLI CLASSIFICATION  (cross-encoder/nli-deberta-v3-base)       │
│  (response, passage) → entailment / contradiction / neutral scores      │
│  Aggregation:                                                           │
│    contradiction ≥ 0.6 AND > entailment  →  HALLUCINATED                │
│    entailment    ≥ 0.6                   →  CORRECT                     │
│    otherwise                             →  UNCERTAIN                   │
└────────────────────────┬────────────────────────────────────────────────┘
                         │ NLI signal + best evidence
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3 · JUDGE LLM                                                     │
│  Synthesises NLI scores + evidence + response                           │
│  → OUTPUT ②  Verdict  (CORRECT | HALLUCINATED | UNCERTAIN)              │
│  → OUTPUT ③  Confidence %  + natural language explanation               │
└─────────────────────────────────────────────────────────────────────────┘
```
""")

    st.markdown("---")
    la, lb = st.columns(2)
    with la:
        st.markdown("#### Tech Stack")
        st.dataframe(pd.DataFrame([
            ("Student / Judge LLM","google-genai · gemini-2.5-flash","Meta-LLM roles"),
            ("Embeddings","sentence-transformers · MiniLM","384-dim dense vectors"),
            ("Vector search","faiss-cpu · IndexFlatIP","Cosine similarity"),
            ("Knowledge source","Wikipedia API","Free, no key required"),
            ("NLI model","DeBERTa-v3-base (cross-encoder)","State-of-art NLI"),
            ("Benchmark","TruthfulQA (HuggingFace)","817 adversarial QA pairs"),
            ("Metrics","scikit-learn","Precision / Recall / F1 / CM"),
            ("Dashboard","Streamlit + Plotly","This app"),
        ], columns=["Component","Library","Notes"]), width='stretch', hide_index=True)
    with lb:
        st.markdown("#### Output Schema")
        st.markdown("""
| # | Field | Description |
|---|---|---|
| ① | **Answer** | The LLM response being verified |
| ② | **Verdict** | `CORRECT` · `HALLUCINATED` · `UNCERTAIN` |
| ③ | **Confidence** | 0–100 % from Judge LLM |
| ④ | **Evidence** | Top-K Wikipedia passages + sources |
| — | NLI scores | Entailment · Contradiction · Neutral |
| — | Explanation | Natural language reasoning |
""")
        st.markdown("#### Threshold Decisions")
        st.markdown("""
| Condition | Label |
|---|---|
| contradiction ≥ 0.6 AND > entailment | 🚨 HALLUCINATED |
| entailment ≥ 0.6 | ✅ CORRECT |
| neither threshold met | ❓ UNCERTAIN |
""")


# #2

# """
# app.py ─ Meta-LLM Hallucination Detection Dashboard
# ─────────────────────────────────────────────────────
# Run:  streamlit run app.py

# Outputs per verification:
#   ① Answer   — generated or user-supplied response
#   ② Verdict  — CORRECT | HALLUCINATED | UNCERTAIN
#   ③ Confidence Score (0-100 %)
#   ④ Evidence — retrieved Wikipedia facts

# Evaluation tab:
#   • Accuracy, Precision, Recall, F1-Score on TruthfulQA
#   • Hallucination category comparison + NLI scatter

# Category Comparison tab:
#   • Donut distribution, avg confidence, scatter plot
# """

# import json, time, random
# from pathlib import Path

# import plotly.graph_objects as go
# import plotly.express as px
# import pandas as pd
# import streamlit as st

# # ── Page config ─────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Meta-LLM · Hallucination Detector",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ══════════════════════════════════════════════════════════════════════════════
# # CSS
# # ══════════════════════════════════════════════════════════════════════════════
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,300&family=DM+Sans:wght@400;500;700&display=swap');

# :root{
#   --bg:#f5f2eb;--surface:#faf8f3;--ink:#1a1410;--muted:#7a6f62;
#   --border:#d9d2c5;--accent:#c8401a;
#   --correct:#1a7a42;--halluc:#c8401a;--uncert:#b07c10;
#   --correct-bg:#eef7f2;--halluc-bg:#fdf0ec;--uncert-bg:#fdf6e8;
#   --card:#ffffff;--shadow:rgba(26,20,16,.08);
# }
# html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{
#   background:var(--bg)!important;font-family:'DM Sans',sans-serif;color:var(--ink);}
# [data-testid="stSidebar"]{background:var(--ink)!important;border-right:none;}
# [data-testid="stSidebar"] *{color:#f5f2eb!important;}
# [data-testid="stSidebar"] a{color:#f0a97a!important;}
# [data-testid="stSidebar"] hr{border-color:#3a3028!important;}
# [data-testid="stSidebar"] label{color:#c8b8a8!important;font-size:.8rem!important;}
# h1,h2,h3{font-family:'Fraunces',serif;font-weight:700;color:var(--ink);}
# .stTextArea textarea,.stTextInput>div>div>input{
#   background:var(--card)!important;border:1.5px solid var(--border)!important;
#   border-radius:6px!important;color:var(--ink)!important;
#   font-family:'DM Sans',sans-serif!important;font-size:.93rem!important;}
# .stTextArea textarea:focus,.stTextInput>div>div>input:focus{
#   border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(200,64,26,.12)!important;}
# div[data-testid="stButton"]>button{
#   background:var(--ink)!important;color:#f5f2eb!important;border:none!important;
#   border-radius:6px!important;font-family:'IBM Plex Mono',monospace!important;
#   font-size:.78rem!important;font-weight:600!important;letter-spacing:.06em!important;
#   text-transform:uppercase!important;padding:.55rem 1.5rem!important;
#   transition:background .15s,transform .1s!important;}
# div[data-testid="stButton"]>button:hover{background:var(--accent)!important;transform:translateY(-1px)!important;}
# [data-testid="stTabs"] [role="tablist"]{border-bottom:2px solid var(--border);gap:0;}
# [data-testid="stTabs"] button[role="tab"]{
#   font-family:'IBM Plex Mono',monospace;font-size:.73rem;text-transform:uppercase;
#   letter-spacing:.09em;color:var(--muted)!important;padding:.5rem 1.1rem!important;
#   border-bottom:2px solid transparent!important;margin-bottom:-2px;}
# [data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
#   color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;font-weight:700!important;}
# [data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);
#   border-radius:8px;padding:.8rem 1rem!important;}
# [data-testid="stMetricLabel"]{font-size:.7rem!important;color:var(--muted)!important;
#   font-family:'IBM Plex Mono',monospace!important;text-transform:uppercase!important;}
# [data-testid="stMetricValue"]{font-family:'Fraunces',serif!important;font-size:1.65rem!important;color:var(--ink)!important;}
# [data-testid="stExpander"]{background:var(--card);border:1px solid var(--border)!important;border-radius:8px!important;}
# hr{border-color:var(--border)!important;margin:1.1rem 0!important;}

# .hero{padding:2.2rem 0 1.4rem;border-bottom:2px solid var(--ink);margin-bottom:1.8rem;}
# .hero-eyebrow{font-family:'IBM Plex Mono',monospace;font-size:.7rem;letter-spacing:.18em;
#   text-transform:uppercase;color:var(--accent);margin-bottom:.35rem;}
# .hero-title{font-family:'Fraunces',serif;font-size:2.8rem;font-weight:700;line-height:1.06;color:var(--ink);}
# .hero-sub{font-size:.93rem;color:var(--muted);margin-top:.5rem;max-width:520px;}

# .result-shell{border:2px solid var(--border);border-radius:12px;overflow:hidden;
#   margin:1.1rem 0;background:var(--card);box-shadow:0 2px 16px var(--shadow);}
# .result-header{padding:.9rem 1.3rem;border-bottom:1px solid var(--border);
#   display:flex;align-items:center;gap:.65rem;}
# .verdict-label{font-family:'IBM Plex Mono',monospace;font-size:.7rem;font-weight:700;
#   letter-spacing:.13em;text-transform:uppercase;padding:.22rem .65rem;border-radius:4px;}
# .verdict-CORRECT{background:var(--correct-bg);color:var(--correct);border:1px solid #a3d4b7;}
# .verdict-HALLUCINATED{background:var(--halluc-bg);color:var(--halluc);border:1px solid #f0b8a8;}
# .verdict-UNCERTAIN{background:var(--uncert-bg);color:var(--uncert);border:1px solid #e8d09a;}
# .result-section{padding:.85rem 1.3rem;border-bottom:1px solid var(--border);}
# .result-section:last-child{border-bottom:none;}
# .section-tag{font-family:'IBM Plex Mono',monospace;font-size:.64rem;letter-spacing:.15em;
#   text-transform:uppercase;color:var(--muted);margin-bottom:.32rem;}
# .answer-text{font-family:'Fraunces',serif;font-size:1.03rem;line-height:1.56;color:var(--ink);}
# .conf-bar-track{height:8px;background:var(--border);border-radius:4px;overflow:hidden;margin-top:.4rem;}
# .conf-bar-fill{height:100%;border-radius:4px;}

# .evidence-card{background:var(--bg);border:1px solid var(--border);
#   border-left:3px solid var(--accent);border-radius:0 6px 6px 0;
#   padding:.65rem .95rem;margin:.4rem 0;font-size:.83rem;line-height:1.54;color:var(--ink);}
# .evidence-source{font-family:'IBM Plex Mono',monospace;font-size:.63rem;
#   color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.22rem;}

# .eval-card{background:var(--card);border:1px solid var(--border);
#   border-radius:10px;padding:1.1rem 1.3rem;text-align:center;}
# .eval-val{font-family:'Fraunces',serif;font-size:2.1rem;font-weight:700;color:var(--ink);}
# .eval-lbl{font-family:'IBM Plex Mono',monospace;font-size:.67rem;text-transform:uppercase;
#   letter-spacing:.12em;color:var(--muted);margin-top:.2rem;}

# .hist-item{background:var(--card);border:1px solid var(--border);border-radius:8px;
#   padding:.7rem .95rem;margin:.38rem 0;font-size:.83rem;display:flex;align-items:center;gap:.7rem;}
# </style>
# """, unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # PIPELINE
# # ══════════════════════════════════════════════════════════════════════════════
# @st.cache_resource(show_spinner="Loading NLI model & RAG retriever…")
# def load_pipeline(use_judge: bool):
#     from pipeline.verifier_pipeline import VerifierPipeline
#     return VerifierPipeline(use_judge=use_judge)


# def _demo_result(question, response):
#     import hashlib
#     seed = int(hashlib.md5((question + response).encode()).hexdigest(), 16) % 9999
#     random.seed(seed)
#     lbl  = random.choices(["CORRECT","HALLUCINATED","UNCERTAIN"], [.44,.37,.19])[0]
#     conf = random.randint(63,96) if lbl != "UNCERTAIN" else random.randint(38,59)
#     ent  = round(random.uniform(.57,.93) if lbl=="CORRECT"      else random.uniform(.10,.38), 3)
#     con  = round(random.uniform(.57,.90) if lbl=="HALLUCINATED" else random.uniform(.05,.28), 3)
#     exps = {
#         "CORRECT":      "The response is well-supported by retrieved Wikipedia evidence. Key facts align.",
#         "HALLUCINATED": "The response contradicts retrieved evidence. Specific claims could not be verified.",
#         "UNCERTAIN":    "Evidence is ambiguous. Passages neither confirm nor deny the response.",
#     }
#     return {
#         "question":question,"response":response or f"[Auto-answer for: {question}]",
#         "final_label":lbl,"confidence":conf,"explanation":exps[lbl],
#         "nli_label":lbl,"entailment_score":ent,"contradiction_score":con,
#         "retrieved_passages":[
#             f"Historical records show that {question.rstrip('?')} is documented in academic literature.",
#             "Multiple encyclopaedic sources provide detailed information on this subject.",
#             "Peer-reviewed references confirm the established consensus on this topic.",
#         ],
#         "retrieved_sources":["Wikipedia (Demo)","Encyclopedia Britannica (Demo)","Academic DB (Demo)"],
#         "auto_generated":not bool(response.strip()),"demo_mode":True,
#     }


# def run_verify(question, response, auto_gen, use_judge):
#     import config as cfg
#     if not cfg.GEMINI_API_KEY:
#         return _demo_result(question, response)
#     try:
#         pipe = load_pipeline(use_judge)
#         r    = pipe.verify(question=question, response=response, generate_answer=auto_gen)
#         return r.to_dict()
#     except Exception as e:
#         st.warning(f"Pipeline error ({e}) — showing demo result.")
#         return _demo_result(question, response)


# # ══════════════════════════════════════════════════════════════════════════════
# # SESSION STATE
# # ══════════════════════════════════════════════════════════════════════════════
# if "history" not in st.session_state:
#     st.session_state["history"] = []


# # ══════════════════════════════════════════════════════════════════════════════
# # SIDEBAR
# # ══════════════════════════════════════════════════════════════════════════════
# with st.sidebar:
#     st.markdown("""
# <div style='font-family:"IBM Plex Mono",monospace;font-size:.63rem;letter-spacing:.18em;
#   text-transform:uppercase;color:#f0a97a;margin-bottom:.3rem;'>Meta-LLM System</div>
# <div style='font-family:"Fraunces",serif;font-size:1.25rem;font-weight:700;
#   line-height:1.2;margin-bottom:1.2rem;'>Hallucination<br>Detector</div>
# """, unsafe_allow_html=True)

#     st.markdown("#### ⚙️ Settings")
#     use_judge = st.toggle("Judge LLM (Gemini)", value=True,
#         help="Disable for NLI-only mode — no API key needed.")

#     st.markdown("---")
#     st.markdown("#### 🧪 TruthfulQA Evaluation")
#     n_eval = st.slider("Samples", 10, 150, 40, step=10)
#     run_eval_btn = st.button("▶ Run Evaluation", use_container_width=True)

#     st.markdown("---")
#     st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;font-size:.69rem;
# line-height:2;color:#c8b8a8;'>
# <b style='color:#f5f2eb;'>Dataset</b><br>
# <a href='https://huggingface.co/datasets/truthful_qa'>TruthfulQA ↗</a><br>
# <b style='color:#f5f2eb;'>NLI Model</b><br>
# <a href='https://huggingface.co/cross-encoder/nli-deberta-v3-base'>DeBERTa-v3-base ↗</a><br>
# <b style='color:#f5f2eb;'>Embeddings</b><br>
# <a href='https://www.sbert.net/'>sentence-transformers ↗</a><br>
# <b style='color:#f5f2eb;'>Vector DB</b><br>
# <a href='https://github.com/facebookresearch/faiss'>FAISS ↗</a><br>
# <b style='color:#f5f2eb;'>LLM API</b><br>
# <a href='https://aistudio.google.com/'>Google Gemini ↗</a>
# </div>""", unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # HERO
# # ══════════════════════════════════════════════════════════════════════════════
# st.markdown("""
# <div class="hero">
#   <div class="hero-eyebrow">RAG · NLI · LLM-as-Judge · TruthfulQA</div>
#   <div class="hero-title">Meta-LLM Hallucination<br>Detection System</div>
#   <div class="hero-sub">
#     End-to-end factual verification — retrieves Wikipedia evidence, runs NLI,
#     and delivers a verdict with confidence score and cited facts.
#   </div>
# </div>
# """, unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # TABS
# # ══════════════════════════════════════════════════════════════════════════════
# tab_v, tab_e, tab_c, tab_a = st.tabs([
#     "🔬 Verify Query",
#     "📊 Evaluation Results",
#     "📂 Category Comparison",
#     "🏗 Architecture",
# ])


# # ─────────────────────────────────────────────────────────────────────────────
# #  TAB 1 · VERIFY
# # ─────────────────────────────────────────────────────────────────────────────
# with tab_v:
#     ci, cr = st.columns([1, 1], gap="large")

#     with ci:
#         st.markdown("#### ① Enter Query")
#         question = st.text_area("Question", height=100,
#             value="Who invented the World Wide Web and when?",
#             placeholder="Enter a factual question…")

#         st.markdown("**② Response to verify**")
#         mode = st.radio("src", ["Type manually", "Auto-generate with Gemini"],
#                         horizontal=True, label_visibility="collapsed")
#         auto_gen = (mode == "Auto-generate with Gemini")

#         response_text = ""
#         if not auto_gen:
#             response_text = st.text_area("Response", height=105, label_visibility="collapsed",
#                 value="Tim Berners-Lee invented the World Wide Web in 1989 at CERN in Geneva.",
#                 placeholder="Paste LLM response…")
#         else:
#             st.info("🤖 Gemini generates then self-verifies an answer.")

#         go_btn = st.button("🔍 Verify Now", use_container_width=True)

#     with cr:
#         st.markdown("#### Verification Output")

#         if go_btn:
#             if not question.strip():
#                 st.warning("Please enter a question.")
#             else:
#                 with st.spinner("Running: RAG → NLI → Judge…"):
#                     t0  = time.time()
#                     res = run_verify(question, response_text, auto_gen, use_judge)
#                     elapsed = round(time.time()-t0, 2)

#                 res["elapsed"] = elapsed
#                 st.session_state["history"].insert(0, res)

#                 lbl   = res["final_label"]
#                 conf  = res["confidence"]
#                 emoji = {"CORRECT":"✅","HALLUCINATED":"🚨","UNCERTAIN":"❓"}[lbl]
#                 clr   = {"CORRECT":"#1a7a42","HALLUCINATED":"#c8401a","UNCERTAIN":"#b07c10"}[lbl]

#                 # ── Structured result card ──
#                 demo_note = (
#                     "<div style='margin-top:.3rem;font-size:.73rem;color:var(--accent);"
#                     "font-family:IBM Plex Mono,monospace;'>⚠ Demo mode — add GEMINI_API_KEY to .env</div>"
#                     if res.get("demo_mode") else ""
#                 )
#                 auto_note = (
#                     "<div style='margin-top:.25rem;font-size:.73rem;color:var(--muted);"
#                     "font-family:IBM Plex Mono,monospace;'>★ Auto-generated by Gemini</div>"
#                     if res.get("auto_generated") else ""
#                 )

#                 st.markdown(f"""
# <div class="result-shell">
#   <div class="result-header">
#     <span style="font-size:1.35rem">{emoji}</span>
#     <span class="verdict-label verdict-{lbl}">{lbl}</span>
#     <span style="flex:1"></span>
#     <span style="font-family:'IBM Plex Mono',monospace;font-size:.76rem;color:var(--muted)">⏱ {elapsed}s</span>
#   </div>

#   <div class="result-section">
#     <div class="section-tag">① Answer</div>
#     <div class="answer-text">{res["response"]}</div>
#     {auto_note}{demo_note}
#   </div>

#   <div class="result-section">
#     <div class="section-tag">② Verdict</div>
#     <div style="display:flex;align-items:center;gap:.6rem;margin-top:.25rem;">
#       <span class="verdict-label verdict-{lbl}" style="font-size:.82rem;padding:.3rem .9rem;">{emoji} {lbl}</span>
#       <span style="font-size:.83rem;color:var(--muted);">{res.get("explanation","")}</span>
#     </div>
#   </div>

#   <div class="result-section">
#     <div class="section-tag">③ Confidence Score</div>
#     <div style="display:flex;align-items:center;gap:.75rem;margin-top:.3rem;">
#       <div style="font-family:'Fraunces',serif;font-size:2rem;font-weight:700;color:{clr};">{conf}%</div>
#       <div style="flex:1;">
#         <div class="conf-bar-track">
#           <div class="conf-bar-fill" style="width:{conf}%;background:{clr};"></div>
#         </div>
#       </div>
#     </div>
#     <div style="display:flex;gap:1.5rem;margin-top:.55rem;font-family:'IBM Plex Mono',monospace;font-size:.78rem;">
#       <div><div style="color:var(--muted);font-size:.62rem;margin-bottom:.08rem;">ENTAILMENT</div>
#            <div style="color:var(--correct);font-weight:600;">{res.get("entailment_score",0):.3f}</div></div>
#       <div><div style="color:var(--muted);font-size:.62rem;margin-bottom:.08rem;">CONTRADICTION</div>
#            <div style="color:var(--halluc);font-weight:600;">{res.get("contradiction_score",0):.3f}</div></div>
#       <div><div style="color:var(--muted);font-size:.62rem;margin-bottom:.08rem;">NLI LABEL</div>
#            <div style="font-weight:600;">{res.get("nli_label","—")}</div></div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#                 # ── Evidence (④) ──
#                 st.markdown("##### ④ Evidence (Retrieved Facts)")
#                 passages = res.get("retrieved_passages", [])
#                 sources  = res.get("retrieved_sources",  [])
#                 if passages:
#                     for p, src in zip(passages[:4], sources[:4]):
#                         snippet = p[:340] + ("…" if len(p)>340 else "")
#                         st.markdown(f"""
# <div class="evidence-card">
#   <div class="evidence-source">📖 {src}</div>
#   {snippet}
# </div>""", unsafe_allow_html=True)
#                 else:
#                     st.info("No Wikipedia passages retrieved.")

#         else:
#             st.markdown("""
# <div style="background:var(--card);border:1.5px dashed var(--border);border-radius:12px;
#   padding:2.8rem 2rem;text-align:center;color:var(--muted);">
#   <div style="font-size:2.3rem;margin-bottom:.6rem;">🧠</div>
#   <div style="font-family:'Fraunces',serif;font-size:1.05rem;color:var(--ink);">
#     Enter a question &amp; click <b>Verify Now</b>
#   </div>
#   <div style="font-size:.83rem;margin-top:.35rem;">
#     Returns: Answer · Verdict · Confidence Score · Evidence
#   </div>
# </div>""", unsafe_allow_html=True)

#     # ── History ──
#     if st.session_state["history"]:
#         st.markdown("---")
#         st.markdown("#### 🕓 Session History")
#         for h in st.session_state["history"][:8]:
#             lbl   = h["final_label"]
#             emoji = {"CORRECT":"✅","HALLUCINATED":"🚨","UNCERTAIN":"❓"}.get(lbl,"❓")
#             clr   = {"CORRECT":"var(--correct)","HALLUCINATED":"var(--halluc)","UNCERTAIN":"var(--uncert)"}.get(lbl,"")
#             st.markdown(f"""
# <div class="hist-item">
#   <span style="font-size:1.05rem">{emoji}</span>
#   <div style="flex:1;overflow:hidden;">
#     <div style="font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{h["question"][:80]}</div>
#     <div style="font-size:.74rem;color:var(--muted);font-family:'IBM Plex Mono',monospace;">{h["response"][:60]}…</div>
#   </div>
#   <span style="font-family:'IBM Plex Mono',monospace;font-size:.71rem;font-weight:700;color:{clr};">
#     {lbl} · {h["confidence"]}%
#   </span>
# </div>""", unsafe_allow_html=True)


# # ─────────────────────────────────────────────────────────────────────────────
# #  TAB 2 · EVALUATION RESULTS
# # ─────────────────────────────────────────────────────────────────────────────
# with tab_e:
#     if run_eval_btn:
#         st.session_state["trigger_eval"] = n_eval

#     if st.session_state.get("trigger_eval"):
#         n = st.session_state.pop("trigger_eval")
#         st.markdown(f"### Running TruthfulQA Evaluation — {n} samples")
#         try:
#             from evaluation.truthfulqa_loader import load_truthfulqa
#             from evaluation.evaluator import Evaluator
#             samples = load_truthfulqa(n_samples=n)
#             pipe    = load_pipeline(use_judge)
#             ev      = Evaluator(pipeline=pipe, output_dir="results")
#             prog    = st.progress(0, text="Starting…")
#             rows_l, y_true_l, y_pred_l = [], [], []
#             for i, s in enumerate(samples):
#                 prog.progress((i+1)/len(samples), text=f"Sample {i+1}/{len(samples)}: {s['question'][:55]}…")
#                 r = pipe.verify(question=s["question"], response=s["response"])
#                 rows_l.append({"Question":s["question"][:65],"Response":s["response"][:60],
#                                 "True":s["true_label"],"Predicted":r.final_label,
#                                 "Conf%":r.confidence,"Match":"✅" if s["true_label"]==r.final_label else "❌"})
#                 y_true_l.append(s["true_label"]); y_pred_l.append(r.final_label)
#             prog.empty()
#             report = ev._compute_metrics(y_true_l, y_pred_l, rows_l)
#             st.session_state["eval_report"] = report
#             st.session_state["eval_rows"]   = rows_l
#         except Exception as e:
#             st.error(f"Evaluation failed: {e}"); st.exception(e)

#     # Load report
#     if "eval_report" in st.session_state:
#         report = st.session_state["eval_report"]
#         rows   = st.session_state.get("eval_rows", [])
#     elif Path("results/metrics_report.json").exists():
#         report = json.loads(Path("results/metrics_report.json").read_text())
#         rows   = pd.read_csv("results/predictions.csv").to_dict("records") if Path("results/predictions.csv").exists() else []
#     else:
#         report = {"num_samples":100,"accuracy":.743,
#                   "per_class":{
#                       "CORRECT":     {"precision":.812,"recall":.791,"f1":.801,"support":52},
#                       "HALLUCINATED":{"precision":.718,"recall":.745,"f1":.731,"support":38},
#                       "UNCERTAIN":   {"precision":.624,"recall":.590,"f1":.607,"support":10},
#                   },
#                   "macro_avg":{"precision":.718,"recall":.709,"f1-score":.713},
#                   "weighted_avg":{"precision":.758,"recall":.743,"f1-score":.750},
#                   "confusion_matrix":[[41,8,3],[7,28,3],[2,2,6]],
#                   "labels":["CORRECT","HALLUCINATED","UNCERTAIN"],"demo":True}
#         rows = []
#         st.info("📊 Demo results shown. Click **▶ Run Evaluation** in the sidebar for live TruthfulQA metrics.")

#     # ── Top metrics ──
#     st.markdown("### 📊 TruthfulQA Evaluation Metrics")
#     if report.get("demo"): st.caption("⚠ Demo metrics")

#     ma = report.get("macro_avg", {}); wa = report.get("weighted_avg",{})
#     def pct(v): return f"{float(v)*100:.1f}%"

#     m1,m2,m3,m4,m5 = st.columns(5)
#     for col, val, lbl in [
#         (m1, pct(report.get("accuracy",0)), "Accuracy"),
#         (m2, f"{float(ma.get('precision',0)):.3f}", "Macro Precision"),
#         (m3, f"{float(ma.get('recall',0)):.3f}",    "Macro Recall"),
#         (m4, f"{float(ma.get('f1-score',0)):.3f}",  "Macro F1-Score"),
#         (m5, str(report.get("num_samples",0)),       "Total Samples"),
#     ]:
#         col.markdown(f'<div class="eval-card"><div class="eval-val">{val}</div>'
#                      f'<div class="eval-lbl">{lbl}</div></div>', unsafe_allow_html=True)

#     st.markdown("---")
#     col_l, col_r = st.columns([1,1.2], gap="large")
#     LABELS_E = ["CORRECT","HALLUCINATED","UNCERTAIN"]
#     COLOURS_E = {"CORRECT":"#1a7a42","HALLUCINATED":"#c8401a","UNCERTAIN":"#b07c10"}

#     with col_l:
#         st.markdown("#### Per-Class Precision / Recall / F1")
#         tbl = [{"Category":l,
#                 "Precision": f"{report['per_class'].get(l,{}).get('precision',0):.4f}",
#                 "Recall":    f"{report['per_class'].get(l,{}).get('recall',0):.4f}",
#                 "F1-Score":  f"{report['per_class'].get(l,{}).get('f1',0):.4f}",
#                 "Support":    report['per_class'].get(l,{}).get('support','—')}
#                for l in LABELS_E]
#         st.dataframe(pd.DataFrame(tbl).set_index("Category"), use_container_width=True)

#         st.markdown("#### Weighted Averages")
#         st.dataframe(pd.DataFrame([
#             {"Metric":"Precision","Score":f"{float(wa.get('precision',0)):.4f}"},
#             {"Metric":"Recall",   "Score":f"{float(wa.get('recall',0)):.4f}"},
#             {"Metric":"F1-Score", "Score":f"{float(wa.get('f1-score',0)):.4f}"},
#         ]).set_index("Metric"), use_container_width=True)

#     with col_r:
#         st.markdown("#### Grouped Bar — Precision / Recall / F1")
#         fig_bar = go.Figure()
#         for lbl in LABELS_E:
#             m = report["per_class"].get(lbl,{})
#             vals = [m.get("precision",0), m.get("recall",0), m.get("f1",0)]
#             fig_bar.add_trace(go.Bar(
#                 name=lbl, x=["Precision","Recall","F1-Score"], y=vals,
#                 marker_color=COLOURS_E[lbl],
#                 text=[f"{v:.3f}" for v in vals], textposition="outside",
#                 textfont=dict(size=9,family="IBM Plex Mono"),
#             ))
#         fig_bar.update_layout(
#             barmode="group",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
#             font=dict(family="DM Sans",color="#1a1410"),
#             legend=dict(orientation="h",y=-0.18,x=0),
#             yaxis=dict(range=[0,1.18],gridcolor="#d9d2c5",tickformat=".2f"),
#             xaxis=dict(gridcolor="#d9d2c5"),
#             height=285,margin=dict(t=15,b=65,l=10,r=10),
#         )
#         st.plotly_chart(fig_bar, use_container_width=True)

#     # ── Confusion matrix ──
#     st.markdown("#### Confusion Matrix")
#     import numpy as np
#     cm_arr = np.array(report.get("confusion_matrix",[[0,0,0],[0,0,0],[0,0,0]]))
#     lbls_cm = report.get("labels", LABELS_E)
#     fig_cm = go.Figure(go.Heatmap(
#         z=cm_arr, x=lbls_cm, y=lbls_cm,
#         colorscale=[[0,"#f5f2eb"],[.5,"#f0a97a"],[1,"#c8401a"]],
#         text=cm_arr, texttemplate="<b>%{text}</b>",
#         textfont=dict(size=14,family="IBM Plex Mono"), showscale=True,
#     ))
#     fig_cm.update_layout(
#         xaxis_title="Predicted", yaxis_title="True",
#         paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
#         font=dict(family="DM Sans",color="#1a1410"),
#         height=310,margin=dict(t=15,b=50,l=80,r=20),
#     )
#     st.plotly_chart(fig_cm, use_container_width=True)

#     if rows:
#         st.markdown("#### Prediction Details")
#         st.dataframe(pd.DataFrame(rows), use_container_width=True, height=270)


# # ─────────────────────────────────────────────────────────────────────────────
# #  TAB 3 · CATEGORY COMPARISON
# # ─────────────────────────────────────────────────────────────────────────────
# with tab_c:
#     st.markdown("### 📂 Hallucination Category Comparison")

#     hist = st.session_state.get("history", [])
#     CMAP = {"CORRECT":"#1a7a42","HALLUCINATED":"#c8401a","UNCERTAIN":"#b07c10"}

#     if len(hist) >= 3:
#         df_h   = pd.DataFrame(hist)
#         counts = df_h["final_label"].value_counts().reset_index(); counts.columns=["Category","Count"]
#         avgc   = df_h.groupby("final_label")["confidence"].mean().reset_index(); avgc.columns=["Category","Avg Conf"]
#     else:
#         counts = pd.DataFrame({"Category":["CORRECT","HALLUCINATED","UNCERTAIN"],"Count":[44,37,19]})
#         avgc   = pd.DataFrame({"Category":["CORRECT","HALLUCINATED","UNCERTAIN"],"Avg Conf":[83.4,78.1,51.6]})
#         st.info("Showing demo data. Verify more queries to see live category stats.")

#     c_left, c_right = st.columns(2, gap="large")

#     with c_left:
#         st.markdown("#### Verdict Distribution")
#         colours = [CMAP.get(c,"#888") for c in counts["Category"]]
#         fig_pie = go.Figure(go.Pie(
#             labels=counts["Category"], values=counts["Count"], hole=.54,
#             marker_colors=colours,
#             textinfo="label+percent", textfont=dict(family="IBM Plex Mono",size=11),
#             pull=[.04,.04,.04],
#         ))
#         fig_pie.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
#             font=dict(family="DM Sans",color="#1a1410"),
#             legend=dict(orientation="h",y=-0.1),
#             height=295,margin=dict(t=15,b=40,l=15,r=15),
#             annotations=[dict(text=f"<b>{counts['Count'].sum()}</b><br>Queries",
#                               font=dict(size=13,family="Fraunces",color="#1a1410"),showarrow=False)],
#         )
#         st.plotly_chart(fig_pie, use_container_width=True)

#     with c_right:
#         st.markdown("#### Avg Confidence by Category")
#         fig_conf = go.Figure(go.Bar(
#             x=avgc["Category"], y=avgc["Avg Conf"],
#             marker_color=[CMAP.get(c,"#888") for c in avgc["Category"]],
#             text=[f"{v:.1f}%" for v in avgc["Avg Conf"]], textposition="outside",
#             textfont=dict(size=11,family="IBM Plex Mono"),
#         ))
#         fig_conf.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
#             font=dict(family="DM Sans",color="#1a1410"),
#             yaxis=dict(range=[0,110],gridcolor="#d9d2c5",title="Avg Confidence %"),
#             xaxis=dict(gridcolor="#d9d2c5"),
#             height=295,margin=dict(t=15,b=15,l=25,r=15),
#         )
#         st.plotly_chart(fig_conf, use_container_width=True)

#     # ── Category definition cards ──
#     st.markdown("#### Category Definitions")
#     cc1,cc2,cc3 = st.columns(3)
#     for col,(lbl,em,fg,bg,bd,desc) in zip([cc1,cc2,cc3],[
#         ("CORRECT",     "✅","#1a7a42","#eef7f2","#a3d4b7",
#          "NLI entailment ≥ 0.6 for at least one passage. The Judge LLM confirms the response aligns with retrieved Wikipedia facts."),
#         ("HALLUCINATED","🚨","#c8401a","#fdf0ec","#f0b8a8",
#          "NLI contradiction ≥ 0.6 and exceeds entailment. The model invented or distorted facts not supported by evidence."),
#         ("UNCERTAIN",   "❓","#b07c10","#fdf6e8","#e8d09a",
#          "Neither entailment nor contradiction threshold met. Evidence is insufficient to confirm or deny the claim."),
#     ]):
#         with col:
#             st.markdown(f"""
# <div style="background:{bg};border:1.5px solid {bd};border-radius:10px;padding:1.1rem;">
#   <div style="font-size:1.45rem;margin-bottom:.35rem;">{em}</div>
#   <div style="font-family:'IBM Plex Mono',monospace;font-size:.69rem;font-weight:700;
#     text-transform:uppercase;letter-spacing:.12em;color:{fg};margin-bottom:.45rem;">{lbl}</div>
#   <div style="font-size:.83rem;line-height:1.58;color:#1a1410;">{desc}</div>
# </div>""", unsafe_allow_html=True)

#     # ── NLI scatter ──
#     st.markdown("---")
#     st.markdown("#### NLI Score Space — Entailment vs Contradiction")

#     if len(hist) >= 3:
#         df_sc = pd.DataFrame([{
#             "Category":res["final_label"],
#             "Entailment":res.get("entailment_score",0),
#             "Contradiction":res.get("contradiction_score",0),
#         } for res in hist])
#     else:
#         random.seed(42)
#         sc_rows = []
#         for lbl, er, cr_ in [("CORRECT",(.55,.95),(.02,.28)),
#                               ("HALLUCINATED",(.05,.35),(.55,.92)),
#                               ("UNCERTAIN",(.20,.54),(.20,.50))]:
#             for _ in range(22):
#                 sc_rows.append({"Category":lbl,
#                                  "Entailment":random.uniform(*er),
#                                  "Contradiction":random.uniform(*cr_)})
#         df_sc = pd.DataFrame(sc_rows)

#     fig_sc = px.scatter(df_sc, x="Entailment", y="Contradiction", color="Category",
#                         color_discrete_map=CMAP, opacity=.78, size_max=10)
#     fig_sc.add_hline(y=.6, line_dash="dash", line_color="#7a6f62",
#                      annotation_text="Contradiction threshold", annotation_position="right")
#     fig_sc.add_vline(x=.6, line_dash="dash", line_color="#7a6f62",
#                      annotation_text="Entailment threshold", annotation_position="top left")
#     fig_sc.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#         font=dict(family="DM Sans", color="#1a1410"),
#         legend=dict(orientation="h",y=-0.2),
#         xaxis=dict(range=[0,1],gridcolor="#d9d2c5",title="Entailment Score"),
#         yaxis=dict(range=[0,1],gridcolor="#d9d2c5",title="Contradiction Score"),
#         height=360, margin=dict(t=20,b=80,l=20,r=20),
#     )
#     st.plotly_chart(fig_sc, use_container_width=True)


# # ─────────────────────────────────────────────────────────────────────────────
# #  TAB 4 · ARCHITECTURE
# # ─────────────────────────────────────────────────────────────────────────────
# with tab_a:
#     st.markdown("### 🏗 End-to-End Pipeline Architecture")
#     st.markdown("""
# ```
# INPUT: Question (+ optional response)
#          │
#          ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  STEP 0 · STUDENT LLM  (Gemini 2.5 Flash)         [optional]           │
# │  Generates an initial answer → OUTPUT ①  Answer                        │
# └────────────────────────┬────────────────────────────────────────────────┘
#                          │ response to verify
#                          ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  STEP 1 · RAG RETRIEVAL                                                 │
# │  Query → Wikipedia API → sentence-transformers (all-MiniLM-L6-v2)      │
# │  → FAISS IndexFlatIP cosine search → Top-K passages                    │
# │                                   → OUTPUT ④  Evidence                 │
# └────────────────────────┬────────────────────────────────────────────────┘
#                          │ evidence passages
#                          ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  STEP 2 · NLI CLASSIFICATION  (cross-encoder/nli-deberta-v3-base)      │
# │  (response, passage) → entailment / contradiction / neutral scores      │
# │  Aggregation:                                                           │
# │    contradiction ≥ 0.6 AND > entailment  →  HALLUCINATED               │
# │    entailment    ≥ 0.6                   →  CORRECT                    │
# │    otherwise                             →  UNCERTAIN                  │
# └────────────────────────┬────────────────────────────────────────────────┘
#                          │ NLI signal + best evidence
#                          ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  STEP 3 · JUDGE LLM  (Gemini 2.5 Flash)                                │
# │  Synthesises NLI scores + evidence + response                           │
# │  → OUTPUT ②  Verdict  (CORRECT | HALLUCINATED | UNCERTAIN)            │
# │  → OUTPUT ③  Confidence %  + natural language explanation              │
# └─────────────────────────────────────────────────────────────────────────┘
# ```
# """)

#     st.markdown("---")
#     la, lb = st.columns(2)
#     with la:
#         st.markdown("#### Tech Stack")
#         st.dataframe(pd.DataFrame([
#             ("Student / Judge LLM","google-genai · gemini-2.5-flash","Meta-LLM roles"),
#             ("Embeddings","sentence-transformers · MiniLM","384-dim dense vectors"),
#             ("Vector search","faiss-cpu · IndexFlatIP","Cosine similarity"),
#             ("Knowledge source","Wikipedia API","Free, no key required"),
#             ("NLI model","DeBERTa-v3-base (cross-encoder)","State-of-art NLI"),
#             ("Benchmark","TruthfulQA (HuggingFace)","817 adversarial QA pairs"),
#             ("Metrics","scikit-learn","Precision / Recall / F1 / CM"),
#             ("Dashboard","Streamlit + Plotly","This app"),
#         ], columns=["Component","Library","Notes"]), use_container_width=True, hide_index=True)
#     with lb:
#         st.markdown("#### Output Schema")
#         st.markdown("""
# | # | Field | Description |
# |---|---|---|
# | ① | **Answer** | The LLM response being verified |
# | ② | **Verdict** | `CORRECT` · `HALLUCINATED` · `UNCERTAIN` |
# | ③ | **Confidence** | 0–100 % from Judge LLM |
# | ④ | **Evidence** | Top-K Wikipedia passages + sources |
# | — | NLI scores | Entailment · Contradiction · Neutral |
# | — | Explanation | Natural language reasoning |
# """)
#         st.markdown("#### Threshold Decisions")
#         st.markdown("""
# | Condition | Label |
# |---|---|
# | contradiction ≥ 0.6 AND > entailment | 🚨 HALLUCINATED |
# | entailment ≥ 0.6 | ✅ CORRECT |
# | neither threshold met | ❓ UNCERTAIN |
# """)










# """
# app.py ─ Meta-LLM Hallucination Detection Dashboard
# ─────────────────────────────────────────────────────
# Run:  streamlit run app.py

# Outputs per verification:
#   ① Answer   — generated or user-supplied response
#   ② Verdict  — CORRECT | HALLUCINATED | UNCERTAIN
#   ③ Confidence Score (0-100 %)
#   ④ Evidence — retrieved Wikipedia facts

# Evaluation tab:
#   • Accuracy, Precision, Recall, F1-Score on TruthfulQA
#   • Hallucination category comparison + NLI scatter

# Category Comparison tab:
#   • Donut distribution, avg confidence, scatter plot
# """

# import json, time, random
# from pathlib import Path

# import plotly.graph_objects as go
# import plotly.express as px
# import pandas as pd
# import streamlit as st

# # ── Page config ─────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Meta-LLM · Hallucination Detector",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ══════════════════════════════════════════════════════════════════════════════
# # CSS
# # ══════════════════════════════════════════════════════════════════════════════
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,300&family=DM+Sans:wght@400;500;700&display=swap');

# :root{
#   --bg:#f5f2eb;--surface:#faf8f3;--ink:#1a1410;--muted:#7a6f62;
#   --border:#d9d2c5;--accent:#c8401a;
#   --correct:#1a7a42;--halluc:#c8401a;--uncert:#b07c10;
#   --correct-bg:#eef7f2;--halluc-bg:#fdf0ec;--uncert-bg:#fdf6e8;
#   --card:#ffffff;--shadow:rgba(26,20,16,.08);
# }
# html,body,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{
#   background:var(--bg)!important;font-family:'DM Sans',sans-serif;color:var(--ink);}
# [data-testid="stSidebar"]{background:var(--ink)!important;border-right:none;}
# [data-testid="stSidebar"] *{color:#f5f2eb!important;}
# [data-testid="stSidebar"] a{color:#f0a97a!important;}
# [data-testid="stSidebar"] hr{border-color:#3a3028!important;}
# [data-testid="stSidebar"] label{color:#c8b8a8!important;font-size:.8rem!important;}
# h1,h2,h3{font-family:'Fraunces',serif;font-weight:700;color:var(--ink);}
# .stTextArea textarea,.stTextInput>div>div>input{
#   background:var(--card)!important;border:1.5px solid var(--border)!important;
#   border-radius:6px!important;color:var(--ink)!important;
#   font-family:'DM Sans',sans-serif!important;font-size:.93rem!important;}
# .stTextArea textarea:focus,.stTextInput>div>div>input:focus{
#   border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(200,64,26,.12)!important;}
# div[data-testid="stButton"]>button{
#   background:var(--ink)!important;color:#f5f2eb!important;border:none!important;
#   border-radius:6px!important;font-family:'IBM Plex Mono',monospace!important;
#   font-size:.78rem!important;font-weight:600!important;letter-spacing:.06em!important;
#   text-transform:uppercase!important;padding:.55rem 1.5rem!important;
#   transition:background .15s,transform .1s!important;}
# div[data-testid="stButton"]>button:hover{background:var(--accent)!important;transform:translateY(-1px)!important;}
# [data-testid="stTabs"] [role="tablist"]{border-bottom:2px solid var(--border);gap:0;}
# [data-testid="stTabs"] button[role="tab"]{
#   font-family:'IBM Plex Mono',monospace;font-size:.73rem;text-transform:uppercase;
#   letter-spacing:.09em;color:var(--muted)!important;padding:.5rem 1.1rem!important;
#   border-bottom:2px solid transparent!important;margin-bottom:-2px;}
# [data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
#   color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;font-weight:700!important;}
# [data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);
#   border-radius:8px;padding:.8rem 1rem!important;}
# [data-testid="stMetricLabel"]{font-size:.7rem!important;color:var(--muted)!important;
#   font-family:'IBM Plex Mono',monospace!important;text-transform:uppercase!important;}
# [data-testid="stMetricValue"]{font-family:'Fraunces',serif!important;font-size:1.65rem!important;color:var(--ink)!important;}
# [data-testid="stExpander"]{background:var(--card);border:1px solid var(--border)!important;border-radius:8px!important;}
# hr{border-color:var(--border)!important;margin:1.1rem 0!important;}

# .hero{padding:2.2rem 0 1.4rem;border-bottom:2px solid var(--ink);margin-bottom:1.8rem;}
# .hero-eyebrow{font-family:'IBM Plex Mono',monospace;font-size:.7rem;letter-spacing:.18em;
#   text-transform:uppercase;color:var(--accent);margin-bottom:.35rem;}
# .hero-title{font-family:'Fraunces',serif;font-size:2.8rem;font-weight:700;line-height:1.06;color:var(--ink);}
# .hero-sub{font-size:.93rem;color:var(--muted);margin-top:.5rem;max-width:520px;}

# .result-shell{border:2px solid var(--border);border-radius:12px;overflow:hidden;
#   margin:1.1rem 0;background:var(--card);box-shadow:0 2px 16px var(--shadow);}
# .result-header{padding:.9rem 1.3rem;border-bottom:1px solid var(--border);
#   display:flex;align-items:center;gap:.65rem;}
# .verdict-label{font-family:'IBM Plex Mono',monospace;font-size:.7rem;font-weight:700;
#   letter-spacing:.13em;text-transform:uppercase;padding:.22rem .65rem;border-radius:4px;}
# .verdict-CORRECT{background:var(--correct-bg);color:var(--correct);border:1px solid #a3d4b7;}
# .verdict-HALLUCINATED{background:var(--halluc-bg);color:var(--halluc);border:1px solid #f0b8a8;}
# .verdict-UNCERTAIN{background:var(--uncert-bg);color:var(--uncert);border:1px solid #e8d09a;}
# .result-section{padding:.85rem 1.3rem;border-bottom:1px solid var(--border);}
# .result-section:last-child{border-bottom:none;}
# .section-tag{font-family:'IBM Plex Mono',monospace;font-size:.64rem;letter-spacing:.15em;
#   text-transform:uppercase;color:var(--muted);margin-bottom:.32rem;}
# .answer-text{font-family:'Fraunces',serif;font-size:1.03rem;line-height:1.56;color:var(--ink);}
# .conf-bar-track{height:8px;background:var(--border);border-radius:4px;overflow:hidden;margin-top:.4rem;}
# .conf-bar-fill{height:100%;border-radius:4px;}

# .evidence-card{background:var(--bg);border:1px solid var(--border);
#   border-left:3px solid var(--accent);border-radius:0 6px 6px 0;
#   padding:.65rem .95rem;margin:.4rem 0;font-size:.83rem;line-height:1.54;color:var(--ink);}
# .evidence-source{font-family:'IBM Plex Mono',monospace;font-size:.63rem;
#   color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.22rem;}

# .eval-card{background:var(--card);border:1px solid var(--border);
#   border-radius:10px;padding:1.1rem 1.3rem;text-align:center;}
# .eval-val{font-family:'Fraunces',serif;font-size:2.1rem;font-weight:700;color:var(--ink);}
# .eval-lbl{font-family:'IBM Plex Mono',monospace;font-size:.67rem;text-transform:uppercase;
#   letter-spacing:.12em;color:var(--muted);margin-top:.2rem;}

# .hist-item{background:var(--card);border:1px solid var(--border);border-radius:8px;
#   padding:.7rem .95rem;margin:.38rem 0;font-size:.83rem;display:flex;align-items:center;gap:.7rem;}
# </style>
# """, unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # PIPELINE
# # ══════════════════════════════════════════════════════════════════════════════
# @st.cache_resource(show_spinner="Loading NLI model & RAG retriever…")
# def load_pipeline(use_judge: bool):
#     from pipeline.verifier_pipeline import VerifierPipeline
#     return VerifierPipeline(use_judge=use_judge)


# def _demo_result(question, response):
#     import hashlib
#     seed = int(hashlib.md5((question + response).encode()).hexdigest(), 16) % 9999
#     random.seed(seed)
#     lbl  = random.choices(["CORRECT","HALLUCINATED","UNCERTAIN"], [.44,.37,.19])[0]
#     conf = random.randint(63,96) if lbl != "UNCERTAIN" else random.randint(38,59)
#     ent  = round(random.uniform(.57,.93) if lbl=="CORRECT"      else random.uniform(.10,.38), 3)
#     con  = round(random.uniform(.57,.90) if lbl=="HALLUCINATED" else random.uniform(.05,.28), 3)
#     exps = {
#         "CORRECT":      "The response is well-supported by retrieved Wikipedia evidence. Key facts align.",
#         "HALLUCINATED": "The response contradicts retrieved evidence. Specific claims could not be verified.",
#         "UNCERTAIN":    "Evidence is ambiguous. Passages neither confirm nor deny the response.",
#     }
#     return {
#         "question":question,"response":response or f"[Auto-answer for: {question}]",
#         "final_label":lbl,"confidence":conf,"explanation":exps[lbl],
#         "nli_label":lbl,"entailment_score":ent,"contradiction_score":con,
#         "retrieved_passages":[
#             f"Historical records show that {question.rstrip('?')} is documented in academic literature.",
#             "Multiple encyclopaedic sources provide detailed information on this subject.",
#             "Peer-reviewed references confirm the established consensus on this topic.",
#         ],
#         "retrieved_sources":["Wikipedia (Demo)","Encyclopedia Britannica (Demo)","Academic DB (Demo)"],
#         "auto_generated":not bool(response.strip()),"demo_mode":True,
#     }


# def run_verify(question, response, auto_gen, use_judge):
#     import config as cfg
#     # if not cfg.ANTHROPIC_API_KEY:
#     #     return _demo_result(question, response)
#     if not cfg.GEMINI_API_KEY:
#       return _demo_result(question, response)
#     try:
#         pipe = load_pipeline(use_judge)
#         r    = pipe.verify(question=question, response=response, generate_answer=auto_gen)
#         return r.to_dict()
#     except Exception as e:
#         st.warning(f"Pipeline error ({e}) — showing demo result.")
#         return _demo_result(question, response)


# # ══════════════════════════════════════════════════════════════════════════════
# # SESSION STATE
# # ══════════════════════════════════════════════════════════════════════════════
# if "history" not in st.session_state:
#     st.session_state["history"] = []


# # ══════════════════════════════════════════════════════════════════════════════
# # SIDEBAR
# # ══════════════════════════════════════════════════════════════════════════════
# with st.sidebar:
#     st.markdown("""
# <div style='font-family:"IBM Plex Mono",monospace;font-size:.63rem;letter-spacing:.18em;
#   text-transform:uppercase;color:#f0a97a;margin-bottom:.3rem;'>Meta-LLM System</div>
# <div style='font-family:"Fraunces",serif;font-size:1.25rem;font-weight:700;
#   line-height:1.2;margin-bottom:1.2rem;'>Hallucination<br>Detector</div>
# """, unsafe_allow_html=True)

#     st.markdown("#### ⚙️ Settings")
#     use_judge = st.toggle("Judge LLM (Gemini)", value=True,
#         help="Disable for NLI-only mode — no API key needed.")

#     st.markdown("---")
#     st.markdown("#### 🧪 TruthfulQA Evaluation")
#     n_eval = st.slider("Samples", 10, 150, 40, step=10)
#     run_eval_btn = st.button("▶ Run Evaluation", use_container_width=True)

#     st.markdown("---")
#     st.markdown("""<div style='font-family:"IBM Plex Mono",monospace;font-size:.69rem;
# line-height:2;color:#c8b8a8;'>
# <b style='color:#f5f2eb;'>Dataset</b><br>
# <a href='https://huggingface.co/datasets/truthful_qa'>TruthfulQA ↗</a><br>
# <b style='color:#f5f2eb;'>NLI Model</b><br>
# <a href='https://huggingface.co/cross-encoder/nli-deberta-v3-base'>DeBERTa-v3-base ↗</a><br>
# <b style='color:#f5f2eb;'>Embeddings</b><br>
# <a href='https://www.sbert.net/'>sentence-transformers ↗</a><br>
# <b style='color:#f5f2eb;'>Vector DB</b><br>
# <a href='https://github.com/facebookresearch/faiss'>FAISS ↗</a><br>
# <b style='color:#f5f2eb;'>LLM API</b><br>
# <a href='https://console.anthropic.com/'>Anthropic Claude ↗</a>
# </div>""", unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # HERO
# # ══════════════════════════════════════════════════════════════════════════════
# st.markdown("""
# <div class="hero">
#   <div class="hero-eyebrow">RAG · NLI · LLM-as-Judge · TruthfulQA</div>
#   <div class="hero-title">Meta-LLM Hallucination<br>Detection System</div>
#   <div class="hero-sub">
#     End-to-end factual verification — retrieves Wikipedia evidence, runs NLI,
#     and delivers a verdict with confidence score and cited facts.
#   </div>
# </div>
# """, unsafe_allow_html=True)


# # ══════════════════════════════════════════════════════════════════════════════
# # TABS
# # ══════════════════════════════════════════════════════════════════════════════
# tab_v, tab_e, tab_c, tab_a = st.tabs([
#     "🔬 Verify Query",
#     "📊 Evaluation Results",
#     "📂 Category Comparison",
#     "🏗 Architecture",
# ])


# # ─────────────────────────────────────────────────────────────────────────────
# #  TAB 1 · VERIFY
# # ─────────────────────────────────────────────────────────────────────────────
# with tab_v:
#     ci, cr = st.columns([1, 1], gap="large")

#     with ci:
#         st.markdown("#### ① Enter Query")
#         question = st.text_area("Question", height=100,
#             value="Who invented the World Wide Web and when?",
#             placeholder="Enter a factual question…")

#         st.markdown("**② Response to verify**")
#         mode = st.radio("src", ["Type manually","Auto-generate with Claude"],
#                         horizontal=True, label_visibility="collapsed")
#         auto_gen = (mode == "Auto-generate with Claude")

#         response_text = ""
#         if not auto_gen:
#             response_text = st.text_area("Response", height=105, label_visibility="collapsed",
#                 value="Tim Berners-Lee invented the World Wide Web in 1989 at CERN in Geneva.",
#                 placeholder="Paste LLM response…")
#         else:
#             st.info("🤖 Claude generates then self-verifies an answer.")

#         go_btn = st.button("🔍 Verify Now", use_container_width=True)

#     with cr:
#         st.markdown("#### Verification Output")

#         if go_btn:
#             if not question.strip():
#                 st.warning("Please enter a question.")
#             else:
#                 with st.spinner("Running: RAG → NLI → Judge…"):
#                     t0  = time.time()
#                     res = run_verify(question, response_text, auto_gen, use_judge)
#                     elapsed = round(time.time()-t0, 2)

#                 res["elapsed"] = elapsed
#                 st.session_state["history"].insert(0, res)

#                 lbl   = res["final_label"]
#                 conf  = res["confidence"]
#                 emoji = {"CORRECT":"✅","HALLUCINATED":"🚨","UNCERTAIN":"❓"}[lbl]
#                 clr   = {"CORRECT":"#1a7a42","HALLUCINATED":"#c8401a","UNCERTAIN":"#b07c10"}[lbl]

#                 # ── Structured result card ──
#                 demo_note = (
#                     "<div style='margin-top:.3rem;font-size:.73rem;color:var(--accent);"
#                     "font-family:IBM Plex Mono,monospace;'>⚠ Demo mode — add ANTHROPIC_API_KEY to .env</div>"
#                     if res.get("demo_mode") else ""
#                 )
#                 auto_note = (
#                     "<div style='margin-top:.25rem;font-size:.73rem;color:var(--muted);"
#                     "font-family:IBM Plex Mono,monospace;'>★ Auto-generated by Claude</div>"
#                     if res.get("auto_generated") else ""
#                 )

#                 st.markdown(f"""
# <div class="result-shell">
#   <div class="result-header">
#     <span style="font-size:1.35rem">{emoji}</span>
#     <span class="verdict-label verdict-{lbl}">{lbl}</span>
#     <span style="flex:1"></span>
#     <span style="font-family:'IBM Plex Mono',monospace;font-size:.76rem;color:var(--muted)">⏱ {elapsed}s</span>
#   </div>

#   <div class="result-section">
#     <div class="section-tag">① Answer</div>
#     <div class="answer-text">{res["response"]}</div>
#     {auto_note}{demo_note}
#   </div>

#   <div class="result-section">
#     <div class="section-tag">② Verdict</div>
#     <div style="display:flex;align-items:center;gap:.6rem;margin-top:.25rem;">
#       <span class="verdict-label verdict-{lbl}" style="font-size:.82rem;padding:.3rem .9rem;">{emoji} {lbl}</span>
#       <span style="font-size:.83rem;color:var(--muted);">{res.get("explanation","")}</span>
#     </div>
#   </div>

#   <div class="result-section">
#     <div class="section-tag">③ Confidence Score</div>
#     <div style="display:flex;align-items:center;gap:.75rem;margin-top:.3rem;">
#       <div style="font-family:'Fraunces',serif;font-size:2rem;font-weight:700;color:{clr};">{conf}%</div>
#       <div style="flex:1;">
#         <div class="conf-bar-track">
#           <div class="conf-bar-fill" style="width:{conf}%;background:{clr};"></div>
#         </div>
#       </div>
#     </div>
#     <div style="display:flex;gap:1.5rem;margin-top:.55rem;font-family:'IBM Plex Mono',monospace;font-size:.78rem;">
#       <div><div style="color:var(--muted);font-size:.62rem;margin-bottom:.08rem;">ENTAILMENT</div>
#            <div style="color:var(--correct);font-weight:600;">{res.get("entailment_score",0):.3f}</div></div>
#       <div><div style="color:var(--muted);font-size:.62rem;margin-bottom:.08rem;">CONTRADICTION</div>
#            <div style="color:var(--halluc);font-weight:600;">{res.get("contradiction_score",0):.3f}</div></div>
#       <div><div style="color:var(--muted);font-size:.62rem;margin-bottom:.08rem;">NLI LABEL</div>
#            <div style="font-weight:600;">{res.get("nli_label","—")}</div></div>
#     </div>
#   </div>
# </div>
# """, unsafe_allow_html=True)

#                 # ── Evidence (④) ──
#                 st.markdown("##### ④ Evidence (Retrieved Facts)")
#                 passages = res.get("retrieved_passages", [])
#                 sources  = res.get("retrieved_sources",  [])
#                 if passages:
#                     for p, src in zip(passages[:4], sources[:4]):
#                         snippet = p[:340] + ("…" if len(p)>340 else "")
#                         st.markdown(f"""
# <div class="evidence-card">
#   <div class="evidence-source">📖 {src}</div>
#   {snippet}
# </div>""", unsafe_allow_html=True)
#                 else:
#                     st.info("No Wikipedia passages retrieved.")

#         else:
#             st.markdown("""
# <div style="background:var(--card);border:1.5px dashed var(--border);border-radius:12px;
#   padding:2.8rem 2rem;text-align:center;color:var(--muted);">
#   <div style="font-size:2.3rem;margin-bottom:.6rem;">🧠</div>
#   <div style="font-family:'Fraunces',serif;font-size:1.05rem;color:var(--ink);">
#     Enter a question &amp; click <b>Verify Now</b>
#   </div>
#   <div style="font-size:.83rem;margin-top:.35rem;">
#     Returns: Answer · Verdict · Confidence Score · Evidence
#   </div>
# </div>""", unsafe_allow_html=True)

#     # ── History ──
#     if st.session_state["history"]:
#         st.markdown("---")
#         st.markdown("#### 🕓 Session History")
#         for h in st.session_state["history"][:8]:
#             lbl   = h["final_label"]
#             emoji = {"CORRECT":"✅","HALLUCINATED":"🚨","UNCERTAIN":"❓"}.get(lbl,"❓")
#             clr   = {"CORRECT":"var(--correct)","HALLUCINATED":"var(--halluc)","UNCERTAIN":"var(--uncert)"}.get(lbl,"")
#             st.markdown(f"""
# <div class="hist-item">
#   <span style="font-size:1.05rem">{emoji}</span>
#   <div style="flex:1;overflow:hidden;">
#     <div style="font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{h["question"][:80]}</div>
#     <div style="font-size:.74rem;color:var(--muted);font-family:'IBM Plex Mono',monospace;">{h["response"][:60]}…</div>
#   </div>
#   <span style="font-family:'IBM Plex Mono',monospace;font-size:.71rem;font-weight:700;color:{clr};">
#     {lbl} · {h["confidence"]}%
#   </span>
# </div>""", unsafe_allow_html=True)


# # ─────────────────────────────────────────────────────────────────────────────
# #  TAB 2 · EVALUATION RESULTS
# # ─────────────────────────────────────────────────────────────────────────────
# with tab_e:
#     if run_eval_btn:
#         st.session_state["trigger_eval"] = n_eval

#     if st.session_state.get("trigger_eval"):
#         n = st.session_state.pop("trigger_eval")
#         st.markdown(f"### Running TruthfulQA Evaluation — {n} samples")
#         try:
#             from evaluation.truthfulqa_loader import load_truthfulqa
#             from evaluation.evaluator import Evaluator
#             samples = load_truthfulqa(n_samples=n)
#             pipe    = load_pipeline(use_judge)
#             ev      = Evaluator(pipeline=pipe, output_dir="results")
#             prog    = st.progress(0, text="Starting…")
#             rows_l, y_true_l, y_pred_l = [], [], []
#             for i, s in enumerate(samples):
#                 prog.progress((i+1)/len(samples), text=f"Sample {i+1}/{len(samples)}: {s['question'][:55]}…")
#                 r = pipe.verify(question=s["question"], response=s["response"])
#                 rows_l.append({"Question":s["question"][:65],"Response":s["response"][:60],
#                                 "True":s["true_label"],"Predicted":r.final_label,
#                                 "Conf%":r.confidence,"Match":"✅" if s["true_label"]==r.final_label else "❌"})
#                 y_true_l.append(s["true_label"]); y_pred_l.append(r.final_label)
#             prog.empty()
#             report = ev._compute_metrics(y_true_l, y_pred_l, rows_l)
#             st.session_state["eval_report"] = report
#             st.session_state["eval_rows"]   = rows_l
#         except Exception as e:
#             st.error(f"Evaluation failed: {e}"); st.exception(e)

#     # Load report
#     if "eval_report" in st.session_state:
#         report = st.session_state["eval_report"]
#         rows   = st.session_state.get("eval_rows", [])
#     elif Path("results/metrics_report.json").exists():
#         report = json.loads(Path("results/metrics_report.json").read_text())
#         rows   = pd.read_csv("results/predictions.csv").to_dict("records") if Path("results/predictions.csv").exists() else []
#     else:
#         report = {"num_samples":100,"accuracy":.743,
#                   "per_class":{
#                       "CORRECT":     {"precision":.812,"recall":.791,"f1":.801,"support":52},
#                       "HALLUCINATED":{"precision":.718,"recall":.745,"f1":.731,"support":38},
#                       "UNCERTAIN":   {"precision":.624,"recall":.590,"f1":.607,"support":10},
#                   },
#                   "macro_avg":{"precision":.718,"recall":.709,"f1-score":.713},
#                   "weighted_avg":{"precision":.758,"recall":.743,"f1-score":.750},
#                   "confusion_matrix":[[41,8,3],[7,28,3],[2,2,6]],
#                   "labels":["CORRECT","HALLUCINATED","UNCERTAIN"],"demo":True}
#         rows = []
#         st.info("📊 Demo results shown. Click **▶ Run Evaluation** in the sidebar for live TruthfulQA metrics.")

#     # ── Top metrics ──
#     st.markdown("### 📊 TruthfulQA Evaluation Metrics")
#     if report.get("demo"): st.caption("⚠ Demo metrics")

#     ma = report.get("macro_avg", {}); wa = report.get("weighted_avg",{})
#     def pct(v): return f"{float(v)*100:.1f}%"

#     m1,m2,m3,m4,m5 = st.columns(5)
#     for col, val, lbl in [
#         (m1, pct(report.get("accuracy",0)), "Accuracy"),
#         (m2, f"{float(ma.get('precision',0)):.3f}", "Macro Precision"),
#         (m3, f"{float(ma.get('recall',0)):.3f}",    "Macro Recall"),
#         (m4, f"{float(ma.get('f1-score',0)):.3f}",  "Macro F1-Score"),
#         (m5, str(report.get("num_samples",0)),       "Total Samples"),
#     ]:
#         col.markdown(f'<div class="eval-card"><div class="eval-val">{val}</div>'
#                      f'<div class="eval-lbl">{lbl}</div></div>', unsafe_allow_html=True)

#     st.markdown("---")
#     col_l, col_r = st.columns([1,1.2], gap="large")
#     LABELS_E = ["CORRECT","HALLUCINATED","UNCERTAIN"]
#     COLOURS_E = {"CORRECT":"#1a7a42","HALLUCINATED":"#c8401a","UNCERTAIN":"#b07c10"}

#     with col_l:
#         st.markdown("#### Per-Class Precision / Recall / F1")
#         tbl = [{"Category":l,
#                 "Precision": f"{report['per_class'].get(l,{}).get('precision',0):.4f}",
#                 "Recall":    f"{report['per_class'].get(l,{}).get('recall',0):.4f}",
#                 "F1-Score":  f"{report['per_class'].get(l,{}).get('f1',0):.4f}",
#                 "Support":    report['per_class'].get(l,{}).get('support','—')}
#                for l in LABELS_E]
#         st.dataframe(pd.DataFrame(tbl).set_index("Category"), use_container_width=True)

#         st.markdown("#### Weighted Averages")
#         st.dataframe(pd.DataFrame([
#             {"Metric":"Precision","Score":f"{float(wa.get('precision',0)):.4f}"},
#             {"Metric":"Recall",   "Score":f"{float(wa.get('recall',0)):.4f}"},
#             {"Metric":"F1-Score", "Score":f"{float(wa.get('f1-score',0)):.4f}"},
#         ]).set_index("Metric"), use_container_width=True)

#     with col_r:
#         st.markdown("#### Grouped Bar — Precision / Recall / F1")
#         fig_bar = go.Figure()
#         for lbl in LABELS_E:
#             m = report["per_class"].get(lbl,{})
#             vals = [m.get("precision",0), m.get("recall",0), m.get("f1",0)]
#             fig_bar.add_trace(go.Bar(
#                 name=lbl, x=["Precision","Recall","F1-Score"], y=vals,
#                 marker_color=COLOURS_E[lbl],
#                 text=[f"{v:.3f}" for v in vals], textposition="outside",
#                 textfont=dict(size=9,family="IBM Plex Mono"),
#             ))
#         fig_bar.update_layout(
#             barmode="group",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
#             font=dict(family="DM Sans",color="#1a1410"),
#             legend=dict(orientation="h",y=-0.18,x=0),
#             yaxis=dict(range=[0,1.18],gridcolor="#d9d2c5",tickformat=".2f"),
#             xaxis=dict(gridcolor="#d9d2c5"),
#             height=285,margin=dict(t=15,b=65,l=10,r=10),
#         )
#         st.plotly_chart(fig_bar, use_container_width=True)

#     # ── Confusion matrix ──
#     st.markdown("#### Confusion Matrix")
#     import numpy as np
#     cm_arr = np.array(report.get("confusion_matrix",[[0,0,0],[0,0,0],[0,0,0]]))
#     lbls_cm = report.get("labels", LABELS_E)
#     fig_cm = go.Figure(go.Heatmap(
#         z=cm_arr, x=lbls_cm, y=lbls_cm,
#         colorscale=[[0,"#f5f2eb"],[.5,"#f0a97a"],[1,"#c8401a"]],
#         text=cm_arr, texttemplate="<b>%{text}</b>",
#         textfont=dict(size=14,family="IBM Plex Mono"), showscale=True,
#     ))
#     fig_cm.update_layout(
#         xaxis_title="Predicted", yaxis_title="True",
#         paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
#         font=dict(family="DM Sans",color="#1a1410"),
#         height=310,margin=dict(t=15,b=50,l=80,r=20),
#     )
#     st.plotly_chart(fig_cm, use_container_width=True)

#     if rows:
#         st.markdown("#### Prediction Details")
#         st.dataframe(pd.DataFrame(rows), use_container_width=True, height=270)


# # ─────────────────────────────────────────────────────────────────────────────
# #  TAB 3 · CATEGORY COMPARISON
# # ─────────────────────────────────────────────────────────────────────────────
# with tab_c:
#     st.markdown("### 📂 Hallucination Category Comparison")

#     hist = st.session_state.get("history", [])
#     CMAP = {"CORRECT":"#1a7a42","HALLUCINATED":"#c8401a","UNCERTAIN":"#b07c10"}

#     if len(hist) >= 3:
#         df_h   = pd.DataFrame(hist)
#         counts = df_h["final_label"].value_counts().reset_index(); counts.columns=["Category","Count"]
#         avgc   = df_h.groupby("final_label")["confidence"].mean().reset_index(); avgc.columns=["Category","Avg Conf"]
#     else:
#         counts = pd.DataFrame({"Category":["CORRECT","HALLUCINATED","UNCERTAIN"],"Count":[44,37,19]})
#         avgc   = pd.DataFrame({"Category":["CORRECT","HALLUCINATED","UNCERTAIN"],"Avg Conf":[83.4,78.1,51.6]})
#         st.info("Showing demo data. Verify more queries to see live category stats.")

#     c_left, c_right = st.columns(2, gap="large")

#     with c_left:
#         st.markdown("#### Verdict Distribution")
#         colours = [CMAP.get(c,"#888") for c in counts["Category"]]
#         fig_pie = go.Figure(go.Pie(
#             labels=counts["Category"], values=counts["Count"], hole=.54,
#             marker_colors=colours,
#             textinfo="label+percent", textfont=dict(family="IBM Plex Mono",size=11),
#             pull=[.04,.04,.04],
#         ))
#         fig_pie.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
#             font=dict(family="DM Sans",color="#1a1410"),
#             legend=dict(orientation="h",y=-0.1),
#             height=295,margin=dict(t=15,b=40,l=15,r=15),
#             annotations=[dict(text=f"<b>{counts['Count'].sum()}</b><br>Queries",
#                               font=dict(size=13,family="Fraunces",color="#1a1410"),showarrow=False)],
#         )
#         st.plotly_chart(fig_pie, use_container_width=True)

#     with c_right:
#         st.markdown("#### Avg Confidence by Category")
#         fig_conf = go.Figure(go.Bar(
#             x=avgc["Category"], y=avgc["Avg Conf"],
#             marker_color=[CMAP.get(c,"#888") for c in avgc["Category"]],
#             text=[f"{v:.1f}%" for v in avgc["Avg Conf"]], textposition="outside",
#             textfont=dict(size=11,family="IBM Plex Mono"),
#         ))
#         fig_conf.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
#             font=dict(family="DM Sans",color="#1a1410"),
#             yaxis=dict(range=[0,110],gridcolor="#d9d2c5",title="Avg Confidence %"),
#             xaxis=dict(gridcolor="#d9d2c5"),
#             height=295,margin=dict(t=15,b=15,l=25,r=15),
#         )
#         st.plotly_chart(fig_conf, use_container_width=True)

#     # ── Category definition cards ──
#     st.markdown("#### Category Definitions")
#     cc1,cc2,cc3 = st.columns(3)
#     for col,(lbl,em,fg,bg,bd,desc) in zip([cc1,cc2,cc3],[
#         ("CORRECT",     "✅","#1a7a42","#eef7f2","#a3d4b7",
#          "NLI entailment ≥ 0.6 for at least one passage. The Judge LLM confirms the response aligns with retrieved Wikipedia facts."),
#         ("HALLUCINATED","🚨","#c8401a","#fdf0ec","#f0b8a8",
#          "NLI contradiction ≥ 0.6 and exceeds entailment. The model invented or distorted facts not supported by evidence."),
#         ("UNCERTAIN",   "❓","#b07c10","#fdf6e8","#e8d09a",
#          "Neither entailment nor contradiction threshold met. Evidence is insufficient to confirm or deny the claim."),
#     ]):
#         with col:
#             st.markdown(f"""
# <div style="background:{bg};border:1.5px solid {bd};border-radius:10px;padding:1.1rem;">
#   <div style="font-size:1.45rem;margin-bottom:.35rem;">{em}</div>
#   <div style="font-family:'IBM Plex Mono',monospace;font-size:.69rem;font-weight:700;
#     text-transform:uppercase;letter-spacing:.12em;color:{fg};margin-bottom:.45rem;">{lbl}</div>
#   <div style="font-size:.83rem;line-height:1.58;color:#1a1410;">{desc}</div>
# </div>""", unsafe_allow_html=True)

#     # ── NLI scatter ──
#     st.markdown("---")
#     st.markdown("#### NLI Score Space — Entailment vs Contradiction")

#     if len(hist) >= 3:
#         df_sc = pd.DataFrame([{
#             "Category":res["final_label"],
#             "Entailment":res.get("entailment_score",0),
#             "Contradiction":res.get("contradiction_score",0),
#         } for res in hist])
#     else:
#         random.seed(42)
#         sc_rows = []
#         for lbl, er, cr_ in [("CORRECT",(.55,.95),(.02,.28)),
#                               ("HALLUCINATED",(.05,.35),(.55,.92)),
#                               ("UNCERTAIN",(.20,.54),(.20,.50))]:
#             for _ in range(22):
#                 sc_rows.append({"Category":lbl,
#                                  "Entailment":random.uniform(*er),
#                                  "Contradiction":random.uniform(*cr_)})
#         df_sc = pd.DataFrame(sc_rows)

#     fig_sc = px.scatter(df_sc, x="Entailment", y="Contradiction", color="Category",
#                         color_discrete_map=CMAP, opacity=.78, size_max=10)
#     fig_sc.add_hline(y=.6, line_dash="dash", line_color="#7a6f62",
#                      annotation_text="Contradiction threshold", annotation_position="right")
#     fig_sc.add_vline(x=.6, line_dash="dash", line_color="#7a6f62",
#                      annotation_text="Entailment threshold", annotation_position="top left")
#     fig_sc.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
#         font=dict(family="DM Sans", color="#1a1410"),
#         legend=dict(orientation="h",y=-0.2),
#         xaxis=dict(range=[0,1],gridcolor="#d9d2c5",title="Entailment Score"),
#         yaxis=dict(range=[0,1],gridcolor="#d9d2c5",title="Contradiction Score"),
#         height=360, margin=dict(t=20,b=80,l=20,r=20),
#     )
#     st.plotly_chart(fig_sc, use_container_width=True)


# # ─────────────────────────────────────────────────────────────────────────────
# #  TAB 4 · ARCHITECTURE
# # ─────────────────────────────────────────────────────────────────────────────
# with tab_a:
#     st.markdown("### 🏗 End-to-End Pipeline Architecture")
#     st.markdown("""
# ```
# INPUT: Question (+ optional response)
#          │
#          ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  STEP 0 · STUDENT LLM  (Claude Sonnet 4)          [optional]           │
# │  Generates an initial answer → OUTPUT ①  Answer                        │
# └────────────────────────┬────────────────────────────────────────────────┘
#                          │ response to verify
#                          ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  STEP 1 · RAG RETRIEVAL                                                 │
# │  Query → Wikipedia API → sentence-transformers (all-MiniLM-L6-v2)      │
# │  → FAISS IndexFlatIP cosine search → Top-K passages                    │
# │                                   → OUTPUT ④  Evidence                 │
# └────────────────────────┬────────────────────────────────────────────────┘
#                          │ evidence passages
#                          ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  STEP 2 · NLI CLASSIFICATION  (cross-encoder/nli-deberta-v3-base)      │
# │  (response, passage) → entailment / contradiction / neutral scores      │
# │  Aggregation:                                                           │
# │    contradiction ≥ 0.6 AND > entailment  →  HALLUCINATED               │
# │    entailment    ≥ 0.6                   →  CORRECT                    │
# │    otherwise                             →  UNCERTAIN                  │
# └────────────────────────┬────────────────────────────────────────────────┘
#                          │ NLI signal + best evidence
#                          ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  STEP 3 · JUDGE LLM  (Claude Sonnet 4)                                 │
# │  Synthesises NLI scores + evidence + response                           │
# │  → OUTPUT ②  Verdict  (CORRECT | HALLUCINATED | UNCERTAIN)            │
# │  → OUTPUT ③  Confidence %  + natural language explanation              │
# └─────────────────────────────────────────────────────────────────────────┘
# ```
# """)

#     st.markdown("---")
#     la, lb = st.columns(2)
#     with la:
#         st.markdown("#### Tech Stack")
#         st.dataframe(pd.DataFrame([
#             ("Student / Judge LLM","anthropic · claude-sonnet-4","Meta-LLM roles"),
#             ("Embeddings","sentence-transformers · MiniLM","384-dim dense vectors"),
#             ("Vector search","faiss-cpu · IndexFlatIP","Cosine similarity"),
#             ("Knowledge source","Wikipedia API","Free, no key required"),
#             ("NLI model","DeBERTa-v3-base (cross-encoder)","State-of-art NLI"),
#             ("Benchmark","TruthfulQA (HuggingFace)","817 adversarial QA pairs"),
#             ("Metrics","scikit-learn","Precision / Recall / F1 / CM"),
#             ("Dashboard","Streamlit + Plotly","This app"),
#         ], columns=["Component","Library","Notes"]), use_container_width=True, hide_index=True)
#     with lb:
#         st.markdown("#### Output Schema")
#         st.markdown("""
# | # | Field | Description |
# |---|---|---|
# | ① | **Answer** | The LLM response being verified |
# | ② | **Verdict** | `CORRECT` · `HALLUCINATED` · `UNCERTAIN` |
# | ③ | **Confidence** | 0–100 % from Judge LLM |
# | ④ | **Evidence** | Top-K Wikipedia passages + sources |
# | — | NLI scores | Entailment · Contradiction · Neutral |
# | — | Explanation | Natural language reasoning |
# """)
#         st.markdown("#### Threshold Decisions")
#         st.markdown("""
# | Condition | Label |
# |---|---|
# | contradiction ≥ 0.6 AND > entailment | 🚨 HALLUCINATED |
# | entailment ≥ 0.6 | ✅ CORRECT |
# | neither threshold met | ❓ UNCERTAIN |
# """)

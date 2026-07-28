"""
app.py — "VERIDEX / The Archive": a nocturnal, evidence-grounded RAG console.

Phase 10 UI, consuming the clean framework-agnostic ``src/`` package only:
    DataLoader -> TextProcessor -> Chunker -> EmbeddingGenerator
             -> VectorStoreManager -> Retriever -> RAGPipeline -> EvaluationEngine
"""

from __future__ import annotations

import html as _html
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime

import streamlit as st
from src.chunker import Chunker
from src.config import LLMConfig, get_settings
from src.data_loader import DataLoader, DatasetLoadingError
from src.embeddings import EmbeddingGenerator
from src.evaluation import EvaluationEngine
from src.rag_pipeline import (
    GroqLLMClient,
    PipelineConnectionError,
    RAGPipeline,
    estimate_retrieval_confidence,
)
from src.retriever import Retriever
from src.text_processor import TextProcessor
from src.vector_store import VectorStoreManager

# ============================================================
# Conversation memory (UI-layer only; src/ is untouched)
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def _augment_question(question: str, history: list) -> str:
    """Carry the last exchange so follow-ups resolve pronouns ('he', 'it')."""
    if not history:
        return question
    last = history[-1]
    return (
        f"Previous question: {last['q']}\n"
        f"Previous answer: {last['a']}\n\n"
        f"Follow-up question: {question}"
    )


def _citation_edges(snippet: str):
    """Decide where to show an ellipsis so a mid-paragraph chunk reads as a
    deliberate excerpt, not a broken sentence."""
    s = snippet.strip()
    lead = "" if (s and s[0].isupper()) else "…"
    trail = "" if (s and s[-1] in ".!?") else "…"
    return lead, trail


def _build_citation(question, result, pages, metas, sims) -> str:
    """Build a Markdown citation the user can copy (shown verbatim in st.code)."""
    lines = [
        "# Answer",
        result.get("answer", ""),
        "",
        f"**Question:** {question}",
        "",
        "## Sources (retrieved evidence)",
    ]
    for i, (p, m, s) in enumerate(zip(pages, metas, sims, strict=False), 1):
        title = str(m.get("title") or "Untitled")
        text = p.replace("\n", " ").strip()
        lines += [f"{i}. **{title}** — similarity {s:.3f}", f"   > {text}", ""]
    return "\n".join(lines)


def _log_feedback(question: str, answer: str, vote: str) -> None:
    """Append a thumbs vote to logs/feedback.jsonl (best-effort, never breaks UI)."""
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/feedback.jsonl", "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "vote": vote,
                        "question": question,
                        "answer": answer,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass


# ============================================================
# Tiny offline fallback corpus (used only if HuggingFace is unreachable)
# ============================================================
_FALLBACK_RECORDS: dict = {
    "train": [
        {
            "id": "f1",
            "title": "Beyoncé",
            "context": (
                "Beyoncé Giselle Knowles-Carter is an American singer, "
                "songwriter and actress. Born and raised in Houston, Texas, she "
                "rose to fame in the late 1990s as lead singer of the R&B "
                "girl-group Destiny's Child. Managed by her father, Mathew "
                "Knowles, the group became one of the best-selling girl groups "
                "of all time."
            ),
            "question": "Who is Beyoncé?",
            "answers": {"text": ["American singer"], "answer_start": [27]},
        },
        {
            "id": "f2",
            "title": "iPod",
            "context": (
                "The iPod is a line of portable media players and multi-purpose "
                "pocket computers designed and marketed by Apple. The first line "
                "was released in 2001."
            ),
            "question": "What is the iPod?",
            "answers": {"text": ["portable media players"], "answer_start": [18]},
        },
        {
            "id": "f3",
            "title": "Solar energy",
            "context": (
                "Solar energy is radiant light and heat from the Sun that is "
                "harnessed using a range of technologies such as solar power to "
                "generate electricity and solar thermal energy for heating."
            ),
            "question": "What is solar energy?",
            "answers": {
                "text": ["radiant light and heat from the Sun"],
                "answer_start": [17],
            },
        },
    ]
}


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="VERIDEX · Evidence-Grounded Q&A",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Design system — "nocturnal archive"
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Hanken+Grotesk:wght@400;500;600&family=Newsreader:ital,opsz@1,6..72&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{
  --ink:#0c0f14; --ink-2:#11151c; --ink-3:#161b24; --line:#262d3a;
  --paper:#f3ecdc; --paper-2:#e9dfc8; --paper-ink:#171a20; --paper-soft:#5b5748;
  --text:#ece6d8; --muted:#8b93a1;
  --brass:#d8b15a; --brass-deep:#b78a2e;
  --oxblood:#9c3b34; --sage:#5f8a5a; --amber:#c98a2c;
}
html,body,[class*="css"]{font-family:'Hanken Grotesk',sans-serif;}
.stApp{background:var(--ink);}
.main .block-container{position:relative;z-index:1;padding-top:1.4rem;max-width:1180px;}
.ambient{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden;background:radial-gradient(120% 80% at 50% -10%, #161c26 0%, var(--ink) 60%);}
.ambient .g{position:absolute;border-radius:50%;filter:blur(90px);opacity:.42;mix-blend-mode:screen;}
.ambient .g1{width:46vw;height:46vw;left:-8vw;top:-12vw;background:radial-gradient(circle,#7a5418 0%,transparent 70%);animation:drift 26s ease-in-out infinite alternate;}
.ambient .g2{width:40vw;height:40vw;right:-10vw;top:18vh;background:radial-gradient(circle,#5a1f1c 0%,transparent 70%);animation:drift 32s ease-in-out infinite alternate-reverse;}
.ambient .g3{width:34vw;height:34vw;left:18vw;bottom:-14vw;background:radial-gradient(circle,#163a39 0%,transparent 70%);animation:drift 38s ease-in-out infinite alternate;}
@keyframes drift{from{transform:translate3d(0,0,0) scale(1);}to{transform:translate3d(4%,3%,0) scale(1.12);}}
.topbar{position:sticky;top:0;z-index:40;display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.7rem .2rem 1rem;backdrop-filter:blur(8px);background:linear-gradient(var(--ink) 60%,transparent);}
.brand{display:flex;align-items:center;gap:.7rem;}
.brand .mark{width:32px;height:32px;border:1.5px solid var(--brass);border-radius:4px;display:grid;place-items:center;color:var(--brass);box-shadow:0 0 18px -6px var(--brass);}
.brand .mark svg{display:block;}
.brand .name{font-family:'IBM Plex Mono',monospace;letter-spacing:.34em;text-transform:uppercase;font-size:12px;color:var(--text);}
.brand-tag{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:.2em;text-transform:uppercase;color:var(--muted);border:1px solid var(--line);padding:3px 9px;border-radius:3px;}
.eyebrow{font-family:'IBM Plex Mono',monospace;letter-spacing:.3em;text-transform:uppercase;font-size:11.5px;color:var(--brass);display:inline-flex;align-items:center;gap:.6rem;margin-bottom:1.1rem;}
.eyebrow::before{content:"";width:26px;height:1px;background:var(--brass);}
.hero-title{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:clamp(3rem,8.5vw,6.4rem);line-height:.92;letter-spacing:-.02em;color:var(--text);margin:0;}
.hero-title em{font-style:normal;color:var(--brass);}
.uline{display:block;width:min(420px,70%);height:14px;margin-top:.2rem;}
.uline path{stroke:var(--brass);stroke-width:4;fill:none;stroke-linecap:round;stroke-dasharray:600;stroke-dashoffset:600;animation:draw 1.6s .3s ease forwards;}
@keyframes draw{to{stroke-dashoffset:0;}}
.hero-sub{font-size:clamp(1rem,1.5vw,1.18rem);color:var(--muted);max-width:720px;line-height:1.65;margin:1.3rem 0 0;}
.hero-sub code{font-family:'IBM Plex Mono',monospace;color:var(--brass);background:rgba(216,177,90,.1);padding:.05em .4em;border-radius:3px;font-size:.92em;}
.prov{display:flex;align-items:stretch;gap:0;margin:2.4rem 0 .4rem;overflow-x:auto;padding-bottom:.4rem;}
.node{flex:1 1 0;min-width:118px;position:relative;padding:.9rem 1rem .9rem 1.2rem;border-left:1px solid var(--line);}
.node:first-child{border-left:none;}
.node::before{content:"";position:absolute;left:-4px;top:50%;transform:translateY(-50%);width:7px;height:7px;border-radius:50%;background:var(--brass);box-shadow:0 0 12px -2px var(--brass);}
.node:first-child::before{display:none;}
.node .k{font-family:'IBM Plex Mono',monospace;font-size:9.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);}
.node .v{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.35rem;color:var(--text);margin-top:.2rem;line-height:1;}
.node .v small{font-size:.62rem;color:var(--brass);font-weight:500;letter-spacing:.04em;}
hr.rule{border:none;border-top:1px dashed var(--line);margin:2rem 0;}
div[data-testid="stTextInput"] > div,
div[data-testid="stTextInput"] div[data-baseweb="input"],
div[data-testid="stTextInput"] div[data-baseweb="base-input"]{background:var(--ink-3)!important;border:1.5px solid var(--line)!important;border-radius:4px!important;box-shadow:none!important;transition:border-color .2s,box-shadow .2s;}
div[data-testid="stTextInput"] input{background:transparent!important;-webkit-text-fill-color:var(--text)!important;caret-color:var(--brass)!important;border:none!important;color:var(--text)!important;font-family:'Hanken Grotesk',sans-serif!important;font-size:1.08rem!important;padding:.95rem 1.1rem!important;}
div[data-testid="stTextInput"] input::placeholder{color:var(--muted)!important;opacity:.8;-webkit-text-fill-color:var(--muted)!important;}
div[data-testid="stTextInput"] input:focus{box-shadow:0 0 0 3px rgba(216,177,90,.12)!important;}
div[data-testid="stTextInput"]:focus-within > div,
div[data-testid="stTextInput"]:focus-within div[data-baseweb="input"]{border-color:var(--brass)!important;}
div.stButton > button, div.stFormSubmitButton > button{font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:.14em;font-size:12px;font-weight:600;background:linear-gradient(180deg,var(--brass),var(--brass-deep));color:#1a1407;border:none;border-radius:4px;padding:.7rem 1.1rem;transition:transform .14s ease,box-shadow .14s ease,filter .14s;}
div.stButton > button:hover, div.stFormSubmitButton > button:hover{transform:translateY(-2px);box-shadow:0 10px 26px -10px var(--brass);filter:brightness(1.06);color:#1a1407;}
.chip-btn button{background:transparent!important;color:var(--text)!important;border:1px solid var(--line)!important;text-transform:none!important;letter-spacing:0!important;font-family:'Hanken Grotesk',sans-serif!important;font-weight:500!important;font-size:12.5px!important;padding:.5rem .85rem!important;border-radius:999px!important;transition:all .18s!important;}
.chip-btn button:hover{border-color:var(--brass)!important;color:var(--brass)!important;box-shadow:none!important;transform:translateX(2px)!important;}
@keyframes rise{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:translateY(0);}}
.answer-card{position:relative;background:linear-gradient(180deg,var(--paper),var(--paper-2));border-radius:6px;padding:1.9rem 2rem 1.7rem;animation:rise .5s ease;box-shadow:0 30px 70px -30px rgba(0,0,0,.8),0 0 0 1px rgba(216,177,90,.25);}
.answer-card::after{content:"";position:absolute;inset:0;border-radius:6px;pointer-events:none;background:linear-gradient(120deg,transparent 30%,rgba(255,255,255,.5) 50%,transparent 70%);opacity:0;animation:sheen 1.1s .25s ease;}
@keyframes sheen{0%{opacity:0;transform:translateX(-30%);}40%{opacity:.5;}100%{opacity:0;transform:translateX(30%);}}
.answer-card .catalog-no{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--paper-soft);margin-bottom:.7rem;}
.answer-text{font-family:'Newsreader',serif;font-style:italic;font-weight:400;font-size:clamp(1.4rem,3vw,2.05rem);line-height:1.34;color:var(--paper-ink);}
.stamp{display:inline-block;font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:.12em;font-weight:600;font-size:11.5px;padding:6px 13px;border:2px solid currentColor;border-radius:4px;transform:rotate(-4deg);margin-top:1.2rem;animation:wob .6s .3s ease;}
@keyframes wob{0%{transform:rotate(-4deg) scale(.6);opacity:0;}60%{transform:rotate(-4deg) scale(1.12);}100%{transform:rotate(-4deg) scale(1);opacity:1;}}
.stamp-high{color:var(--sage);}.stamp-medium{color:var(--amber);}.stamp-low{color:var(--oxblood);}
.timing-row{margin-top:1.1rem;display:flex;gap:.5rem;flex-wrap:wrap;}
.timing-chip{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:var(--paper-soft);border:1px solid rgba(0,0,0,.14);border-radius:999px;padding:.25rem .7rem;background:rgba(0,0,0,.04);}
.evidence-title{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.5rem;margin:2.6rem 0 .2rem;color:var(--text);letter-spacing:-.01em;}
.evidence-sub{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);margin-bottom:1.1rem;letter-spacing:.06em;}
.index-card{background:linear-gradient(180deg,var(--paper),var(--paper-2));border-radius:5px;padding:1rem 1.15rem;margin-bottom:.7rem;animation:rise .45s ease;box-shadow:0 16px 40px -24px rgba(0,0,0,.85);transition:transform .18s,box-shadow .18s;border-left:3px solid var(--brass-deep);}
.index-card:hover{transform:translateY(-3px);box-shadow:0 22px 50px -22px rgba(0,0,0,.9);}
.index-card .row-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.5rem;gap:1rem;}
.index-card .rank{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:11.5px;color:var(--brass-deep);letter-spacing:.04em;}
.index-card .sim-score{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--paper-soft);white-space:nowrap;}
.sim-track{height:5px;background:rgba(0,0,0,.12);border-radius:3px;overflow:hidden;margin:.2rem 0 .7rem;}
.sim-fill{height:100%;background:linear-gradient(90deg,var(--brass-deep),var(--brass));width:0;animation:grow .9s .15s ease forwards;}
@keyframes grow{from{width:0;}}
.index-card .snippet{font-size:13.5px;line-height:1.62;color:var(--paper-ink);white-space:normal;}
.no-evidence{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted);border:1px dashed var(--line);border-radius:5px;padding:1rem 1.1rem;}
.action-row{margin-top:.8rem;}
.action-row div.stButton > button{padding:.45rem .8rem;font-size:10.5px;letter-spacing:.08em;}
.eval-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;}
@media(max-width:760px){.eval-grid{grid-template-columns:repeat(2,1fr);}}
.eval-tile{background:var(--ink-2);border:1px solid var(--line);border-radius:6px;padding:1rem 1.1rem;position:relative;overflow:hidden;transition:border-color .2s,transform .2s;}
.eval-tile:hover{border-color:var(--brass);transform:translateY(-2px);}
.eval-tile::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--brass);}
.eval-tile .k{font-family:'IBM Plex Mono',monospace;font-size:9.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);}
.eval-tile .v{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:2rem;color:var(--text);line-height:1;margin-top:.4rem;}
.error-card{background:rgba(156,59,52,.12);border:1px solid rgba(156,59,52,.5);border-radius:6px;padding:1.1rem 1.3rem;color:#e7b3ae;font-size:14px;line-height:1.55;}
.load-wrap{padding:3rem 0;text-align:center;}
.load-title{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:1.4rem;color:var(--text);}
.load-sub{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);margin-top:.6rem;letter-spacing:.05em;}
.load-bar{width:min(420px,80%);height:3px;margin:1.4rem auto 0;background:var(--line);border-radius:3px;overflow:hidden;}
.load-bar i{display:block;height:100%;width:40%;background:linear-gradient(90deg,transparent,var(--brass),transparent);animation:slide 1.3s ease-in-out infinite;}
@keyframes slide{0%{transform:translateX(-120%);}100%{transform:translateX(320%);}}
section[data-testid="stSidebar"]{background:var(--ink-2);border-right:1px solid var(--line);}
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label{color:var(--text)!important;}
.side-brand{display:flex;align-items:center;gap:.6rem;font-family:'IBM Plex Mono',monospace;letter-spacing:.22em;text-transform:uppercase;font-size:13px;color:var(--text);font-weight:600;margin:.2rem 0 1.1rem;}
.side-brand svg{color:var(--brass);display:block;}
.sidebar-label{font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:.14em;font-size:10px;color:var(--brass);margin:1.3rem 0 .45rem;}
.stat-line{display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted);padding:.3rem 0;border-bottom:1px dotted var(--line);}
.stat-line b{color:var(--text);font-weight:600;}
div[data-testid="stSlider"]{font-family:'IBM Plex Mono',monospace;}
.thread{margin:0 0 1.2rem 0;}
.thread-q{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--brass);margin:.5rem 0 .15rem;letter-spacing:.03em;}
.thread-a{font-size:13.5px;color:var(--muted);line-height:1.5;padding-left:.7rem;border-left:2px solid var(--line);}
.foot{margin-top:1rem;text-align:center;padding:1.4rem 0 .5rem;}
.foot-mark{display:inline-flex;align-items:center;gap:.5rem;font-family:'IBM Plex Mono',monospace;letter-spacing:.3em;text-transform:uppercase;font-size:12px;color:var(--brass);}
.foot-mark svg{color:var(--brass);display:block;}
.foot-tag{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;font-size:clamp(1.1rem,2.4vw,1.6rem);color:var(--text);margin:.5rem 0 .4rem;letter-spacing:-.01em;}
.foot-note{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:var(--muted);letter-spacing:.04em;line-height:1.7;max-width:760px;margin:0 auto;}
html, body, .stApp, [data-testid="stAppViewContainer"]{background-color:var(--ink)!important;}
header[data-testid="stHeader"]{background:transparent!important;}
section[data-testid="stSidebar"], section[data-testid="stSidebar"]>div{background-color:var(--ink-2)!important;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="ambient"><div class="g g1"></div><div class="g g2"></div>'
    '<div class="g g3"></div></div>',
    unsafe_allow_html=True,
)


# ============================================================
# Build the archive (cached; de-duplicated; self-healing index)
# ============================================================
@dataclass
class _Archive:
    retriever: Retriever
    pipeline: RAGPipeline
    vector_count: int
    num_records: int
    num_unique: int
    num_chunks: int
    gold: list


@st.cache_resource(show_spinner=False)
def _build_archive() -> _Archive:
    settings = get_settings()

    loader = DataLoader(settings.dataset, settings.paths)
    try:
        loader.load()
        docs = loader.documents("train") or loader.documents(loader.splits()[0])
    except DatasetLoadingError:
        loader.load_from_records(_FALLBACK_RECORDS, validate=True)
        docs = loader.documents("train")

    gold = [
        {
            "question": d.question,
            "gold_answers": list(d.answers),
            "is_impossible": not d.metadata.get("has_answer", False),
        }
        for d in docs
        if d.metadata.get("has_answer", False) and d.answers
    ]

    seen_ctx = set()
    records_for_proc = []
    for d in docs:
        ctx = d.context.strip()
        if not ctx or ctx in seen_ctx:
            continue
        seen_ctx.add(ctx)
        records_for_proc.append(
            {
                "document_id": d.id,
                "text": d.context,
                "metadata": {**d.metadata, "title": d.title, "question": d.question},
            }
        )

    processed = TextProcessor(settings.chunking).process_documents(records_for_proc)
    chunks = Chunker(settings.chunking).chunk_documents(processed)

    embedder = EmbeddingGenerator(settings.embedding)
    manager = VectorStoreManager(settings.vector_store)
    manager.connect()

    expected = len(chunks)
    if manager.count() != expected:
        manager.reset()
        emb_records = embedder.encode_chunks(chunks, show_progress=False)
        manager.add_records(emb_records, [c.text for c in chunks])

    retriever = Retriever(vector_store=manager, embedding_generator=embedder)
    pipeline = RAGPipeline(retriever=retriever)

    return _Archive(
        retriever=retriever,
        pipeline=pipeline,
        vector_count=manager.count(),
        num_records=len(docs),
        num_unique=len(seen_ctx),
        num_chunks=expected,
        gold=gold,
    )


def _build_llm(model_name: str, api_key: str) -> GroqLLMClient:
    return GroqLLMClient(config=LLMConfig(model_name=model_name), api_key=api_key)


# ============================================================
# Render the persisted answer (so citation / feedback never wipe it)
# ============================================================
def _render_answer(slot, q, result, clean_pages, clean_meta, clean_sim):
    """Draw the answer card + evidence + actions into ``slot``."""
    confidence = estimate_retrieval_confidence(clean_sim)
    stamp_class = {
        "High": "stamp-high",
        "Medium": "stamp-medium",
        "Low": "stamp-low",
    }[confidence["label"]]

    with slot.container():
        st.markdown(
            f"""
            <div class="answer-card">
              <div class="catalog-no">FILED UNDER — Q&amp;A No. {abs(hash(q)) % 9000 + 1000}</div>
              <div class="answer-text">{_html.escape(result["answer"])}</div>
              <div class="stamp {stamp_class}">{confidence["label"]} confidence</div>
              <div class="timing-row">
                <div class="timing-chip">⏱ retrieval {result["retrieval_time_sec"]}s</div>
                <div class="timing-chip">⏱ generation {result["generation_time_sec"]}s</div>
                <div class="timing-chip">⏱ total {result["total_time_sec"]}s</div>
                <div class="timing-chip">avg sim {confidence["avg_similarity"]:.3f}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="evidence-title">Evidence trail</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="evidence-sub">{len(clean_pages)} unique passage(s) '
            "retrieved, ranked by similarity</div>",
            unsafe_allow_html=True,
        )

        if clean_pages:
            for i, (chunk, meta, sim) in enumerate(
                zip(clean_pages, clean_meta, clean_sim, strict=True),
                start=1,
            ):
                pct = max(0.0, min(1.0, (sim + 1) / 2)) * 100
                clean_text = chunk.replace("\n", " ").strip()
                lead, trail = _citation_edges(clean_text)
                full = lead + _html.escape(clean_text) + trail
                title = _html.escape(str(meta.get("title") or "Untitled"))
                st.markdown(
                    f"""
                    <div class="index-card">
                      <div class="row-top">
                        <span class="rank">CARD {i:02d} · {title}</span>
                        <span class="sim-score">similarity {sim:.3f}</span>
                      </div>
                      <div class="sim-track"><div class="sim-fill" style="width:{pct:.1f}%"></div></div>
                      <div class="snippet">{full}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="no-evidence">No retrieved passage passed the '
                "similarity floor — raise “Top-K” or lower “Min evidence "
                "similarity” to see more.</div>",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="action-row">', unsafe_allow_html=True)
        cc, cu, cd, _ = st.columns([3, 1, 1, 5])
        with cc:
            if st.button("📋 Copy as citation", key=f"cite_{q}"):
                st.session_state["show_cite"] = (
                    None if st.session_state.get("show_cite") == q else q
                )
        with cu:
            if st.button("👍", key=f"up_{q}"):
                _log_feedback(q, result["answer"], "up")
                st.session_state[f"fb_{q}"] = "👍 Thanks — glad it helped."
        with cd:
            if st.button("👎", key=f"down_{q}"):
                _log_feedback(q, result["answer"], "down")
                st.session_state[f"fb_{q}"] = "👎 Noted — we'll improve this."
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.get("show_cite") == q:
            st.code(
                _build_citation(q, result, clean_pages, clean_meta, clean_sim),
                language="markdown",
            )

        if st.session_state.get(f"fb_{q}"):
            st.caption(st.session_state[f"fb_{q}"])


# ============================================================
# Top bar + hero
# ============================================================
st.markdown(
    """
    <div class="topbar">
      <div class="brand">
        <div class="mark"><svg viewBox="0 0 32 32" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 3 L27 7 V15 C27 22 22 27 16 29 C10 27 5 22 5 15 V7 Z"/><path d="M11 15.5 L14.5 19 L21 11.5"/></svg></div>
        <div class="name">VERIDEX</div>
      </div>
      <div class="brand-tag">Evidence Engine</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="eyebrow">Evidence-grounded retrieval · nocturnal console</div>
    <h1 class="hero-title">Ask the<br><em>Archive</em>
      <svg class="uline" viewBox="0 0 420 14" preserveAspectRatio="none">
        <path d="M3 9 Q120 2 210 7 T417 6"/>
      </svg>
    </h1>
    <p class="hero-sub">
      <strong style="color:var(--text)">VERIDEX</strong> is an evidence engine: it turns a
      document collection into answers you can trust — each one retrieved from a real passage,
      ranked by meaning, and stamped with how confident the archive is. Where the record is
      silent, VERIDEX stays silent too. That is not a limitation; that is the product.
    </p>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Build (with a branded loading / error state)
# ============================================================
load_slot = st.empty()
load_slot.markdown(
    """
    <div class="load-wrap">
      <div class="load-title">Opening the archive…</div>
      <div class="load-sub">first run embeds the (de-duplicated) corpus &amp; builds the index · later runs are instant</div>
      <div class="load-bar"><i></i></div>
    </div>
    """,
    unsafe_allow_html=True,
)

build_error = None
try:
    archive = _build_archive()
except Exception as exc:  # noqa: BLE001
    archive = None
    build_error = str(exc)
finally:
    load_slot.empty()

if build_error is not None:
    st.markdown(
        f"""
        <div class="error-card">
          ⚠ <strong>The archive could not finish building.</strong><br>
          {_html.escape(build_error)}<br><br>
          If this follows a code change, delete the <code>chroma_db</code> folder once and
          reload — the index will rebuild cleanly.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ============================================================
# Provenance strip
# ============================================================
st.markdown(
    f"""
    <div class="prov">
      <div class="node"><div class="k">Ingest</div><div class="v">{archive.num_records:,} <small>questions</small></div></div>
      <div class="node"><div class="k">Unique</div><div class="v">{archive.num_unique:,} <small>passages</small></div></div>
      <div class="node"><div class="k">Chunk</div><div class="v">{archive.num_chunks:,} <small>spans</small></div></div>
      <div class="node"><div class="k">Embed</div><div class="v">384<small>-d · bge</small></div></div>
      <div class="node"><div class="k">Index</div><div class="v">{archive.vector_count:,} <small>cosine</small></div></div>
      <div class="node"><div class="k">Retrieve</div><div class="v">top-k <small>scored</small></div></div>
      <div class="node"><div class="k">Generate</div><div class="v">Groq <small>grounded</small></div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<hr class="rule">', unsafe_allow_html=True)


# ============================================================
# Sidebar — catalog controls
# ============================================================
with st.sidebar:
    st.markdown(
        '<div class="side-brand"><svg viewBox="0 0 32 32" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 3 L27 7 V15 C27 22 22 27 16 29 C10 27 5 22 5 15 V7 Z"/><path d="M11 15.5 L14.5 19 L21 11.5"/></svg> CATALOG</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-label">Groq API key</div>', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_…",
        label_visibility="collapsed",
        key="groq_key",
        help="Free, no credit card — console.groq.com/keys",
    )
    st.caption("🔗 [Get a free key](https://console.groq.com/keys) — no billing, ever.")

    st.markdown(
        '<div class="sidebar-label">Retrieval depth</div>', unsafe_allow_html=True
    )
    top_k = st.slider("Top-K passages", 1, 10, 5, label_visibility="collapsed")

    st.markdown(
        '<div class="sidebar-label">Min evidence similarity</div>',
        unsafe_allow_html=True,
    )
    min_sim_floor = st.slider(
        "Hide evidence below",
        0.0,
        0.9,
        0.0,
        0.05,
        label_visibility="collapsed",
        help="0 = show every unique passage the model saw. Raise it to drop weak matches.",
    )

    st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox(
        "Model",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Conversation</div>', unsafe_allow_html=True)
    conv_mode = st.toggle(
        "Memory across follow-ups",
        value=True,
        help="Lets you ask 'where was he born?' after a previous answer.",
    )
    if st.button("🧹 New session", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-label">Try one of these</div>', unsafe_allow_html=True
    )
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    for q in [
        "Who managed Destiny's Child?",
        "What is the iPod?",
        "What is solar energy?",
    ]:
        st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
        if st.button(q, key=f"chip_{q}"):
            st.session_state.question_input = q
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Benchmark</div>', unsafe_allow_html=True)
    eval_n = st.slider(
        "Eval questions",
        3,
        20,
        8,
        label_visibility="collapsed",
        help="Each question calls Groq once.",
    )
    run_eval = st.button("▶ Run benchmark", use_container_width=True)

    st.markdown('<div class="sidebar-label">Corpus</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="stat-line"><span>Questions loaded</span><b>{archive.num_records:,}</b></div>
        <div class="stat-line"><span>Unique passages</span><b>{archive.num_unique:,}</b></div>
        <div class="stat-line"><span>Vectors indexed</span><b>{archive.vector_count:,}</b></div>
        <div class="stat-line"><span>Embedding</span><b>bge-small-en</b></div>
        <div class="stat-line"><span>Distance</span><b>cosine</b></div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Optional benchmark panel
# ============================================================
if run_eval:
    if not (api_key_input or os.environ.get("GROQ_API_KEY", "")).strip():
        st.markdown(
            '<div class="error-card">⚠ Paste a Groq key in the sidebar (or set '
            "GROQ_API_KEY) to run the benchmark.</div>",
            unsafe_allow_html=True,
        )
    elif not archive.gold:
        st.markdown(
            '<div class="error-card">⚠ No answerable gold questions available in this corpus.</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.spinner(
            f"Evaluating {min(eval_n, len(archive.gold))} questions against ground truth…"
        ):
            try:
                llm = _build_llm(
                    model_name,
                    api_key_input or os.environ.get("GROQ_API_KEY", ""),
                )
                eval_pipeline = RAGPipeline(retriever=archive.retriever, llm_client=llm)
                engine = EvaluationEngine(pipeline=eval_pipeline)
                report = engine.evaluate(archive.gold[:eval_n])
                s = report.summary()
                st.markdown(
                    '<div class="evidence-title">Live benchmark</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="eval-grid">
                      <div class="eval-tile"><div class="k">Exact Match</div><div class="v">{s["exact_match"]:.2f}</div></div>
                      <div class="eval-tile"><div class="k">Token F1</div><div class="v">{s["f1"]:.2f}</div></div>
                      <div class="eval-tile"><div class="k">Hit@{top_k}</div><div class="v">{(s["retrieval_hit_rate"] or 0):.2f}</div></div>
                      <div class="eval-tile"><div class="k">MRR</div><div class="v">{(s["mrr"] or 0):.2f}</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"evaluated {s['num_evaluated']} · errors {s['num_errors']} · "
                    f"avg total {s['avg_total_time_sec']:.2f}s · "
                    f"refusal recall {(s['refusal_recall'] or 0):.2f}"
                )
            except Exception as exc:  # noqa: BLE001
                st.markdown(
                    f'<div class="error-card">⚠ {_html.escape(str(exc))}</div>',
                    unsafe_allow_html=True,
                )


# ============================================================
# Conversation thread (visible history for follow-ups)
# ============================================================
if len(st.session_state.chat_history) > 1:
    with st.expander(
        f"💬 Conversation so far — {len(st.session_state.chat_history)} turns",
        expanded=True,
    ):
        st.markdown('<div class="thread">', unsafe_allow_html=True)
        for i, turn in enumerate(st.session_state.chat_history[:-1], 1):
            st.markdown(
                f'<div class="thread-q">Q{i} · {_html.escape(turn["q"])}</div>'
                f'<div class="thread-a">{_html.escape(turn["a"])}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Search drawer
# ============================================================
col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        "Your question",
        placeholder="e.g. Who managed Destiny's Child?",
        label_visibility="collapsed",
        key="question_input",
    )
with col_btn:
    ask_clicked = st.button("Search →", use_container_width=True)

result_slot = st.empty()

if ask_clicked and question.strip():
    if not (api_key_input or os.environ.get("GROQ_API_KEY", "")).strip():
        result_slot.markdown(
            '<div class="error-card">⚠ Enter a free Groq API key in the sidebar first — '
            'get one instantly at <a href="https://console.groq.com/keys" '
            'style="color:#e7b3ae">console.groq.com/keys</a>.</div>',
            unsafe_allow_html=True,
        )
    else:
        loading = st.empty()
        loading.markdown(
            '<div class="load-wrap" style="padding:1.4rem 0">'
            '<div class="load-sub">consulting the archive…</div>'
            '<div class="load-bar"><i></i></div></div>',
            unsafe_allow_html=True,
        )
        try:
            llm = _build_llm(
                model_name,
                api_key_input or os.environ.get("GROQ_API_KEY", ""),
            )
            pipeline_question = (
                _augment_question(question, st.session_state.chat_history)
                if conv_mode
                else question
            )
            result = archive.pipeline.answer_question(
                pipeline_question, top_k=top_k, llm_client=llm
            )
        except PipelineConnectionError as exc:
            result = {
                "error": str(exc),
                "answer": "",
                "retrieved_pages": [],
                "metadata": [],
                "similarities": [],
                "retrieval_time_sec": 0.0,
                "generation_time_sec": 0.0,
                "total_time_sec": 0.0,
            }
        except Exception as exc:  # noqa: BLE001
            result = {
                "error": str(exc),
                "answer": "",
                "retrieved_pages": [],
                "metadata": [],
                "similarities": [],
                "retrieval_time_sec": 0.0,
                "generation_time_sec": 0.0,
                "total_time_sec": 0.0,
            }
        finally:
            loading.empty()

        if result.get("error"):
            result_slot.markdown(
                f'<div class="error-card">⚠ {_html.escape(str(result["error"]))}</div>',
                unsafe_allow_html=True,
            )
        else:
            seen_pages = set()
            clean_pages = []
            clean_meta = []
            clean_sim = []
            for page, meta, sim in zip(
                result["retrieved_pages"],
                result["metadata"],
                result["similarities"],
                strict=True,
            ):
                if sim < min_sim_floor:
                    continue
                if page in seen_pages:
                    continue
                seen_pages.add(page)
                clean_pages.append(page)
                clean_meta.append(meta)
                clean_sim.append(sim)

            st.session_state["last_q"] = question
            st.session_state["last_result"] = result
            st.session_state["last_clean"] = (clean_pages, clean_meta, clean_sim)
            st.session_state["show_cite"] = None

            if conv_mode:
                h = st.session_state.chat_history
                if not h or h[-1]["q"] != question:
                    h.append({"q": question, "a": result["answer"]})
                st.session_state.chat_history = h[-6:]

elif ask_clicked:
    result_slot.markdown(
        '<div class="error-card">Type a question before searching the archive.</div>',
        unsafe_allow_html=True,
    )
    st.session_state["last_result"] = None

# ---- render the persisted answer (runs on every rerun) ----
if st.session_state.get("last_result") is not None:
    _render_answer(
        result_slot,
        st.session_state["last_q"],
        st.session_state["last_result"],
        *st.session_state["last_clean"],
    )


# ============================================================
# Footer — brand line + useful note (no repetition)
# ============================================================
st.markdown('<hr class="rule">', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="foot">
      <div class="foot-mark"><svg viewBox="0 0 32 32" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 3 L27 7 V15 C27 22 22 27 16 29 C10 27 5 22 5 15 V7 Z"/><path d="M11 15.5 L14.5 19 L21 11.5"/></svg> VERIDEX</div>
      <div class="foot-tag">Evidence over memory.</div>
      <div class="foot-note">A domain-agnostic retrieval engine — SQuAD v2 is the first corpus wired in. Swap in legal, medical, or enterprise documents and the same grounded pipeline answers from them, without touching the core. Model · Groq ({model_name}) · vectors · Chroma / cosine · embedding · bge-small-en-v1.5.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

import os
import re
import json
import warnings
import numpy as np
import transformers
import streamlit as st
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline as hf_pipeline
from ddgs import DDGS

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent BI Pipeline",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Multi-Agent Business Intelligence Pipeline")
st.caption(
    "Orchestrator → Research Agent → ML Layer → Analyst → Writer | "
    "Powered by DeepSeek-V3-0324 + local DL models"
)

# ─── Config ───────────────────────────────────────────────────────────────────
GEN_MODEL = "deepseek-ai/DeepSeek-V3-0324"

# Allow key from env (for HF Spaces secrets) or from the sidebar input
def get_api_key() -> str:
    return os.environ.get("HF_API_KEY", "") or st.session_state.get("hf_key", "")

# ─── Pydantic models ──────────────────────────────────────────────────────────
class ResearchPlan(BaseModel):
    tasks: list[str]
    focus_areas: list[str]

class ResearchFinding(BaseModel):
    query: str
    content: str

class MLInsights(BaseModel):
    sentiment: dict
    clusters: dict
    entities: dict
    top_findings: list[str]

class SWOTAnalysis(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    opportunities: list[str]
    threats: list[str]

class Analysis(BaseModel):
    swot: SWOTAnalysis
    key_metrics: list[str]
    competitive_landscape: str
    main_insights: list[str]

# ─── LLM helpers ──────────────────────────────────────────────────────────────
def get_client() -> InferenceClient:
    return InferenceClient(api_key=get_api_key())

def chat(messages: list, max_tokens: int = 1024, temperature: float = 0.1) -> str:
    response = get_client().chat.completions.create(
        model=GEN_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content

def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?[^\n]*\n?", "", text).strip(" `\n")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found:\n{text[:400]}")

def chat_json(user_prompt: str, template: str, max_tokens: int = 1024) -> dict:
    prompt = (
        f"{user_prompt}\n\n"
        "Complete the following JSON by replacing every placeholder value "
        "with real content. Output ONLY the completed JSON — no extra text:\n\n"
        f"{template}"
    )
    raw = chat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return extract_json(raw)

# ─── ML Processing Layer (cached — loaded once per session) ───────────────────
@st.cache_resource(show_spinner="Loading ML models (one-time ~30s)…")
def load_ml_layer():
    class MLProcessingLayer:
        def __init__(self):
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.sentiment_pipe = hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True, max_length=512,
            )
            self.ner_pipe = hf_pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
            )

        def process(self, query: str, findings: list) -> MLInsights:
            chunks = [f.content for f in findings]
            if not chunks:
                return MLInsights(sentiment={}, clusters={}, entities={}, top_findings=[])

            embeddings = self.embedder.encode(chunks, show_progress_bar=False)
            query_emb  = self.embedder.encode([query])
            sim_scores   = cosine_similarity(query_emb, embeddings)[0]
            ranked_idx   = np.argsort(sim_scores)[::-1]
            top_findings = [chunks[i] for i in ranked_idx[:5]]

            n_clusters = min(4, len(chunks))
            labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(embeddings)
            clusters: dict = {}
            for i, label in enumerate(labels.tolist()):
                clusters.setdefault(f"theme_{label}", []).append(chunks[i][:180] + "…")

            results = [self.sentiment_pipe(c[:512])[0] for c in chunks]
            pos = sum(1 for r in results if r["label"] == "POSITIVE") / len(results)
            neg = sum(1 for r in results if r["label"] == "NEGATIVE") / len(results)
            sentiment = {
                "positive": round(pos, 2),
                "negative": round(neg, 2),
                "neutral":  round(max(0.0, 1.0 - pos - neg), 2),
            }

            entities: dict = {"ORG": set(), "PER": set(), "LOC": set()}
            for chunk in chunks[:6]:
                for ent in self.ner_pipe(chunk[:512]):
                    grp = ent["entity_group"]
                    if grp in entities:
                        entities[grp].add(ent["word"])

            return MLInsights(
                sentiment=sentiment,
                clusters=clusters,
                entities={k: list(v)[:8] for k, v in entities.items()},
                top_findings=top_findings,
            )

    return MLProcessingLayer()

# ─── Agents ───────────────────────────────────────────────────────────────────
_ANALYSIS_TEMPLATE = """{
  "swot": {
    "strengths": ["strength 1", "strength 2", "strength 3"],
    "weaknesses": ["weakness 1", "weakness 2", "weakness 3"],
    "opportunities": ["opportunity 1", "opportunity 2", "opportunity 3"],
    "threats": ["threat 1", "threat 2", "threat 3"]
  },
  "key_metrics": ["metric 1", "metric 2", "metric 3", "metric 4"],
  "competitive_landscape": "description of competitive landscape",
  "main_insights": ["insight 1", "insight 2", "insight 3", "insight 4"]
}"""

def _plan_research(query: str) -> ResearchPlan:
    prompt = (
        f"Create a research plan for: {query}\n\n"
        "Write 7 specific multi-word web search queries, numbered 1–7.\n"
        "Each query must be a full phrase, NOT a single generic word.\n"
        "Cover: revenue/valuation, main competitors, products/services, "
        "market share, recent news, funding, and regulatory challenges.\n\n"
        "1. [full search query about revenue/valuation]\n"
        "2. [full search query about main competitors]\n"
        "3. [full search query about products/services]\n"
        "4. [full search query about market share]\n"
        "5. [full search query about recent news]\n"
        "6. [full search query about funding]\n"
        "7. [full search query about regulation/challenges]"
    )
    raw = chat([{"role": "user", "content": prompt}], max_tokens=600, temperature=0.0)
    tasks = re.findall(r'^\s*\d+[\.\)]\s+(.+)$', raw, re.MULTILINE)
    tasks = [t for t in tasks if len(t.split()) >= 3]
    if len(tasks) < 3:
        tasks = [l.strip() for l in raw.splitlines() if len(l.strip().split()) >= 3][:8]
    return ResearchPlan(tasks=tasks[:8], focus_areas=["financials", "competitors", "products", "market trends"])

def research_agent(tasks: list[str]) -> list:
    findings = []
    for task in tasks:
        with DDGS() as ddgs:
            results = list(ddgs.text(task, max_results=6))
        raw = "\n\n".join(f"{r['title']}: {r['body']}" for r in results if r.get("body"))
        if not raw:
            continue
        summary = chat(
            messages=[
                {"role": "system", "content": "You are a research analyst. Summarise web snippets into 3-5 key facts and insights."},
                {"role": "user", "content": f"Search query: {task}\n\nWeb snippets:\n{raw[:3000]}"},
            ],
            max_tokens=512,
        )
        if summary:
            findings.append(ResearchFinding(query=task, content=summary))
    return findings

def analyst_agent(query: str, findings: list, ml: MLInsights) -> Analysis:
    research_text = "\n\n---\n\n".join(f"QUERY: {f.query}\n{f.content}" for f in findings)
    ml_context = (
        f"\nML INSIGHTS:\n"
        f"  Sentiment  — {ml.sentiment.get('positive', 0)*100:.0f}% positive, "
        f"{ml.sentiment.get('negative', 0)*100:.0f}% negative\n"
        f"  Clusters   — {len(ml.clusters)} themes\n"
        f"  Orgs (NER) — {', '.join(ml.entities.get('ORG', [])[:6])}\n"
        f"  People     — {', '.join(ml.entities.get('PER', [])[:5])}\n"
    )
    prompt = f"You are a senior business analyst. Analyse: {query}\n\nRESEARCH:\n{research_text[:5000]}\n\n{ml_context}"
    data = chat_json(prompt, template=_ANALYSIS_TEMPLATE, max_tokens=3000)
    return Analysis(
        swot=SWOTAnalysis(**data["swot"]),
        key_metrics=data["key_metrics"],
        competitive_landscape=data["competitive_landscape"],
        main_insights=data["main_insights"],
    )

def writer_agent_stream(query: str, analysis: Analysis, ml: MLInsights):
    """Generator that yields tokens from the writer agent."""
    swot = analysis.swot
    prompt = (
        f"Write a professional Business Intelligence Report for: **{query}**\n\n"
        "SWOT ANALYSIS:\n"
        "Strengths:\n"     + "\n".join(f"  - {s}" for s in swot.strengths)     + "\n"
        "Weaknesses:\n"    + "\n".join(f"  - {s}" for s in swot.weaknesses)    + "\n"
        "Opportunities:\n" + "\n".join(f"  - {s}" for s in swot.opportunities) + "\n"
        "Threats:\n"       + "\n".join(f"  - {s}" for s in swot.threats)       + "\n\n"
        "KEY METRICS:\n"   + "\n".join(f"  * {m}" for m in analysis.key_metrics) + "\n\n"
        f"COMPETITIVE LANDSCAPE:\n{analysis.competitive_landscape}\n\n"
        "ML-POWERED INSIGHTS:\n"
        f"  Sentiment: {ml.sentiment.get('positive', 0)*100:.0f}% positive, "
        f"{ml.sentiment.get('negative', 0)*100:.0f}% negative\n"
        f"  {len(ml.clusters)} research themes via KMeans\n"
        f"  Key organisations (BERT NER): {', '.join(ml.entities.get('ORG', [])[:6])}\n\n"
        "Write the full report with these sections:\n"
        "1. Executive Summary\n2. ML-Powered Sentiment & Theme Analysis\n"
        "3. SWOT Analysis\n4. Competitive Landscape\n"
        "5. Key Metrics & Data Points\n6. Strategic Recommendations\n\n"
        "Be concise, data-driven, and professional."
    )
    stream = get_client().chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert business intelligence analyst. Write clear, data-driven reports."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.3,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        yield token

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    env_key = os.environ.get("HF_API_KEY", "")
    if env_key:
        st.success("HF_API_KEY loaded from environment.")
    else:
        hf_key = st.text_input(
            "HuggingFace API Key",
            type="password",
            placeholder="hf_...",
            help="Get yours at huggingface.co/settings/tokens",
        )
        st.session_state["hf_key"] = hf_key

    st.divider()
    st.markdown("**Pipeline stages:**")
    st.markdown("""
1. 🧭 Orchestrator (query planning)
2. 🔎 Research Agent (DuckDuckGo)
3. 🧠 ML Layer (embeddings, clustering, NER)
4. 📊 Analyst Agent (SWOT)
5. ✍️ Writer Agent (report)
""")
    st.divider()
    st.caption("Model: DeepSeek-V3-0324 via HF Inference API")
    st.caption("Local models: MiniLM · DistilBERT · BERT-NER")

# ─── Main form ────────────────────────────────────────────────────────────────
query = st.text_area(
    "Enter your research query",
    placeholder="e.g. Analyze OpenAI's competitive position in the AI industry",
    height=100,
)

run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

if run_btn:
    if not get_api_key():
        st.error("Please enter your HuggingFace API key in the sidebar.")
        st.stop()
    if not query.strip():
        st.error("Please enter a query.")
        st.stop()

    # Load ML models (cached)
    ml_layer = load_ml_layer()

    # ── Step 1: Orchestrator ──────────────────────────────────────────────────
    with st.status("🧭 Step 1/5 — Orchestrator: planning research tasks…", expanded=True) as status:
        plan = _plan_research(query)
        st.write(f"Generated **{len(plan.tasks)}** search tasks:")
        for t in plan.tasks:
            st.write(f"  · {t}")
        status.update(label="✅ Step 1/5 — Orchestrator done", state="complete", expanded=False)

    # ── Step 2: Research Agent ────────────────────────────────────────────────
    with st.status(f"🔎 Step 2/5 — Research Agent: searching {len(plan.tasks)} topics…", expanded=True) as status:
        task_placeholder = st.empty()
        findings = []
        for task in plan.tasks:
            task_placeholder.write(f"Searching: *{task}*")
            with DDGS() as ddgs:
                results = list(ddgs.text(task, max_results=6))
            raw = "\n\n".join(f"{r['title']}: {r['body']}" for r in results if r.get("body"))
            if not raw:
                continue
            summary = chat(
                messages=[
                    {"role": "system", "content": "Summarise web snippets into 3-5 key facts."},
                    {"role": "user", "content": f"Query: {task}\n\nSnippets:\n{raw[:3000]}"},
                ],
                max_tokens=512,
            )
            if summary:
                findings.append(ResearchFinding(query=task, content=summary))
        task_placeholder.empty()
        st.write(f"Gathered **{len(findings)}** findings.")
        status.update(label=f"✅ Step 2/5 — Research done ({len(findings)} findings)", state="complete", expanded=False)

    # ── Step 3: ML Layer ──────────────────────────────────────────────────────
    with st.status("🧠 Step 3/5 — ML Layer: embeddings / clustering / sentiment / NER…", expanded=True) as status:
        ml_insights = ml_layer.process(query, findings)
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive sentiment", f"{ml_insights.sentiment.get('positive', 0)*100:.0f}%")
        col2.metric("Negative sentiment", f"{ml_insights.sentiment.get('negative', 0)*100:.0f}%")
        col3.metric("Themes discovered", len(ml_insights.clusters))
        orgs = ml_insights.entities.get("ORG", [])
        if orgs:
            st.write("**Key organisations (NER):** " + ", ".join(orgs))
        status.update(label="✅ Step 3/5 — ML Layer done", state="complete", expanded=False)

    # ── Step 4: Analyst Agent ─────────────────────────────────────────────────
    with st.status("📊 Step 4/5 — Analyst Agent: structured SWOT analysis…", expanded=True) as status:
        analysis = analyst_agent(query, findings, ml_insights)
        st.write(f"Extracted **{len(analysis.main_insights)}** key insights.")
        status.update(label="✅ Step 4/5 — Analysis done", state="complete", expanded=False)

    # ── Step 5: Writer Agent (streaming) ─────────────────────────────────────
    st.subheader("📄 Business Intelligence Report")
    report_placeholder = st.empty()
    full_report = ""
    with st.status("✍️ Step 5/5 — Writer Agent: generating report…", expanded=False) as status:
        for token in writer_agent_stream(query, analysis, ml_insights):
            full_report += token
            report_placeholder.markdown(full_report + "▌")
        report_placeholder.markdown(full_report)
        status.update(label="✅ Step 5/5 — Report complete", state="complete")

    # ── Download button ───────────────────────────────────────────────────────
    st.download_button(
        label="⬇️ Download Report (Markdown)",
        data=full_report,
        file_name="bi_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

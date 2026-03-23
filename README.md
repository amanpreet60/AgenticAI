# Company Insight

Give it a company name. Get back a full business intelligence report.

**[Try it live →](https://huggingface.co/spaces/aman2200/Company_insight)**

---

## What it does

You type something like *"Analyze Tesla's competitive position"* and the app runs a 5-step pipeline behind the scenes:

1. Breaks your query into 7 targeted search questions
2. Searches the web (DuckDuckGo) and summarizes what it finds
3. Runs the text through local ML models — sentiment, clustering, named entity recognition
4. Synthesizes everything into a SWOT analysis
5. Writes a full professional report, streamed token by token

The final report covers an executive summary, competitive landscape, key metrics, and strategic recommendations. You can download it as Markdown when it's done.

---

## Stack

- **DeepSeek-V3.2** via HuggingFace Inference API — for planning, summarizing, analyzing, and writing
- **MiniLM** — sentence embeddings and semantic ranking
- **DistilBERT** — sentiment analysis
- **BERT-NER** — extracting organizations, people, and locations
- **KMeans** — grouping research findings into themes
- **DuckDuckGo** — web search, no API key needed
- **Streamlit** — the UI

---

## Running locally

You'll need a [HuggingFace API key](https://huggingface.co/settings/tokens).

```bash
git clone https://github.com/<your-username>/AgenticAI.git
cd AgenticAI
pip install -r requirements.txt
streamlit run app.py
```

Paste your key in the sidebar and you're good to go.

---

## Deploying to HuggingFace Spaces

Create a Streamlit Space, push this repo, and add `HF_API_KEY` as a Space secret. The app picks it up automatically.

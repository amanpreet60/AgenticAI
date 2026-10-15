# Company Insight

Give it a company name. Get back a full business intelligence report.

## Try it

- **Live app:** [huggingface.co/spaces/aman2200/Company_insight](https://huggingface.co/spaces/aman2200/Company_insight)
- **Notebook:** [open `index.ipynb` in Google Colab](https://colab.research.google.com/github/amanpreet60/AgenticAI/blob/main/index.ipynb) — set `ANTHROPIC_API_KEY` in the environment before running the setup cell

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

- **Claude Opus 5.5** via the Anthropic API — for planning, summarizing, analyzing, and writing
- **MiniLM** — sentence embeddings and semantic ranking
- **DistilBERT** — sentiment analysis
- **BERT-NER** — extracting organizations, people, and locations
- **KMeans** — grouping research findings into themes
- **DuckDuckGo** — web search, no API key needed
- **Streamlit** — the UI

---

## Running locally

You'll need an [Anthropic API key](https://console.anthropic.com/settings/keys).

```bash
git clone https://github.com/amanpreet60/AgenticAI.git
cd AgenticAI
pip install -r requirements.txt
streamlit run app.py
```

Either paste your key in the sidebar, or create a `.env` file in the project root (it's gitignored):

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Deploying to HuggingFace Spaces

Create a Streamlit Space, push this repo, and add `ANTHROPIC_API_KEY` as a Space secret. The app picks it up automatically.

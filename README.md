# Fake News Detection – News Credibility Verification

A **local-first Python Streamlit application** that verifies news claims by cross-referencing **Reddit**, **Wikipedia**, and **web/news sources**. It computes an explainable credibility score (0–100) and presents evidence in an interactive UI.

---

## Features

- 🔍 **Real-time claim verification** — enter any news claim and get immediate results
- 📊 **Explainable credibility score** (0–100) with signal breakdown
- 🏷️ **Verdict labels** — *Likely True*, *Unclear*, or *Likely False*
- 🌐 **Multi-source cross-referencing**:
  - **Reddit** — community discussion and corroboration (PRAW, read-only)
  - **Wikipedia** — entity and background validation
  - **Web / News** — GDELT real-time news events + RSS feeds from Reuters, BBC, AP, NPR, Guardian
- 🔬 **Explainability panel** — per-signal contribution with rationale
- 🛡️ **Graceful degradation** — works even when individual sources are unavailable or rate-limited
- ⚙️ **Configurable** — weights, thresholds, and source toggles via `.env`

---

## Project Structure

```
fake-news-detection-using-social-media-crossreferencing/
├── app.py                        # Streamlit UI entry point
├── requirements.txt
├── .env.example                  # Template for environment variables
├── README.md
├── LICENSE
└── src/
    ├── __init__.py
    ├── config.py                 # Settings, weights, thresholds
    ├── collectors/
    │   ├── __init__.py
    │   ├── reddit_collector.py   # PRAW-based Reddit search
    │   ├── wikipedia_collector.py# Wikipedia API / entity lookup
    │   └── web_collector.py      # GDELT + RSS feeds
    ├── nlp/
    │   ├── __init__.py
    │   └── claim_normalizer.py   # Keyword extraction, entity detection
    └── scoring/
        ├── __init__.py
        └── credibility.py        # Weighted signal scoring engine
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/milindborse/fake-news-detection-using-social-media-crossreferencing.git
cd fake-news-detection-using-social-media-crossreferencing
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your Reddit API credentials:

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=news-credibility-checker/0.1 by your_reddit_username
```

> **Getting free Reddit API credentials:**
> 1. Go to https://www.reddit.com/prefs/apps
> 2. Click **"create another app"**
> 3. Select **"script"** type
> 4. Fill in name + description, set redirect URI to `http://localhost:8080`
> 5. Copy the `client_id` (under app name) and `client_secret`

> **Wikipedia and Web sources do not require any API keys** — they work out of the box.

---

## Running the App

```bash
streamlit run app.py
```

The app opens in your browser at **http://localhost:8501**.

---

## Sample Workflow

1. Enter a news claim, e.g.:
   > *"Scientists discover new species of giant dinosaur in Patagonia"*
2. (Optional) Paste the source article URL for reference.
3. Click **🚀 Verify Claim**.
4. The app will:
   - Extract keywords and entities from your claim
   - Search Reddit, Wikipedia, and web/news sources
   - Compute a weighted credibility score
   - Display evidence cards grouped by source
   - Show an explainability panel with per-signal contributions
5. Review the verdict: **Likely True** / **Unclear** / **Likely False**

---

## Scoring Signals

| Signal | Default Weight | Description |
|---|---|---|
| Independent Corroboration | 35% | Distinct sources/domains that mention the claim |
| Contradiction / Debunking | 30% | Evidence containing hoax/fake/debunked keywords |
| Source Quality | 20% | Reliable domain whitelist + Reddit quality heuristics |
| Recency Relevance | 15% | Recency of evidence (recent = more relevant) |

All weights are configurable via `.env`.

---

## Limitations & Ethical Disclaimer

- **Not a fact-checker:** This tool provides automated credibility *signals*, not verified facts. It should supplement, not replace, expert fact-checking.
- **API dependencies:** Reddit analysis requires valid API credentials. Web search depends on GDELT availability and RSS feed accessibility.
- **No real-time social media:** Twitter/X is not included due to API cost restrictions; Reddit is used as a proxy for social discussion.
- **Bias in sources:** The reliable-domain whitelist and quality subreddits reflect English-language, Western-centric sources. Non-English claims may score lower due to limited coverage.
- **No persistence:** Analysis results are not stored. Each run is stateless.
- **Rate limits:** Heavy usage may trigger rate limits on Reddit or GDELT. The app handles this gracefully by degrading to available sources.
- **Intended use:** This tool is for educational and research purposes. Do not use it as the sole basis for editorial, legal, or policy decisions.

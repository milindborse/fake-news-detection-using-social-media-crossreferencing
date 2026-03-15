# Fake News Detection – News Credibility Verification

A **local-first Python Streamlit application** that verifies news claims by cross-referencing **Reddit**, **Wikipedia**, **Hacker News**, **web/news sources**, **professional fact-check databases**, and **semantic similarity analysis**. It computes an explainable credibility score (0–100) and presents evidence in an interactive UI.

---

## Features

- 🔍 **Real-time claim verification** — enter any news claim and get immediate results
- 📊 **Explainable credibility score** (0–100) with signal breakdown
- 🏷️ **Verdict labels** — *Likely True*, *Unclear*, or *Likely False*
- 🌐 **Multi-source cross-referencing**:
  - **Reddit** — community discussion and corroboration (PRAW, read-only)
  - **Wikipedia** — entity and background validation
  - **Web / News** — GDELT real-time news events + RSS feeds from Reuters, BBC, AP, NPR, Guardian
  - **Hacker News** — tech-community discussion (Algolia API, no key required)
  - **Google Fact Check API** — professional fact-check verdicts from Snopes, PolitiFact, etc.
- 🧠 **Semantic similarity matching** — uses sentence-transformers (`all-MiniLM-L6-v2`) to find evidence that is semantically related to the claim, replacing keyword-only matching
- ⚖️ **Granular source reputation scoring** — each source is weighted by domain trustworthiness (e.g. Reuters 0.95 vs. unknown blog 0.3) instead of a binary reliable/unreliable check
- 🔬 **Explainability panel** — per-signal contribution with rationale
- 🛡️ **Graceful degradation** — works even when individual sources are unavailable or rate-limited
- ⚙️ **Configurable** — weights, thresholds, and source toggles via `.env`

---

## Architecture & Pipeline

```
User Claim
    ↓
Claim Normalization (keyword extraction, entity detection)
    ↓
Collect Sources (Reddit, Wikipedia, Web/News, Hacker News)
    ↓
Semantic Evidence Retrieval (embedding similarity > 0.65)
    ↓
Fact Check API Verification (Google Fact Check Tools)
    ↓
Source Credibility Scoring (granular domain reputation)
    ↓
Agreement Analysis (corroboration + contradiction signals)
    ↓
Final Credibility Score (0–100 with explanation)
```

### Output Example

```json
{
  "claim": "COVID vaccines cause infertility",
  "credibility_score": 18,
  "fact_check_rating": "False",
  "supporting_sources": [],
  "contradicting_sources": ["snopes.com", "reuters.com"],
  "explanation": "Claim contradicted by snopes.com, reuters.com and Snopes fact-check"
}
```

---

## Project Structure

```
fake-news-detection-using-social-media-crossreferencing/
├── app.py                             # Streamlit UI entry point
├── requirements.txt
├── .env.example                       # Template for environment variables
├── README.md
├── LICENSE
├── tests/
│   └── test_new_features.py           # Tests for new modules
└── src/
    ├── __init__.py
    ├── config.py                      # Settings, weights, thresholds
    ├── collectors/
    │   ├── __init__.py
    │   ├── reddit_collector.py        # PRAW-based Reddit search
    │   ├── hackernews_collector.py    # Hacker News Algolia API
    │   ├── wikipedia_collector.py     # Wikipedia API / entity lookup
    │   └── web_collector.py           # GDELT + RSS feeds
    ├── nlp/
    │   ├── __init__.py
    │   ├── claim_normalizer.py        # Keyword extraction, entity detection
    │   └── semantic_matcher.py        # ✨ Embedding similarity (Feature 3)
    ├── scoring/
    │   ├── __init__.py
    │   ├── credibility.py             # Weighted signal scoring engine
    │   └── source_reputation.py       # ✨ Domain reputation DB (Feature 2)
    └── services/
        ├── __init__.py
        └── factcheck_service.py       # ✨ Google Fact Check API (Feature 1)
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

**Key dependencies:**

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `praw` | Reddit API |
| `wikipedia` | Wikipedia API |
| `requests` | HTTP client (web collector, Fact Check API) |
| `sentence-transformers` | Semantic similarity embeddings |
| `numpy` | Numerical operations |
| `pandas` | Data display |
| `python-dotenv` | Environment variable loading |

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

```env
# Reddit API (free read-only access)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=news-credibility-checker/0.1 by your_reddit_username

# Google Fact Check API (free)
GOOGLE_FACTCHECK_API_KEY=your_google_api_key

# Semantic similarity
ENABLE_SEMANTIC=true
SEMANTIC_MODEL_NAME=all-MiniLM-L6-v2
SEMANTIC_SIMILARITY_THRESHOLD=0.65
```

> **Getting free Reddit API credentials:**
> 1. Go to https://www.reddit.com/prefs/apps
> 2. Click **"create another app"**
> 3. Select **"script"** type
> 4. Fill in name + description, set redirect URI to `http://localhost:8080`
> 5. Copy the `client_id` (under app name) and `client_secret`

> **Getting a Google Fact Check API key:**
> 1. Go to [Google Cloud Console](https://console.cloud.google.com/)
> 2. Enable the **Fact Check Tools API**
> 3. Create an API key under **Credentials**
> 4. Add the key to your `.env` file

> **Wikipedia, Hacker News, and Web sources do not require any API keys** — they work out of the box.

---

## Running the App

```bash
streamlit run app.py
```

The app opens in your browser at **http://localhost:8501**.

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Sample Workflow

1. Enter a news claim, e.g.:
   > *"Scientists discover new species of giant dinosaur in Patagonia"*
2. (Optional) Paste the source article URL for reference.
3. Click **🚀 Verify Claim**.
4. The app will:
   - Extract keywords and entities from your claim
   - Search Reddit, Wikipedia, Hacker News, and web/news sources
   - Compute semantic similarity between claim and evidence
   - Query Google Fact Check API for professional verdicts
   - Apply domain reputation weighting to each source
   - Compute a weighted credibility score
   - Display evidence cards grouped by source (including fact-checks and semantic matches)
   - Show an explainability panel with per-signal contributions
5. Review the verdict: **Likely True** / **Unclear** / **Likely False**

---

## Scoring Signals

| Signal | Default Weight | Description |
|---|---|---|
| Independent Corroboration | 35% | Distinct sources/domains that mention the claim |
| Contradiction / Debunking | 30% | Evidence containing hoax/fake/debunked keywords |
| Source Quality | 20% | Granular domain reputation scoring (0–1 per source) |
| Recency Relevance | 15% | Recency of evidence (recent = more relevant) |
| Fact-Check Verification | 40%* | Professional fact-check verdicts (when available) |

\* When fact-check results are found, all signal weights are normalised so the total still maps to 0–100.

All weights are configurable via `.env`.

### Source Reputation Tiers

| Tier | Example Domains | Score |
|---|---|---|
| Trusted global news | bbc.com, reuters.com, apnews.com | 0.85–0.95 |
| Fact-checkers | snopes.com, factcheck.org, politifact.com | 0.90–0.95 |
| Scientific / official | who.int, nature.com, nasa.gov | 0.85–0.95 |
| Reference | wikipedia.org, britannica.com | 0.80–0.85 |
| Social platforms | reddit.com, news.ycombinator.com | 0.40–0.55 |
| Unknown / blogs | (default) | 0.30 |

---

## New Features (v2)

### Feature 1 — Google Fact Check API Integration

The system now queries Google's Fact Check Tools API to retrieve professional fact-check verdicts. When verdicts are found, they **heavily influence** the credibility score via a dedicated "Fact-Check Verification" signal.

**Module:** `src/services/factcheck_service.py`

```python
from src.services.factcheck_service import search_fact_checks

results = search_fact_checks("COVID vaccines cause infertility")
# Returns: [{"claim": "...", "rating": "False", "publisher": "Snopes", "url": "...", "confidence": 0.1}]
```

### Feature 2 — Granular Source Reputation Scoring

Instead of a binary reliable/unreliable check, each source is now weighted with a granular 0–1 reputation score based on its domain.

**Module:** `src/scoring/source_reputation.py`

```python
from src.scoring.source_reputation import get_domain_score

get_domain_score("https://www.bbc.com/news/article")  # → 0.95
get_domain_score("reddit.com")                         # → 0.50
get_domain_score("unknownblog.example.com")            # → 0.30
```

### Feature 3 — Semantic Similarity Search

Evidence matching now uses embedding-based semantic similarity (sentence-transformers, `all-MiniLM-L6-v2`) instead of keyword matching. Only evidence with cosine similarity > 0.65 is considered relevant.

**Module:** `src/nlp/semantic_matcher.py`

```python
from src.nlp.semantic_matcher import find_semantic_matches

matches = find_semantic_matches("claim text", evidence_articles)
# Returns: [{"source": "...", "url": "...", "similarity": 0.82, "text": "..."}]
```

---

## Limitations & Ethical Disclaimer

- **Not a fact-checker:** This tool provides automated credibility *signals*, not verified facts. It should supplement, not replace, expert fact-checking.
- **API dependencies:** Reddit analysis requires valid API credentials. Google Fact Check API requires a free API key. Web search depends on GDELT availability and RSS feed accessibility.
- **No real-time social media:** Twitter/X is not included due to API cost restrictions; Reddit is used as a proxy for social discussion.
- **Bias in sources:** The reliable-domain whitelist and quality subreddits reflect English-language, Western-centric sources. Non-English claims may score lower due to limited coverage.
- **Model download:** The semantic similarity model (`all-MiniLM-L6-v2`, ~80 MB) is downloaded on first use.
- **No persistence:** Analysis results are not stored. Each run is stateless.
- **Rate limits:** Heavy usage may trigger rate limits on Reddit, GDELT, or Google APIs. The app handles this gracefully by degrading to available sources.
- **Intended use:** This tool is for educational and research purposes. Do not use it as the sole basis for editorial, legal, or policy decisions.

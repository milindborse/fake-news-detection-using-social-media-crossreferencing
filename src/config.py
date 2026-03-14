"""
Application configuration: timeouts, weights, thresholds, and source toggles.
Values can be overridden via environment variables (loaded via python-dotenv).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Reddit (PRAW) credentials
# ---------------------------------------------------------------------------
REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "fake-news-detector/0.1")

# ---------------------------------------------------------------------------
# Source toggles  (set to "false" to disable a collector)
# ---------------------------------------------------------------------------
ENABLE_REDDIT: bool = os.getenv("ENABLE_REDDIT", "true").lower() == "true"
ENABLE_WIKIPEDIA: bool = os.getenv("ENABLE_WIKIPEDIA", "true").lower() == "true"
ENABLE_WEB: bool = os.getenv("ENABLE_WEB", "true").lower() == "true"

# ---------------------------------------------------------------------------
# HTTP / network settings
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "10"))   # seconds
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))

# ---------------------------------------------------------------------------
# Collection limits
# ---------------------------------------------------------------------------
REDDIT_MAX_RESULTS: int = int(os.getenv("REDDIT_MAX_RESULTS", "20"))
WIKIPEDIA_MAX_RESULTS: int = int(os.getenv("WIKIPEDIA_MAX_RESULTS", "5"))
WEB_MAX_RESULTS: int = int(os.getenv("WEB_MAX_RESULTS", "10"))

# ---------------------------------------------------------------------------
# Scoring weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_CORROBORATION: float = float(os.getenv("WEIGHT_CORROBORATION", "0.35"))
WEIGHT_CONTRADICTION: float = float(os.getenv("WEIGHT_CONTRADICTION", "0.30"))
WEIGHT_SOURCE_QUALITY: float = float(os.getenv("WEIGHT_SOURCE_QUALITY", "0.20"))
WEIGHT_RECENCY: float = float(os.getenv("WEIGHT_RECENCY", "0.15"))

# ---------------------------------------------------------------------------
# Credibility thresholds
# ---------------------------------------------------------------------------
LABEL_TRUE_THRESHOLD: int = int(os.getenv("LABEL_TRUE_THRESHOLD", "60"))
LABEL_FALSE_THRESHOLD: int = int(os.getenv("LABEL_FALSE_THRESHOLD", "40"))

# ---------------------------------------------------------------------------
# Reliable / high-quality domains whitelist
# ---------------------------------------------------------------------------
RELIABLE_DOMAINS: list[str] = [
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "nytimes.com",
    "theguardian.com",
    "washingtonpost.com",
    "npr.org",
    "pbs.org",
    "economist.com",
    "scientificamerican.com",
    "nature.com",
    "science.org",
    "cdc.gov",
    "who.int",
    "nih.gov",
    "gov.uk",
    "snopes.com",
    "factcheck.org",
    "politifact.com",
]

# ---------------------------------------------------------------------------
# High-quality subreddits (Reddit quality baseline)
# ---------------------------------------------------------------------------
QUALITY_SUBREDDITS: list[str] = [
    "worldnews",
    "news",
    "science",
    "politics",
    "technology",
    "askscience",
    "explainlikeimfive",
    "todayilearned",
    "geopolitics",
    "environment",
]

# Minimum Reddit post score to be considered a quality signal
REDDIT_MIN_SCORE: int = int(os.getenv("REDDIT_MIN_SCORE", "5"))

# ---------------------------------------------------------------------------
# Contradiction / debunking keywords
# ---------------------------------------------------------------------------
CONTRADICTION_KEYWORDS: list[str] = [
    "fake",
    "false",
    "hoax",
    "debunked",
    "misinformation",
    "disinformation",
    "misleading",
    "fabricated",
    "untrue",
    "rumor",
    "rumour",
    "satire",
    "parody",
    "not true",
    "fact check",
    "fact-check",
    "disputed",
    "correction",
    "retraction",
]

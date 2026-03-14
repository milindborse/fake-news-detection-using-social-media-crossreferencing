"""
Reddit collector: fetches posts related to a claim using PRAW (read-only).

Falls back gracefully when credentials are missing or API is unavailable.
"""

import time
from datetime import datetime, timezone
from typing import Any

from src import config
from src.nlp.claim_normalizer import build_search_query

# ---------------------------------------------------------------------------
# Type alias for a normalised Reddit record
# ---------------------------------------------------------------------------
RedditRecord = dict[str, Any]


def _make_error_meta(message: str) -> dict[str, Any]:
    return {"error": True, "message": message, "source": "reddit"}


def collect(
    claim: str,
    max_results: int | None = None,
) -> tuple[list[RedditRecord], dict[str, Any]]:
    """Search Reddit for posts related to *claim*.

    Args:
        claim: The news claim text to search for.
        max_results: Override the default max results from config.

    Returns:
        A tuple of ``(records, meta)`` where *records* is a (possibly empty)
        list of normalised dicts and *meta* contains status/error info.
    """
    if not config.ENABLE_REDDIT:
        return [], {"source": "reddit", "skipped": True, "reason": "disabled in config"}

    limit = max_results if max_results is not None else config.REDDIT_MAX_RESULTS
    query = build_search_query(claim, max_keywords=6)

    meta: dict[str, Any] = {"source": "reddit", "query": query}

    # Check credentials
    if not config.REDDIT_CLIENT_ID or not config.REDDIT_CLIENT_SECRET:
        meta.update(
            {
                "error": True,
                "message": (
                    "Reddit credentials not configured. "
                    "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env"
                ),
            }
        )
        return [], meta

    try:
        import praw  # type: ignore
        from praw.exceptions import PRAWException  # type: ignore
    except ImportError:
        meta.update(
            {
                "error": True,
                "message": "praw package not installed. Run: pip install praw",
            }
        )
        return [], meta

    records: list[RedditRecord] = []

    for attempt in range(config.MAX_RETRIES + 1):
        try:
            reddit = praw.Reddit(
                client_id=config.REDDIT_CLIENT_ID,
                client_secret=config.REDDIT_CLIENT_SECRET,
                user_agent=config.REDDIT_USER_AGENT,
                read_only=True,
            )

            submissions = reddit.subreddit("all").search(
                query,
                sort="relevance",
                time_filter="year",
                limit=limit,
            )

            for submission in submissions:
                records.append(
                    {
                        "source": "reddit",
                        "title": submission.title,
                        "text": (submission.selftext or "")[:500],
                        "author": str(submission.author) if submission.author else "[deleted]",
                        "subreddit": submission.subreddit.display_name,
                        "score": submission.score,
                        "upvote_ratio": getattr(submission, "upvote_ratio", None),
                        "num_comments": submission.num_comments,
                        "created_utc": datetime.fromtimestamp(
                            submission.created_utc, tz=timezone.utc
                        ).isoformat(),
                        "url": f"https://reddit.com{submission.permalink}",
                        "is_quality_subreddit": submission.subreddit.display_name.lower()
                        in [s.lower() for s in config.QUALITY_SUBREDDITS],
                    }
                )

            meta["count"] = len(records)
            meta["error"] = False
            return records, meta

        except Exception as exc:  # noqa: BLE001
            if attempt < config.MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            meta.update({"error": True, "message": str(exc)})
            return [], meta

    return records, meta

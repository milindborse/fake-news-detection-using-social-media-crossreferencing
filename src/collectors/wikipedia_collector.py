"""
Wikipedia collector: fetches article summaries for entities found in a claim.

Uses the ``wikipedia`` package (wraps MediaWiki API) with graceful fallback.
"""

import time
from typing import Any

from src import config
from src.nlp.claim_normalizer import build_search_query, extract_entities

WikiRecord = dict[str, Any]


def _make_error_meta(message: str) -> dict[str, Any]:
    return {"error": True, "message": message, "source": "wikipedia"}


def collect(
    claim: str,
    max_results: int | None = None,
) -> tuple[list[WikiRecord], dict[str, Any]]:
    """Search Wikipedia for pages relevant to *claim*.

    Extracts entities from the claim, searches Wikipedia for each, and
    returns normalised summary records.

    Args:
        claim: The news claim text.
        max_results: Override default max results from config.

    Returns:
        Tuple of ``(records, meta)``.
    """
    if not config.ENABLE_WIKIPEDIA:
        return [], {"source": "wikipedia", "skipped": True, "reason": "disabled in config"}

    limit = max_results if max_results is not None else config.WIKIPEDIA_MAX_RESULTS
    meta: dict[str, Any] = {"source": "wikipedia"}

    try:
        import wikipedia  # type: ignore
        wikipedia.set_lang("en")
    except ImportError:
        meta.update(
            {
                "error": True,
                "message": "wikipedia package not installed. Run: pip install wikipedia",
            }
        )
        return [], meta

    # Build search queries: primary query + individual entity terms
    primary_query = build_search_query(claim, max_keywords=5)
    entities = extract_entities(claim)

    search_terms: list[str] = [primary_query]
    for entity_type in ("persons", "organizations", "locations", "misc"):
        for ent in entities.get(entity_type, [])[:2]:
            if ent not in search_terms:
                search_terms.append(ent)

    seen_titles: set[str] = set()
    records: list[WikiRecord] = []

    for term in search_terms:
        if len(records) >= limit:
            break
        for attempt in range(config.MAX_RETRIES + 1):
            try:
                search_results = wikipedia.search(term, results=3)
                for page_title in search_results:
                    if len(records) >= limit:
                        break
                    if page_title in seen_titles:
                        continue
                    seen_titles.add(page_title)
                    try:
                        page = wikipedia.page(page_title, auto_suggest=False)
                        summary = page.summary[:600] if page.summary else ""
                        records.append(
                            {
                                "source": "wikipedia",
                                "title": page.title,
                                "summary": summary,
                                "url": page.url,
                                "categories": page.categories[:5] if hasattr(page, "categories") else [],
                                "search_term": term,
                            }
                        )
                    except wikipedia.exceptions.DisambiguationError as de:
                        # Pick first non-ambiguous option
                        for option in de.options[:2]:
                            if option in seen_titles:
                                continue
                            seen_titles.add(option)
                            try:
                                page = wikipedia.page(option, auto_suggest=False)
                                summary = page.summary[:600] if page.summary else ""
                                records.append(
                                    {
                                        "source": "wikipedia",
                                        "title": page.title,
                                        "summary": summary,
                                        "url": page.url,
                                        "categories": page.categories[:5] if hasattr(page, "categories") else [],
                                        "search_term": term,
                                    }
                                )
                                break
                            except Exception:  # noqa: BLE001
                                continue
                    except wikipedia.exceptions.PageError:
                        continue
                break  # success, don't retry
            except Exception as exc:  # noqa: BLE001
                if attempt < config.MAX_RETRIES:
                    time.sleep(1)
                    continue
                # Log error but continue with other terms
                meta.setdefault("warnings", []).append(f"Failed term '{term}': {exc}")
                break

    meta["count"] = len(records)
    meta["error"] = False
    return records, meta

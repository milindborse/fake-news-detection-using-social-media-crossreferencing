"""
Tests for the three new features:
  1. Google Fact Check API service
  2. Source reputation scoring
  3. Semantic similarity matching (with mocked model)
"""

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Test Feature 2: Source Reputation Scoring
# ---------------------------------------------------------------------------

from src.scoring.source_reputation import get_domain_score, get_source_type


class TestSourceReputation(unittest.TestCase):
    """Tests for src/scoring/source_reputation.py."""

    def test_known_trusted_news(self):
        self.assertAlmostEqual(get_domain_score("bbc.com"), 0.95)
        self.assertAlmostEqual(get_domain_score("reuters.com"), 0.95)
        self.assertAlmostEqual(get_domain_score("apnews.com"), 0.90)

    def test_known_scientific(self):
        self.assertAlmostEqual(get_domain_score("who.int"), 0.95)
        self.assertAlmostEqual(get_domain_score("nature.com"), 0.95)

    def test_known_social(self):
        self.assertAlmostEqual(get_domain_score("reddit.com"), 0.50)
        self.assertAlmostEqual(get_domain_score("news.ycombinator.com"), 0.55)

    def test_wikipedia(self):
        self.assertAlmostEqual(get_domain_score("wikipedia.org"), 0.80)

    def test_unknown_domain(self):
        self.assertAlmostEqual(get_domain_score("randomblog123.com"), 0.30)

    def test_full_url(self):
        self.assertAlmostEqual(
            get_domain_score("https://www.bbc.com/news/article-123"), 0.95
        )

    def test_subdomain_matching(self):
        self.assertAlmostEqual(
            get_domain_score("https://world.bbc.com/news"), 0.95
        )

    def test_www_prefix_stripped(self):
        self.assertAlmostEqual(get_domain_score("www.reuters.com"), 0.95)

    def test_get_source_type(self):
        self.assertEqual(get_source_type("bbc.com"), "trusted_news")
        self.assertEqual(get_source_type("snopes.com"), "factchecker")
        self.assertEqual(get_source_type("who.int"), "scientific_official")
        self.assertEqual(get_source_type("wikipedia.org"), "reference")
        self.assertEqual(get_source_type("reddit.com"), "social_platform")
        self.assertEqual(get_source_type("randomblog.com"), "unknown")


# ---------------------------------------------------------------------------
# Test Feature 1: Fact Check Service
# ---------------------------------------------------------------------------

from src.services.factcheck_service import (
    _normalise_rating,
    _parse_response,
    search_fact_checks,
)


class TestFactCheckService(unittest.TestCase):
    """Tests for src/services/factcheck_service.py."""

    def test_normalise_rating_known(self):
        self.assertAlmostEqual(_normalise_rating("True"), 0.95)
        self.assertAlmostEqual(_normalise_rating("False"), 0.10)
        self.assertAlmostEqual(_normalise_rating("Mostly True"), 0.80)
        self.assertAlmostEqual(_normalise_rating("Misleading"), 0.30)

    def test_normalise_rating_unknown(self):
        self.assertAlmostEqual(_normalise_rating("Some random text"), 0.50)

    def test_parse_response_empty(self):
        result = _parse_response({})
        self.assertEqual(result, [])

    def test_parse_response_with_claims(self):
        data = {
            "claims": [
                {
                    "text": "Test claim",
                    "claimReview": [
                        {
                            "textualRating": "False",
                            "publisher": {"name": "Snopes"},
                            "url": "https://snopes.com/check/123",
                        }
                    ],
                }
            ]
        }
        results = _parse_response(data)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["claim"], "Test claim")
        self.assertEqual(results[0]["rating"], "False")
        self.assertEqual(results[0]["publisher"], "Snopes")
        self.assertAlmostEqual(results[0]["confidence"], 0.10)

    @patch("src.services.factcheck_service.config")
    def test_search_fact_checks_no_api_key(self, mock_config):
        mock_config.GOOGLE_FACTCHECK_API_KEY = ""
        mock_config.FACTCHECK_MAX_RESULTS = 5
        mock_config.MAX_RETRIES = 1
        mock_config.REQUEST_TIMEOUT = 5
        results = search_fact_checks("test claim")
        # With no API key and no network, should return empty or fallback results
        self.assertIsInstance(results, list)


# ---------------------------------------------------------------------------
# Test Feature 3: Semantic Matcher (mocked model)
# ---------------------------------------------------------------------------


class TestSemanticMatcher(unittest.TestCase):
    """Tests for src/nlp/semantic_matcher.py (with mocked model)."""

    @patch("src.nlp.semantic_matcher._get_model")
    def test_find_semantic_matches(self, mock_get_model):
        import numpy as np
        from src.nlp.semantic_matcher import find_semantic_matches

        # Create a mock model that returns predictable embeddings
        mock_model = MagicMock()

        # Claim embedding
        claim_emb = np.array([1.0, 0.0, 0.0])
        # Article embeddings: one similar, one dissimilar
        article_embs = np.array([
            [0.9, 0.1, 0.0],   # similar
            [0.0, 0.0, 1.0],   # dissimilar
        ])

        mock_model.encode = MagicMock(side_effect=[claim_emb, article_embs])
        mock_get_model.return_value = mock_model

        articles = [
            {"title": "Similar article", "text": "Related text", "url": "https://a.com", "source": "web"},
            {"title": "Different article", "text": "Unrelated text", "url": "https://b.com", "source": "web"},
        ]

        matches = find_semantic_matches("test claim", articles, threshold=0.5)
        # Only the first article should match (cosine sim ~0.994)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["title"], "Similar article")
        self.assertGreater(matches[0]["similarity"], 0.5)

    @patch("src.nlp.semantic_matcher._get_model")
    def test_find_semantic_matches_empty(self, mock_get_model):
        from src.nlp.semantic_matcher import find_semantic_matches
        matches = find_semantic_matches("test", [])
        self.assertEqual(matches, [])


# ---------------------------------------------------------------------------
# Test: Credibility scoring integration with fact-check results
# ---------------------------------------------------------------------------

from src.scoring.credibility import compute_score


class TestCredibilityScoringIntegration(unittest.TestCase):
    """Test that compute_score works with the new factcheck_results parameter."""

    def test_compute_score_without_factcheck(self):
        """Backward compatibility: calling without factcheck_results still works."""
        result = compute_score(
            claim="Test claim",
            reddit_records=[],
            wiki_records=[],
            web_records=[],
            hn_records=[],
        )
        self.assertIn("score", result)
        self.assertIn("label", result)
        self.assertIn("fact_check_rating", result)
        self.assertIsNone(result["fact_check_rating"])

    def test_compute_score_with_factcheck(self):
        """Score should include fact-check signal when results provided."""
        factcheck_results = [
            {
                "claim": "Test",
                "rating": "False",
                "publisher": "Snopes",
                "url": "https://snopes.com",
                "confidence": 0.1,
            }
        ]
        result = compute_score(
            claim="Test claim",
            reddit_records=[],
            wiki_records=[],
            web_records=[],
            hn_records=[],
            factcheck_results=factcheck_results,
        )
        self.assertIn("score", result)
        self.assertEqual(result["fact_check_rating"], "False")
        # Should have a fact-check signal in breakdown
        signal_names = [s["name"] for s in result["signal_breakdown"]]
        self.assertIn("Fact-Check Verification", signal_names)

    def test_compute_score_result_structure(self):
        """Verify new fields exist in the result dict."""
        result = compute_score(
            claim="Test claim",
            reddit_records=[],
            wiki_records=[],
            web_records=[],
        )
        self.assertIn("supporting_sources", result)
        self.assertIn("contradicting_sources", result)
        self.assertIn("explanation", result)
        self.assertIn("fact_check_rating", result)


# ---------------------------------------------------------------------------
# Test: Fact-check free fallback functions
# ---------------------------------------------------------------------------

from src.services.factcheck_service import (
    _infer_rating_from_text,
    _extract_keywords,
)


class TestFactCheckFreeFallback(unittest.TestCase):
    """Tests for free fact-check fallback functions."""

    def test_infer_rating_false(self):
        self.assertEqual(_infer_rating_from_text("This claim is False"), "False")

    def test_infer_rating_true(self):
        self.assertEqual(_infer_rating_from_text("Claim confirmed True"), "True")

    def test_infer_rating_misleading(self):
        self.assertEqual(
            _infer_rating_from_text("Misleading claim about vaccines"), "Misleading"
        )

    def test_infer_rating_mostly_false(self):
        self.assertEqual(
            _infer_rating_from_text("Mostly False: politicians exaggerated"), "Mostly False"
        )

    def test_infer_rating_mixture(self):
        self.assertEqual(
            _infer_rating_from_text("A mixture of truth and fiction"), "Mixture"
        )

    def test_infer_rating_unknown(self):
        self.assertEqual(
            _infer_rating_from_text("Regular news article"), "Under Review"
        )

    def test_extract_keywords_basic(self):
        keywords = _extract_keywords("The president signed a new bill")
        self.assertIn("president", keywords)
        self.assertIn("signed", keywords)
        self.assertIn("new", keywords)
        self.assertIn("bill", keywords)
        # Stopwords should be excluded
        self.assertNotIn("the", keywords)

    def test_extract_keywords_empty(self):
        keywords = _extract_keywords("")
        self.assertEqual(keywords, [])


# ---------------------------------------------------------------------------
# Test: Web collector DuckDuckGo fallback
# ---------------------------------------------------------------------------

from src.collectors.web_collector import _collect_duckduckgo, _collect_rss


class TestWebCollectorImprovements(unittest.TestCase):
    """Tests for web collector improvements."""

    def test_rss_keyword_matching_relaxed(self):
        """Verify RSS matching works with single keyword overlap."""
        from xml.etree.ElementTree import Element, SubElement, tostring
        from unittest.mock import patch

        # Build a minimal RSS feed with an item matching 1 keyword
        rss = Element("rss")
        channel = SubElement(rss, "channel")
        item = SubElement(channel, "item")
        SubElement(item, "title").text = "New planet discovered by NASA"
        SubElement(item, "link").text = "https://example.com/article"
        SubElement(item, "description").text = "Scientists found a new planet."
        SubElement(item, "pubDate").text = "Mon, 01 Jan 2024 00:00:00 GMT"

        feed_xml = tostring(rss, encoding="unicode")

        with patch("src.collectors.web_collector._fetch_url", return_value=feed_xml.encode("utf-8")):
            records = _collect_rss("planet habitable zone", limit=5)
            # Should match because "planet" (>= 3 chars) appears in the item
            self.assertGreaterEqual(len(records), 1)
            self.assertEqual(records[0]["title"], "New planet discovered by NASA")

    def test_duckduckgo_parses_html(self):
        """Test DuckDuckGo HTML parser with sample HTML."""
        from unittest.mock import patch

        sample_html = '''
        <div class="result">
            <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnews&amp;rut=abc">
                Test News Article
            </a>
            <a class="result__snippet">A snippet about the article.</a>
        </div>
        '''

        with patch("src.collectors.web_collector._fetch_url", return_value=sample_html.encode("utf-8")):
            records = _collect_duckduckgo("test query", limit=5)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["title"], "Test News Article")
            self.assertEqual(records[0]["url"], "https://example.com/news")
            self.assertEqual(records[0]["provider"], "duckduckgo")


# ---------------------------------------------------------------------------
# Test: HN collector fallback behavior
# ---------------------------------------------------------------------------

from src.collectors.hackernews_collector import _search_hn


class TestHNCollectorImprovements(unittest.TestCase):
    """Tests for Hacker News collector improvements."""

    def test_search_hn_returns_records(self):
        """Test _search_hn parses Algolia response correctly."""
        from unittest.mock import patch, MagicMock
        import json

        sample_response = json.dumps({
            "hits": [
                {
                    "title": "Test Story",
                    "url": "https://example.com/story",
                    "created_at": "2024-01-01T00:00:00.000Z",
                    "points": 100,
                    "num_comments": 50,
                    "author": "testuser",
                    "objectID": "12345",
                    "story_text": "",
                }
            ]
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = sample_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        meta: dict[str, Any] = {"source": "hackernews"}
        with patch("src.collectors.hackernews_collector.urlopen", return_value=mock_resp):
            records = _search_hn("test query", 5, meta)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["title"], "Test Story")
            self.assertEqual(records[0]["source"], "hackernews")

    def test_search_hn_empty_response(self):
        """Test _search_hn handles empty response."""
        from unittest.mock import patch, MagicMock
        import json

        sample_response = json.dumps({"hits": []}).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = sample_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        meta: dict[str, Any] = {"source": "hackernews"}
        with patch("src.collectors.hackernews_collector.urlopen", return_value=mock_resp):
            records = _search_hn("obscure query", 5, meta)
            self.assertEqual(len(records), 0)


if __name__ == "__main__":
    unittest.main()

"""
Tests for the three new features:
  1. Google Fact Check API service
  2. Source reputation scoring
  3. Semantic similarity matching (with mocked model)
"""

import unittest
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
        results = search_fact_checks("test claim")
        self.assertEqual(results, [])


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


if __name__ == "__main__":
    unittest.main()

"""
Fake News Detection – News Credibility Verification
Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import sys
import traceback
from datetime import datetime, timezone

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="News Credibility Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Import project modules (show a friendly error if something is missing)
# ---------------------------------------------------------------------------
try:
    from src import config
    from src.collectors import reddit_collector, web_collector, wikipedia_collector, hackernews_collector
    from src.nlp.claim_normalizer import (
        build_search_query,
        extract_keywords,
        get_claim_summary,
    )
    from src.scoring.credibility import compute_score
except ImportError as e:
    st.error(
        f"**Import error:** {e}\n\n"
        "Make sure you have installed all dependencies:\n"
        "```\npip install -r requirements.txt\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Sidebar: 30% width */
    [data-testid="stSidebar"] {
        min-width: 20%;
        max-width: 30%;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 100%;
    }
    .score-badge {
        font-size: 3rem;
        font-weight: bold;
        padding: 0.5rem 1.5rem;
        border-radius: 12px;
        display: inline-block;
    }
    .verdict-badge {
        font-size: 2rem;
        font-weight: bold;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        display: inline-block;
        margin-top: 0.2rem;
    }
    .badge-green  { background: #d4edda; color: #155724; }
    .badge-orange { background: #fff3cd; color: #856404; }
    .badge-red    { background: #f8d7da; color: #721c24; }
    .signal-row { padding: 6px 0; border-bottom: 1px solid #eee; }
    .evidence-card {
        background: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar – configuration hints
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    """Render sidebar with configuration status and tips."""
    with st.sidebar:
        st.title("⚙️ Configuration")

        st.markdown("### Source Status")
        reddit_ok = bool(config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET)
        st.markdown(
            f"{'✅' if (reddit_ok and config.ENABLE_REDDIT) else '⚠️'} **Reddit** "
            f"({'ready' if reddit_ok else 'no credentials – limited'})"
        )
        st.markdown(
            f"{'✅' if config.ENABLE_WIKIPEDIA else '⚠️'} **Wikipedia** "
            f"({'enabled' if config.ENABLE_WIKIPEDIA else 'disabled'})"
        )
        st.markdown(
            f"{'✅' if config.ENABLE_WEB else '⚠️'} **Web / News** "
            f"({'enabled' if config.ENABLE_WEB else 'disabled'})"
        )
        st.markdown(
            f"{'✅' if config.ENABLE_HACKERNEWS else '⚠️'} **Hacker News** "
            f"({'enabled' if config.ENABLE_HACKERNEWS else 'disabled'})"
        )

        st.divider()
        st.markdown("### Scoring Weights")
        st.markdown(
            f"- Corroboration: **{config.WEIGHT_CORROBORATION:.0%}**\n"
            f"- Contradiction: **{config.WEIGHT_CONTRADICTION:.0%}**\n"
            f"- Source Quality: **{config.WEIGHT_SOURCE_QUALITY:.0%}**\n"
            f"- Recency: **{config.WEIGHT_RECENCY:.0%}**"
        )

        st.divider()
        st.caption(
            "Set Reddit credentials in `.env` file to enable Reddit analysis. "
            "See `.env.example` for required variables."
        )


# ---------------------------------------------------------------------------
# Evidence rendering helpers
# ---------------------------------------------------------------------------
def _render_reddit_evidence(records: list[dict]) -> None:
    if not records:
        st.info("No Reddit evidence found.")
        return
    for r in records:
        quality_badge = "⭐" if r.get("is_quality_subreddit") else ""
        score_str = f"↑ {r.get('score', 0)}"
        st.markdown(
            f"""<div class="evidence-card">
            <strong>{quality_badge} {r.get('title', 'No title')}</strong><br/>
            <small>r/{r.get('subreddit', '?')} &nbsp;|&nbsp; {score_str} &nbsp;|&nbsp;
            {r.get('author', '?')} &nbsp;|&nbsp; {r.get('created_utc', '')[:10]}</small><br/>
            <small>{r.get('text', '')[:200]}</small><br/>
            <a href="{r.get('url', '#')}" target="_blank">View on Reddit ↗</a>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_wikipedia_evidence(records: list[dict]) -> None:
    if not records:
        st.info("No Wikipedia evidence found.")
        return
    for r in records:
        st.markdown(
            f"""<div class="evidence-card">
            <strong>📖 {r.get('title', 'No title')}</strong><br/>
            <small>{r.get('summary', '')[:300]}</small><br/>
            <a href="{r.get('url', '#')}" target="_blank">Read on Wikipedia ↗</a>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_web_evidence(records: list[dict]) -> None:
    if not records:
        st.info("No web evidence found.")
        return
    for r in records:
        domain = r.get("domain", "")
        reliable = "✅" if any(
            domain.endswith(d) for d in config.RELIABLE_DOMAINS
        ) else ""
        st.markdown(
            f"""<div class="evidence-card">
            <strong>{reliable} {r.get('title', 'No title')}</strong><br/>
            <small>{domain} &nbsp;|&nbsp; {r.get('published_date', '')[:16]}</small><br/>
            <small>{r.get('snippet', '')[:250]}</small><br/>
            <a href="{r.get('url', '#')}" target="_blank">Read article ↗</a>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_hackernews_evidence(records: list[dict]) -> None:
    if not records:
        st.info("No Hacker News evidence found.")
        return
    for r in records:
        quality_badge = "⭐" if r.get("is_quality") else ""
        points_str = f"▲ {r.get('points', 0)}"
        st.markdown(
            f"""<div class="evidence-card">
            <strong>{quality_badge} {r.get('title', 'No title')}</strong><br/>
            <small>{points_str} points &nbsp;|&nbsp;
            {r.get('num_comments', 0)} comments &nbsp;|&nbsp;
            {r.get('author', '?')} &nbsp;|&nbsp; {r.get('created_utc', '')[:10]}</small><br/>
            <small>{r.get('text', '')[:200]}</small><br/>
            <a href="{r.get('url', '#')}" target="_blank">View on HN ↗</a>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_signal_breakdown(signals: list[dict]) -> None:
    """Render the signal breakdown table."""
    if not signals:
        return
    import pandas as pd  # imported here to avoid top-level dependency issues

    rows = [
        {
            "Signal": s["name"],
            "Value (0–1)": round(s["value"], 3),
            "Weight": f"{s['weight']:.0%}",
            "Contribution (pts)": round(s["contribution"], 1),
            "Rationale": s["rationale"],
        }
        for s in signals
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------
def run_analysis(claim: str) -> None:
    """Run the full credibility analysis pipeline and display results."""
    st.divider()
    st.subheader("🔄 Analysis in progress…")

    # Progress tracking
    progress = st.progress(0)
    status_text = st.empty()

    reddit_records: list[dict] = []
    wiki_records: list[dict] = []
    web_records: list[dict] = []
    hn_records: list[dict] = []
    collector_metas: list[dict] = []

    # ------------------------------------------------------------------
    # 1. Collect Reddit evidence
    # ------------------------------------------------------------------
    status_text.text("Searching Reddit…")
    progress.progress(10)
    try:
        reddit_records, reddit_meta = reddit_collector.collect(claim)
        collector_metas.append(reddit_meta)
    except Exception as exc:  # noqa: BLE001
        collector_metas.append({"source": "reddit", "error": True, "message": str(exc)})

    # ------------------------------------------------------------------
    # 2. Collect Wikipedia evidence
    # ------------------------------------------------------------------
    status_text.text("Querying Wikipedia…")
    progress.progress(25)
    try:
        wiki_records, wiki_meta = wikipedia_collector.collect(claim)
        collector_metas.append(wiki_meta)
    except Exception as exc:  # noqa: BLE001
        collector_metas.append({"source": "wikipedia", "error": True, "message": str(exc)})

    # ------------------------------------------------------------------
    # 3. Collect web evidence
    # ------------------------------------------------------------------
    status_text.text("Searching web/news sources…")
    progress.progress(45)
    try:
        web_records, web_meta = web_collector.collect(claim)
        collector_metas.append(web_meta)
    except Exception as exc:  # noqa: BLE001
        collector_metas.append({"source": "web", "error": True, "message": str(exc)})

    # ------------------------------------------------------------------
    # 4. Collect Hacker News evidence
    # ------------------------------------------------------------------
    status_text.text("Searching Hacker News…")
    progress.progress(65)
    try:
        hn_records, hn_meta = hackernews_collector.collect(claim)
        collector_metas.append(hn_meta)
    except Exception as exc:  # noqa: BLE001
        collector_metas.append({"source": "hackernews", "error": True, "message": str(exc)})

    # ------------------------------------------------------------------
    # 5. Compute score
    # ------------------------------------------------------------------
    status_text.text("Computing credibility score…")
    progress.progress(85)
    try:
        result = compute_score(claim, reddit_records, wiki_records, web_records, hn_records)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Scoring engine error: {exc}")
        st.text(traceback.format_exc())
        return

    progress.progress(100)
    status_text.empty()
    progress.empty()

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("📊 Credibility Report")

    # Timestamp & source counts
    analyzed_at = result.get("analyzed_at", datetime.now(timezone.utc).isoformat())
    sc = result.get("source_counts", {})
    st.caption(
        f"Analyzed at {analyzed_at[:19].replace('T', ' ')} UTC  |  "
        f"Reddit: {sc.get('reddit', 0)} result(s)  |  "
        f"Wikipedia: {sc.get('wikipedia', 0)} result(s)  |  "
        f"Web: {sc.get('web', 0)} result(s)  |  "
        f"Hacker News: {sc.get('hackernews', 0)} result(s)"
    )

    # Score + label
    score = result["score"]
    label = result["label"]
    color = result.get("label_color", "orange")
    badge_class = {"green": "badge-green", "orange": "badge-orange", "red": "badge-red"}.get(
        color, "badge-orange"
    )

    col_score, col_label = st.columns(2)
    with col_score:
        st.markdown(
            f'<div class="score-badge {badge_class}">{score}/100</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Credibility Score**")
    with col_label:
        st.markdown(
            f'<div class="verdict-badge {badge_class}">{label}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Verdict**")

    st.markdown("")
    st.markdown("**Summary**")
    st.write(result.get("summary", "No summary available."))

    # ------------------------------------------------------------------
    # Explainability panel
    # ------------------------------------------------------------------
    st.divider()
    with st.expander("🔬 Signal Breakdown & Explainability", expanded=True):
        pos = result.get("positive_signals", [])
        neg = result.get("negative_signals", [])

        if pos:
            st.markdown("#### ✅ Positive Signals")
            for s in pos:
                st.markdown(
                    f"**{s['name']}** — +{s['contribution']} pts  \n"
                    f"_{s['rationale']}_"
                )

        if neg:
            st.markdown("#### ⚠️ Weaker Signals")
            for s in neg:
                st.markdown(
                    f"**{s['name']}** — {s['contribution']} pts  \n"
                    f"_{s['rationale']}_"
                )

        st.markdown("#### 📈 Full Signal Table")
        _render_signal_breakdown(result.get("signal_breakdown", []))

    # ------------------------------------------------------------------
    # Evidence panels
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("🗂️ Evidence")

    tabs = st.tabs(
        [
            f"🌐 Web ({sc.get('web', 0)})",
            f"📰 Reddit ({sc.get('reddit', 0)})",
            f"🔶 Hacker News ({sc.get('hackernews', 0)})",
            f"📖 Wikipedia ({sc.get('wikipedia', 0)})",
        ]
    )
    with tabs[0]:
        _render_web_evidence(web_records)
    with tabs[1]:
        _render_reddit_evidence(reddit_records)
    with tabs[2]:
        _render_hackernews_evidence(hn_records)
    with tabs[3]:
        _render_wikipedia_evidence(wiki_records)

    # ------------------------------------------------------------------
    # Collector status / warnings
    # ------------------------------------------------------------------
    errors = [m for m in collector_metas if m.get("error")]
    skipped = [m for m in collector_metas if m.get("skipped")]
    if errors or skipped:
        with st.expander("⚠️ Collection Warnings", expanded=False):
            for m in errors:
                st.warning(
                    f"**{m.get('source', '?').title()}**: {m.get('message', 'Unknown error')}"
                )
            for m in skipped:
                st.info(
                    f"**{m.get('source', '?').title()}** was skipped: {m.get('reason', '')}"
                )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
def main() -> None:
    render_sidebar()

    st.title("🔍 Fake News Detector")
    st.markdown(
        "Verify a news claim by cross-referencing **Reddit**, **Hacker News**, **Wikipedia**, "
        "and **web/news sources**. Get an explainable credibility score."
    )

    st.divider()

    # Input area
    with st.form("claim_form"):
        claim_text = st.text_area(
            "📝 Enter the news claim or article text",
            height=120,
            placeholder=(
                "e.g. 'Scientists discover new species of dinosaur in Argentina' "
                "or paste an article paragraph…"
            ),
            help="Enter the specific claim you want to verify.",
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            article_url = st.text_input(
                "🔗 Optional: Article URL (for reference only)",
                placeholder="https://example.com/news/article",
            )
        with col2:
            submitted = st.form_submit_button("🚀 Verify Claim", use_container_width=True)

    if submitted:
        claim_text = claim_text.strip()
        if not claim_text:
            st.warning("Please enter a claim or article text to verify.")
        elif len(claim_text) < 10:
            st.warning("Claim text is too short. Please provide more context.")
        else:
            # Show parsed keywords for transparency
            keywords = extract_keywords(claim_text, max_keywords=8)
            query = build_search_query(claim_text, max_keywords=6)
            st.info(
                f"🔑 **Extracted keywords:** {', '.join(keywords)}  \n"
                f"🔎 **Search query:** `{query}`"
            )
            run_analysis(claim_text)

    else:
        # Show sample / onboarding when no claim is submitted
        st.markdown("### 💡 How it works")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                "**1. Enter a claim**\n\n"
                "Type or paste a news headline or factual claim in the box above."
            )
        with col_b:
            st.markdown(
                "**2. We search the web**\n\n"
                "The app queries Reddit, Hacker News, Wikipedia, and news sources in real time."
            )
        with col_c:
            st.markdown(
                "**3. Get your score**\n\n"
                "An explainable 0–100 credibility score with evidence cards."
            )

        st.divider()
        st.markdown(
            "> ⚠️ **Disclaimer:** This tool provides automated credibility signals "
            "and is **not** a replacement for expert fact-checking. "
            "Always verify important claims through multiple trusted sources."
        )


if __name__ == "__main__":
    main()

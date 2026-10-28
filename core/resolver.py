from __future__ import annotations

import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{1,}")
STOPWORDS = {
    "the", "and", "for", "you", "your", "are", "our", "with", "from", "this", "that",
    "how", "what", "when", "where", "can", "could", "please", "help", "need", "want",
    "into", "about", "have", "has", "was", "were", "its", "it's", "not", "does", "do",
    "should", "would", "will", "their", "them", "they", "then", "before", "after", "all", "to", "our", "its", "it",
}


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "") if token.lower() not in STOPWORDS}


def retrieve_article(
    request_text: str,
    knowledge_base: list[dict[str, Any]],
    min_score: float = 18.0,
) -> tuple[dict[str, Any] | None, float, dict[str, Any]]:
    """Hybrid local retrieval using TF-IDF/cosine similarity and keyword overlap.

    The returned score is a heuristic relevance score, not a calibrated probability.
    Only approved articles are eligible.
    """
    approved = [article for article in knowledge_base if article.get("approved") is True]
    if not approved or not (request_text or "").strip():
        return None, 0.0, {"method": "hybrid_tfidf_keyword", "reason": "No eligible article or empty request", "matches": []}

    query_tokens = _tokens(request_text)
    corpus = []
    for article in approved:
        corpus.append(" ".join([
            str(article.get("title", "")), str(article.get("intent", "")),
            " ".join(article.get("keywords", [])),
        ]))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(corpus + [request_text])
        similarities = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except ValueError:
        similarities = [0.0] * len(approved)

    ranked: list[tuple[float, dict[str, Any], float, float, list[str]]] = []
    for index, article in enumerate(approved):
        article_text = corpus[index]
        article_tokens = _tokens(article_text)
        overlap_terms = sorted(query_tokens.intersection(article_tokens))
        keyword_score = (len(overlap_terms) / len(query_tokens) * 100.0) if query_tokens else 0.0
        semantic_similarity = float(similarities[index]) if len(similarities) > index else 0.0
        semantic_score = min(100.0, semantic_similarity / 0.22 * 100.0)
        combined_score = round(0.60 * keyword_score + 0.40 * semantic_score, 2)
        ranked.append((combined_score, article, keyword_score, semantic_similarity, overlap_terms))

    ranked.sort(key=lambda item: item[0], reverse=True)
    top_score, top_article, keyword_score, cosine_score, overlap_terms = ranked[0]
    diagnostics = {
        "method": "hybrid_tfidf_keyword",
        "keyword_score": round(keyword_score, 2),
        "cosine_similarity": round(float(cosine_score), 4),
        "matched_terms": overlap_terms,
        "matches": [
            {"article_id": item[1].get("article_id"), "title": item[1].get("title"), "score": item[0]}
            for item in ranked[:3]
        ],
    }
    if top_score < min_score or (not overlap_terms and cosine_score < 0.08):
        return None, round(top_score, 2), diagnostics
    return top_article, round(top_score, 2), diagnostics

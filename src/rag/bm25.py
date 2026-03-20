# src/rag/bm25.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: BM25 (Best Match 25) is the gold standard keyword
# search algorithm. It's what Elasticsearch and most search
# engines use under the hood.
#
# WHY BM25 and not just "does the word appear?"
# BM25 solves two problems with naive keyword matching:
#
#   Problem 1 — Term Frequency Saturation:
#     A doc with "RAG" appearing 100 times shouldn't score
#     100x better than one with "RAG" appearing 1 time.
#     BM25 saturates TF — diminishing returns after a point.
#
#   Problem 2 — Document Length Normalization:
#     A 10,000-word doc has "RAG" more often just by being long.
#     BM25 normalizes by document length so short focused chunks
#     aren't penalized against long verbose ones.
#
# BM25 FORMULA (simplified):
#   score(query, doc) = Σ IDF(term) × TF(term, doc) × saturation_factor
#
#   IDF = Inverse Document Frequency
#       = log((N - df + 0.5) / (df + 0.5))
#       = rare terms score higher than common terms
#       = "RAG" in 2/100 docs scores higher than "the" in 100/100 docs
#
#   TF = Term Frequency (how many times term appears in this doc)
#
#   saturation_factor = (k1 + 1) / (TF + k1 × (1 - b + b × dl/avgdl))
#       k1 = 1.5  controls TF saturation
#       b  = 0.75 controls length normalization
#
# We implement this from scratch so you understand every line.
# In production you'd use `rank_bm25` library, but building it
# yourself is worth doing once.
# ─────────────────────────────────────────────────────────────

import math
import re
from collections import Counter
from ..chunking.chunk import Chunk


def _tokenize(text: str) -> list[str]:
    """
    Convert text to a list of lowercase tokens (words).
    We strip punctuation and lowercase everything so "RAG", "rag",
    and "RAG." all match the same token.
    """
    text = text.lower()
    # Keep only alphanumeric characters and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class BM25Index:
    """
    A BM25 keyword search index built over a list of Chunks.

    Usage:
        index = BM25Index(chunks)
        scores = index.score("what is RAG")
        # returns list of (chunk, score) sorted by relevance
    """

    # BM25 hyperparameters — standard values, rarely need tuning
    K1 = 1.5    # TF saturation: higher = more reward for repeated terms
    B  = 0.75   # length normalization: 1.0 = full normalization, 0 = none

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.n_docs = len(chunks)

        # Tokenize all chunks
        self.tokenized_chunks = [_tokenize(c.text) for c in chunks]

        # Term frequency per document: [{token: count}, ...]
        self.tf = [Counter(tokens) for tokens in self.tokenized_chunks]

        # Document lengths (in tokens)
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_chunks]

        # Average document length — used for normalization
        self.avg_dl = (sum(self.doc_lengths) / self.n_docs) if self.n_docs else 1

        # Document frequency: how many docs contain each term
        # df["rag"] = 5 means 5 chunks contain the word "rag"
        self.df: dict[str, int] = {}
        for token_counts in self.tf:
            for term in token_counts:
                self.df[term] = self.df.get(term, 0) + 1

        # Precompute IDF for all known terms
        self.idf: dict[str, float] = {}
        for term, df in self.df.items():
            # Standard BM25 IDF formula
            # +0.5 is a smoothing factor to avoid log(0)
            self.idf[term] = math.log(
                (self.n_docs - df + 0.5) / (df + 0.5) + 1
            )

    def score(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Score all chunks against a query using BM25.

        Returns:
            List of (Chunk, score) tuples sorted by score descending.
            Score of 0.0 means no query terms appeared in that chunk.
        """
        query_terms = _tokenize(query)

        if not query_terms:
            return [(chunk, 0.0) for chunk in self.chunks]

        scores = []
        for i, chunk in enumerate(self.chunks):
            doc_len = self.doc_lengths[i]
            chunk_tf = self.tf[i]
            total_score = 0.0

            for term in query_terms:
                if term not in self.idf:
                    continue  # term never appears in any chunk → skip

                tf_val = chunk_tf.get(term, 0)
                if tf_val == 0:
                    continue  # term not in this chunk → no contribution

                idf_val = self.idf[term]

                # BM25 TF component with saturation and length normalization
                numerator   = tf_val * (self.K1 + 1)
                denominator = tf_val + self.K1 * (
                    1 - self.B + self.B * (doc_len / self.avg_dl)
                )
                total_score += idf_val * (numerator / denominator)

            scores.append((chunk, total_score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scores = scores[:top_k]

        return scores

    def get_top_chunks(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.1,   # filter out zero/near-zero matches
    ) -> list[Chunk]:
        """
        Convenience method: returns just the Chunk objects (no scores).
        Filters out chunks with score below min_score.
        """
        scored = self.score(query, top_k=top_k * 2)  # fetch extra, then filter
        filtered = [chunk for chunk, score in scored if score >= min_score]
        return filtered[:top_k]
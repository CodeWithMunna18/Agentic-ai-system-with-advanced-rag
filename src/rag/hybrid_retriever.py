# src/rag/hybrid_retriever.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: Hybrid Retrieval = Dense Search + Sparse (BM25) Search
#
# Neither method is strictly better:
#   Dense  → great at semantic meaning, struggles with rare keywords
#   BM25   → great at exact terms, blind to paraphrasing/synonyms
#
# FUSION STRATEGY: Reciprocal Rank Fusion (RRF)
#
# RRF doesn't care about raw scores (BM25 scores are 0–10,
# dense scores are 0–1 — incompatible scales). Instead it uses
# only RANK POSITION, which is always comparable:
#
#   RRF_score(doc) = Σ 1 / (k + rank_in_list)
#                   where k=60 is a smoothing constant
#
# Example:
#   Dense results:  [A=1st, B=2nd, C=5th]
#   BM25 results:   [C=1st, A=2nd, D=3rd]
#
#   RRF score A = 1/(60+1) + 1/(60+2) = 0.0164 + 0.0160 = 0.0324
#   RRF score C = 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0318
#   RRF score B = 1/(60+2) + 0        = 0.0160
#   RRF score D = 0        + 1/(60+3) = 0.0159
#
#   Final order: A > C > B > D
#   A wins because it appeared high in BOTH lists.
# ─────────────────────────────────────────────────────────────

from .retriever import Retriever, RetrievedChunk
from .bm25 import BM25Index
from ..embedding.embedder import embed_query
from ..embedding.vector_store import VectorStore
from ..chunking.chunk import Chunk


class HybridRetriever:
    """
    Combines dense vector search (ChromaDB) with BM25 keyword search,
    fused using Reciprocal Rank Fusion (RRF).
    """

    RRF_K = 60  # standard constant — higher = smoother rank blending

    def __init__(
        self,
        vector_store: VectorStore,
        chunks: list[Chunk],           # ALL chunks — needed to build BM25 index
        top_k: int = 5,
        relevance_threshold: float = 0.0,  # RRF scores are small; don't filter hard
        dense_weight: float = 0.6,     # how much to trust dense results (vs BM25)
        sparse_weight: float = 0.4,    # how much to trust BM25 results
    ):
        """
        Args:
            vector_store:       ChromaDB store (for dense search)
            chunks:             all Chunk objects (to build BM25 index in memory)
            top_k:              number of results to return
            dense_weight:       0.0–1.0, weight for semantic search
            sparse_weight:      0.0–1.0, weight for BM25 keyword search
        """
        # Dense retriever (Phase 4's retriever, reused)
        self.dense_retriever = Retriever(
            vector_store=vector_store,
            top_k=top_k * 2,          # fetch extra candidates for fusion
            relevance_threshold=0.0,  # no threshold — RRF handles ranking
        )

        # BM25 index built in memory from all chunks
        print(f"🔨 Building BM25 index over {len(chunks)} chunks...")
        self.bm25 = BM25Index(chunks)
        print(f"   BM25 index ready. Vocabulary size: {len(self.bm25.df)} terms")

        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.chunks = chunks

    def retrieve(
        self,
        question: str,
        source_file: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Hybrid retrieval pipeline:
        1. Dense search  → ranked list of chunks by semantic similarity
        2. BM25 search   → ranked list of chunks by keyword relevance
        3. RRF fusion    → merge both rankings into final ranked list
        4. Return top_k  → as RetrievedChunk objects with hybrid scores
        """

        # ── Step 1: Dense search ────────────────────────────────
        dense_results = self.dense_retriever.retrieve(question, source_file=source_file)
        # Map chunk_id → rank position (0-indexed)
        dense_ranks: dict[str, int] = {}
        for rank, result in enumerate(dense_results):
            chunk_id = f"{result.source_file}::chunk_{result.chunk_index}"
            dense_ranks[chunk_id] = rank

        # ── Step 2: BM25 keyword search ─────────────────────────
        bm25_scored = self.bm25.score(question, top_k=self.top_k * 2)
        # Filter by source_file if requested
        if source_file:
            bm25_scored = [
                (c, s) for c, s in bm25_scored
                if c.source_file == source_file
            ]
        # Map chunk_id → rank position
        bm25_ranks: dict[str, int] = {}
        for rank, (chunk, score) in enumerate(bm25_scored):
            chunk_id = chunk.chunk_id
            bm25_ranks[chunk_id] = rank

        # ── Step 3: Build a unified pool of candidate chunks ────
        # Collect all unique chunks from both result sets
        candidate_map: dict[str, Chunk] = {}
        for chunk, _ in bm25_scored:
            candidate_map[chunk.chunk_id] = chunk
        # Also add dense results — use source_file+index as key
        dense_text_map: dict[str, RetrievedChunk] = {}
        for r in dense_results:
            cid = f"{r.source_file}::chunk_{r.chunk_index}"
            dense_text_map[cid] = r

        # ── Step 4: RRF scoring ──────────────────────────────────
        all_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())
        rrf_scores: dict[str, float] = {}

        for chunk_id in all_ids:
            dense_rank = dense_ranks.get(chunk_id, len(self.chunks))
            bm25_rank  = bm25_ranks.get(chunk_id, len(self.chunks))

            # Weighted RRF: each source contributes proportionally to its weight
            rrf = (
                self.dense_weight  * (1 / (self.RRF_K + dense_rank)) +
                self.sparse_weight * (1 / (self.RRF_K + bm25_rank))
            )
            rrf_scores[chunk_id] = rrf

        # ── Step 5: Sort by RRF score and build results ─────────
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        results: list[RetrievedChunk] = []
        for chunk_id in sorted_ids[:self.top_k]:
            # Prefer the RetrievedChunk from dense (has score), fallback to BM25 chunk
            if chunk_id in dense_text_map:
                dr = dense_text_map[chunk_id]
                results.append(RetrievedChunk(
                    text=dr.text,
                    score=round(rrf_scores[chunk_id], 6),  # use RRF score
                    source_file=dr.source_file,
                    chunk_index=dr.chunk_index,
                    start_char=dr.start_char,
                    end_char=dr.end_char,
                ))
            elif chunk_id in candidate_map:
                c = candidate_map[chunk_id]
                results.append(RetrievedChunk(
                    text=c.text,
                    score=round(rrf_scores[chunk_id], 6),
                    source_file=c.source_file,
                    chunk_index=c.chunk_index,
                    start_char=c.start_char,
                    end_char=c.end_char,
                ))

        return results
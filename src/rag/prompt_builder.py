# src/rag/prompt_builder.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: Prompt engineering is the most impactful thing you
# can do to improve RAG answer quality — more than model choice,
# more than chunk size.
#
# A good RAG prompt has 4 parts:
#
#   [1] ROLE       — tell the model what it is and how to behave
#   [2] CONTEXT    — the retrieved chunks (the grounding material)
#   [3] RULES      — constraints (only use context, cite sources, etc.)
#   [4] QUESTION   — the user's actual question
#
# WHY explicit rules?
# Without rules, Gemini may mix training knowledge with retrieved
# context — producing plausible but ungrounded answers.
# "Only answer from the context below" forces grounding.
#
# WHY numbered sources?
# Numbering chunks lets the model cite [Source 1], [Source 2] etc.
# in its answer. In Phase 5 we'll parse these citations to show
# users exactly which document + position the answer came from.
# ─────────────────────────────────────────────────────────────

from .retriever import RetrievedChunk


# ── Main prompt builder ────────────────────────────────────────
def build_rag_prompt(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    max_context_tokens: int = 3000,   # safety limit for context window
) -> str:
    """
    Build the complete prompt to send to Gemini.

    Args:
        question:           the user's question
        retrieved_chunks:   ranked results from the Retriever
        max_context_tokens: hard cap on context size (~4 chars per token)

    Returns:
        A complete prompt string ready to send to Gemini
    """
    if not retrieved_chunks:
        return _build_no_context_prompt(question)

    # Build the context block from retrieved chunks
    context_block, sources_used = _build_context_block(
        retrieved_chunks, max_context_tokens
    )

    prompt = f"""You are a precise document intelligence assistant.
Your job is to answer questions based ONLY on the provided context.

RULES:
- Answer using ONLY the information in the context below
- If the context doesn't contain enough information, say: "I don't have enough information in the provided documents to answer this."
- Cite your sources using [Source N] notation after each claim
- Be concise and direct
- Do not add information from outside the context

CONTEXT ({sources_used} source(s) retrieved):
{context_block}

QUESTION: {question}

ANSWER:"""

    return prompt


def _build_context_block(
    chunks: list[RetrievedChunk],
    max_context_tokens: int,
) -> tuple[str, int]:
    """
    Format retrieved chunks into a numbered context block.
    Respects the max_context_tokens limit by truncating if needed.

    Returns:
        (formatted context string, number of sources included)
    """
    max_chars = max_context_tokens * 4
    sections = []
    total_chars = 0
    sources_used = 0

    for i, chunk in enumerate(chunks):
        header = f"[Source {i+1}] From: {chunk.source_file} (relevance: {chunk.score:.2f})"
        section = f"{header}\n{chunk.text}"
        section_chars = len(section)

        # Stop adding chunks if we'd exceed the context limit
        if total_chars + section_chars > max_chars and sections:
            break

        sections.append(section)
        total_chars += section_chars
        sources_used += 1

    context_block = "\n\n---\n\n".join(sections)
    return context_block, sources_used


def _build_no_context_prompt(question: str) -> str:
    """
    Prompt to use when no relevant chunks were retrieved.
    Tells the model to be honest rather than hallucinate.
    """
    return f"""You are a document intelligence assistant.

No relevant context was found in the document store for this question.

QUESTION: {question}

Respond with: "I couldn't find relevant information in the provided documents to answer this question. Please try rephrasing or uploading more relevant documents."
"""


# ── Prompt inspector (learning tool) ──────────────────────────
def preview_prompt(prompt: str, max_chars: int = 800) -> None:
    """
    Print a formatted preview of the prompt.
    Useful for understanding exactly what Gemini receives.
    """
    print("\n" + "─" * 60)
    print("  PROMPT PREVIEW (what Gemini receives)")
    print("─" * 60)
    if len(prompt) <= max_chars:
        print(prompt)
    else:
        half = max_chars // 2
        print(prompt[:half])
        print(f"\n  ... [{len(prompt) - max_chars} chars hidden] ...\n")
        print(prompt[-half:])
    print("─" * 60)
    print(f"  Total prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
    print("─" * 60)
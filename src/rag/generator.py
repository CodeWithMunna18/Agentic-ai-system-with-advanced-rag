# src/rag/generator.py
# ─────────────────────────────────────────────────────────────
# CONCEPT: The Generator is the "G" in RAG.
# It takes the prompt (question + retrieved context) and calls
# Gemini to produce a grounded answer.
#
# WHY a separate generator module?
# Later you might want to:
# - Stream responses token by token (better UX)
# - Switch from Gemini Flash to Gemini Pro for harder questions
# - Add retry logic for rate limits
# - Log all generations for evaluation
# Keeping generation separate makes all of this easy to change.
#
# MODEL CHOICE: gemini-2.0-flash
# - Fast and cheap for Q&A tasks
# - Large context window (handles long prompts)
# - Great instruction following (respects our RAG rules)
# ─────────────────────────────────────────────────────────────

import os
from dataclasses import dataclass, field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGResponse:
    """
    The complete output of one RAG query.
    Bundles the answer with everything needed to explain/debug it.
    """
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)   # retrieved chunks metadata
    prompt_tokens: int = 0
    answer_tokens: int = 0
    model: str = "gemini-2.0-flash"

    def display(self) -> None:
        """Pretty-print the full response with sources."""
        print("\n" + "═" * 60)
        print(f"  ❓ QUESTION")
        print(f"  {self.question}")
        print("─" * 60)
        print(f"  💡 ANSWER")
        # Indent each line of the answer
        for line in self.answer.strip().split("\n"):
            print(f"  {line}")
        print("─" * 60)
        print(f"  📚 SOURCES USED ({len(self.sources)})")
        for i, src in enumerate(self.sources):
            print(f"  [{i+1}] {src.get('source_file', 'unknown')} "
                  f"(score: {src.get('score', 0):.4f})")
        print("═" * 60)


class Generator:
    """
    Wraps the Gemini LLM call for RAG generation.
    Takes a prompt string, returns a RAGResponse.
    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        """
        Args:
            model: Gemini model to use for generation
                   "gemini-2.0-flash"   — fast, cheap, great default
                   "gemini-1.5-pro"     — slower, better reasoning
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        question: str,
        retrieved_chunks: list,       # list[RetrievedChunk] from retriever
        temperature: float = 0.1,     # low = more focused/factual answers
    ) -> RAGResponse:
        """
        Call Gemini with the RAG prompt and return a structured response.

        WHY temperature=0.1?
        Temperature controls randomness. For RAG Q&A we want:
        - Low randomness → answer stays close to retrieved context
        - High temperature → creative but may stray from the facts
        0.1 is the sweet spot for factual grounded answers.

        Args:
            prompt:            the complete RAG prompt from prompt_builder
            question:          original user question (for the response object)
            retrieved_chunks:  chunks used (for source attribution)
            temperature:       0.0 = deterministic, 1.0 = creative

        Returns:
            RAGResponse with answer + source metadata
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=1024,
                ),
            )

            answer = response.text or "No response generated."

            # Build source metadata for citations
            sources = [
                {
                    "source_file":  chunk.source_file,
                    "score":        chunk.score,
                    "chunk_index":  chunk.chunk_index,
                    "preview":      chunk.text[:100] + "...",
                }
                for chunk in retrieved_chunks
            ]

            return RAGResponse(
                question=question,
                answer=answer,
                sources=sources,
                model=self.model,
            )

        except Exception as e:
            return RAGResponse(
                question=question,
                answer=f"Generation failed: {e}",
                model=self.model,
            )
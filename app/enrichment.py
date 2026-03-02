"""LLM-based metadata enrichment for document chunks.

Generates keywords, summaries, and synthetic questions per chunk
to improve retrieval quality.  Uses batch processing with rate-limit
handling following the same pattern as ``embed_chunks()`` in ingest.py.

Enrichment data is stored in the existing ``metadata_extra`` JSONB column
(zero schema changes).  Keywords and synthetic questions are also appended
to chunk content so the auto-generated ``ts_content`` TSVECTOR indexes them
for keyword search.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


# ── Structured output model ──────────────────────────────────────────


class ChunkEnrichment(BaseModel):
    """LLM-generated metadata for a single chunk."""

    keywords: list[str] = Field(
        description="5-10 keywords or keyphrases that capture the key concepts",
        min_length=3,
        max_length=10,
    )
    summary: str = Field(
        description="1-2 sentence summary of the chunk content",
        max_length=300,
    )
    synthetic_questions: list[str] = Field(
        description="2-3 questions this chunk could answer",
        min_length=1,
        max_length=3,
    )


# ── Prompt ───────────────────────────────────────────────────────────


_ENRICHMENT_SYSTEM = """\
You are a metadata extraction engine for a compliance document retrieval system.
Given a document chunk, extract:
1. keywords: 5-10 keywords or keyphrases that capture the key concepts.
   Include proper nouns, technical terms, and concepts a user might search for.
2. summary: A 1-2 sentence summary of what this chunk contains.
3. synthetic_questions: 2-3 questions that this chunk could answer.
   Phrase them as a user searching for information would ask them."""

_ENRICHMENT_HUMAN = """\
<document_chunk>
{chunk_text}
</document_chunk>
<source>{source}</source>

Extract metadata for this chunk."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _ENRICHMENT_SYSTEM),
    ("human", _ENRICHMENT_HUMAN),
])


# ── Core enrichment ─────────────────────────────────────────────────


def _get_enrichment_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=512,
    )


def enrich_chunk(chunk_text: str, source: str = "") -> ChunkEnrichment:
    """Enrich a single chunk with LLM-generated metadata."""
    chain = _prompt | _get_enrichment_llm().with_structured_output(ChunkEnrichment)
    return chain.invoke({"chunk_text": chunk_text, "source": source})


def enrich_chunks_batch(
    rows: list[dict[str, Any]],
    batch_size: int = 10,
) -> list[dict[str, Any]]:
    """Enrich chunks in batches, adding enrichment to metadata_extra and content.

    Operates on the row dicts produced by ``embed_chunks()`` (each has
    ``content``, ``source``, ``metadata_extra``, etc.).  Modifies in-place
    and returns the list.

    Rate-limit retry follows the same exponential-backoff pattern as
    ``embed_chunks()``.
    """
    total = len(rows)
    enriched_count = 0
    total_batches = (total + batch_size - 1) // batch_size

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        batch_num = i // batch_size + 1

        for row in batch:
            content = row.get("content", "")
            source = row.get("source", "")

            # Skip very short chunks (not enough content to enrich)
            if len(content.strip()) < 20:
                logger.debug("Skipping short chunk (%d chars)", len(content))
                continue

            for attempt in range(3):
                try:
                    enrichment = enrich_chunk(content, source=source)

                    # Store in metadata_extra
                    extra = row.get("metadata_extra") or {}
                    extra["keywords"] = enrichment.keywords
                    extra["summary"] = enrichment.summary
                    extra["synthetic_questions"] = enrichment.synthetic_questions
                    row["metadata_extra"] = extra

                    # Append to content for tsvector indexing
                    keywords_text = ", ".join(enrichment.keywords)
                    questions_text = " ".join(enrichment.synthetic_questions)
                    row["content"] = (
                        content
                        + f"\n\n[KEYWORDS: {keywords_text}]"
                        + f"\n[QUESTIONS: {questions_text}]"
                    )

                    enriched_count += 1
                    break

                except Exception as exc:
                    if "429" in str(exc) or "rate" in str(exc).lower():
                        wait = 2**attempt * 5
                        logger.warning(
                            "Enrichment rate limited, retrying in %ds...", wait
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            "Enrichment failed for chunk (source=%s): %s",
                            source, str(exc)[:200],
                        )
                        break  # skip this chunk on non-rate-limit errors

        logger.info(
            "Enriched batch %d/%d (%d/%d chunks total)",
            batch_num, total_batches, enriched_count, total,
        )

        # Rate limit spacing between batches
        if i + batch_size < total:
            time.sleep(1)

    logger.info("Enrichment complete: %d/%d chunks enriched", enriched_count, total)
    return rows

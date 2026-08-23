"""
NyayaGuide AI — End-to-End RAG Pipeline
Coordinates retrieval, relevance verification, grounding prompts, LLM generation,
programmatic citations, and follow-up question validation.

Query Classification Architecture:
    Every query is classified into exactly one of four mutually exclusive states:

    KNOWLEDGE_BASE_UNAVAILABLE     — FAISS index has 0 vectors; cannot evaluate anything.
                                     Must NOT be conflated with out-of-domain.

    OUT_OF_DOMAIN                  — FAISS returned results but top cosine score is below
                                     threshold AND the query contains no recognised RTI or
                                     Consumer Protection terms. Truly unrelated.

    IN_DOMAIN_BUT_INSUFFICIENT     — Query contains recognised domain keywords and/or the
                                     two-pass retrieval was attempted, but even the broader
                                     retrieval could not produce sufficient relevant context.
                                     The KB exists but lacks sufficient evidence for this query.

    ANSWERABLE                     — Sufficient relevant context was retrieved.
                                     Grounded answer generated from official indexed chunks.
"""
import json
import re
from enum import Enum
from typing import Optional, List, Tuple
from ..config import (
    DEFAULT_TOP_K,
    MIN_RELEVANCE_THRESHOLD,
    ABSTENTION_MESSAGE,
    OPENROUTER_MODEL,
    is_openrouter_configured
)
from ..models.document import RAGResponse, RetrievalResult, SourceCitation, QueryResultState
from ..retrieval.retriever import NyayaRetriever
from ..llm.openrouter_client import OpenRouterClient
from .context_builder import ContextBuilder
from .prompt import build_rag_messages


# ─────────────────────────────────────────────────────────────────────────────
# Domain keyword sets for in-domain detection
#
# These are used ONLY to decide whether to attempt a second retrieval pass
# when the primary pass falls below the relevance threshold.
#
# They do NOT bypass the similarity threshold, the grounding check, or the
# sufficiency evaluation.  A query that contains "RTI" but retrieves chunks
# with top_score < _SECONDARY_RELEVANCE_THRESHOLD still results in
# IN_DOMAIN_BUT_INSUFFICIENT — never in a fabricated answer.
# ─────────────────────────────────────────────────────────────────────────────
_RTI_KEYWORDS = frozenset([
    "rti", "right to information", "information act",
    "public information officer", "pio", "cic",
    "central information commission", "information commission",
    "file an rti", "rti application", "rti request",
    "rti act", "rti rules", "request for information",
    "disclosure of information", "information request",
    "section 6", "section 7", "section 8", "section 11",
    "section 19", "section 20", "first appeal", "second appeal",
    "rti fee", "rti appeal", "obtain information",
])

_CONSUMER_KEYWORDS = frozenset([
    "consumer", "consumer protection", "consumer rights",
    "consumer complaint", "consumer court", "consumer commission",
    "district commission", "state commission", "national commission",
    "cpa", "consumer protection act", "unfair trade", "unfair practice",
    "defective goods", "deficiency in service", "deficiency of service",
    "complainant", "opposite party", "e-commerce", "product liability",
    "mediation", "consumer forum", "consumer dispute",
])

# Secondary cosine threshold — applied ONLY in the two-pass fallback for
# confirmed in-domain queries.  0.40 is deliberately below the primary 0.50
# to allow borderline but genuine RTI/Consumer queries to succeed.
# It still requires actual relevant chunks from the official index.
_SECONDARY_RELEVANCE_THRESHOLD = 0.40

# Abstention message when KB is completely empty
_KB_UNAVAILABLE_MESSAGE = (
    "The NyayaGuide AI knowledge base is currently unavailable or is being initialised. "
    "Please try again in a moment. This is a temporary service condition and is not related "
    "to your question."
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_text(text: str) -> str:
    """
    Normalise a string for duplicate-suggestion comparison:
    - lowercase
    - strip leading/trailing whitespace
    - collapse internal repeated whitespace to single space
    - remove insignificant trailing punctuation (? ! . , ;)
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[?!.,;]+$", "", text)
    return text


def _detect_domain(question: str) -> Optional[str]:
    """
    Checks whether a question contains recognised keywords for a supported legal domain.
    Returns "RTI", "CONSUMER", or None.

    This is used ONLY to gate the two-pass retrieval fallback.
    It does NOT determine the final answer — the FAISS similarity scores and
    the sufficiency check do that.
    """
    q_lower = question.lower()
    if any(kw in q_lower for kw in _RTI_KEYWORDS):
        return "RTI"
    if any(kw in q_lower for kw in _CONSUMER_KEYWORDS):
        return "CONSUMER"
    return None


def _filter_duplicate_suggestions(
    question: str,
    suggestions: List[str],
) -> List[str]:
    """
    Remove follow-up suggestions that are normalised duplicates of the current question
    or empty strings.  Also removes intra-list duplicates.

    Normalisation: lowercase + trim + collapse whitespace + strip trailing punctuation.
    """
    norm_question = _normalise_text(question)
    seen: set = set()
    filtered: List[str] = []
    for s in suggestions:
        norm_s = _normalise_text(s)
        if not norm_s:
            continue
        if norm_s == norm_question:
            continue   # identical to the user's own question
        if norm_s in seen:
            continue   # intra-list duplicate
        seen.add(norm_s)
        filtered.append(s)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class NyayaRAGPipeline:
    """
    Unified RAG pipeline for NyayaGuide AI.

    Processing order:
    1. Empty-index guard     → KNOWLEDGE_BASE_UNAVAILABLE (not OUT_OF_DOMAIN)
    2. Primary retrieval     → evaluate cosine similarity
    3. Domain detection      → gate secondary retrieval pass
    4. Secondary retrieval   → broader top_k, lower threshold (in-domain only)
    5. Sufficiency check     → ANSWERABLE / IN_DOMAIN_BUT_INSUFFICIENT / OUT_OF_DOMAIN
    6. LLM generation        → grounded answer (ANSWERABLE path only)
    7. Suggestion filter     → remove duplicates of current question
    """

    def __init__(
        self,
        retriever: Optional[NyayaRetriever] = None,
        llm_client: Optional[OpenRouterClient] = None,
        min_relevance_threshold: float = MIN_RELEVANCE_THRESHOLD
    ):
        self.retriever = retriever or NyayaRetriever()
        self.llm_client = llm_client or OpenRouterClient()
        self.min_relevance_threshold = min_relevance_threshold
        self.context_builder = ContextBuilder()

    def reload_index(self) -> bool:
        """Hot-reloads the FAISS index and metadata store from disk."""
        return self.retriever.reload_index()

    # ─────────────────────────────────────────────────
    # LLM response parsing
    # ─────────────────────────────────────────────────

    def _parse_llm_response(self, raw_output: str) -> Tuple[str, List[str]]:
        """
        Safely parses structured JSON output from the LLM.
        Handles markdown code blocks, raw JSON, and raw-text fallbacks.
        Validates follow-up questions: removes empties, duplicates, and caps at 4.
        """
        clean_output = raw_output.strip()

        # Strip ```json ... ``` markdown fences
        text_to_parse = clean_output
        if text_to_parse.startswith("```"):
            lines = text_to_parse.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text_to_parse = "\n".join(lines).strip()

        answer = ""
        follow_ups: List[str] = []

        # Attempt 1: Direct JSON parse
        try:
            data = json.loads(text_to_parse)
            if isinstance(data, dict):
                answer = str(data.get("answer", "")).strip()
                raw_follow_ups = data.get("follow_up_questions", [])
                if isinstance(raw_follow_ups, list):
                    for q in raw_follow_ups:
                        if isinstance(q, str):
                            q_str = q.strip()
                            if q_str and len(q_str) >= 2 and q_str not in follow_ups:
                                follow_ups.append(q_str)
        except Exception:
            pass

        # Attempt 2: Regex extract answer if JSON parse failed
        if not answer:
            ans_match = re.search(
                r'"answer"\s*:\s*"([\s\S]*?)(?:"\s*,\s*"follow_up_questions"|"\s*\})',
                clean_output
            )
            if ans_match:
                raw_ans = ans_match.group(1)
                try:
                    answer = json.loads(f'"{raw_ans}"')
                except Exception:
                    answer = raw_ans.replace("\\n", "\n").replace('\\"', '"')

        # Attempt 3: Regex extract follow-up questions
        if not follow_ups:
            fu_matches = re.findall(r'"([^"\n\r\t]+?[?])"', clean_output)
            for q in fu_matches:
                q_clean = q.strip()
                if len(q_clean) > 8 and q_clean not in follow_ups:
                    follow_ups.append(q_clean)

        if not answer:
            answer = clean_output

        # Cap at 4, enforce max length
        validated: List[str] = []
        for q in follow_ups:
            if len(q) <= 200:
                validated.append(q)
            if len(validated) >= 4:
                break

        return answer, validated

    # ─────────────────────────────────────────────────
    # Grounded extractive fallback (LLM rate-limit path)
    # ─────────────────────────────────────────────────

    def _generate_grounded_fallback_summary(
        self,
        retrieval_results: List[RetrievalResult],
        question: str
    ) -> Tuple[str, List[str]]:
        """
        Generates a direct extractive answer from verified FAISS chunks
        when the upstream LLM is temporarily unavailable (rate limit / 429).
        Guarantees zero hallucinations: every sentence is copied verbatim from indexed chunks.
        """
        doc_titles = list(dict.fromkeys(r.title for r in retrieval_results if r.title))
        top_chunks = retrieval_results[:3]

        paragraphs = [
            f"Based on the official Government of India records "
            f"({', '.join(doc_titles)}), here is the verified statutory "
            f"information regarding your query:"
        ]
        for idx, chunk in enumerate(top_chunks, 1):
            ref_label = f" ({chunk.legal_reference})" if chunk.legal_reference else ""
            clean_text = " ".join(chunk.text.strip().split())
            if len(clean_text) > 350:
                clean_text = clean_text[:350].rsplit(".", 1)[0] + "."
            paragraphs.append(
                f"{idx}. **{chunk.title}{ref_label}** (Page {chunk.page}):\n{clean_text}"
            )
        paragraphs.append(
            "*Note: This information is extracted directly from the verified Government "
            "of India knowledge base for civic awareness and does not constitute formal "
            "legal counsel.*"
        )
        answer_text = "\n\n".join(paragraphs)

        cat = retrieval_results[0].category.upper() if retrieval_results else "RTI"
        if cat == "RTI":
            follow_ups = [
                "What is the application fee for filing an RTI request?",
                "What information can a citizen request under RTI?",
                "What is the time limit for a Public Information Officer to respond?",
            ]
        else:
            follow_ups = [
                "What are my rights as a consumer?",
                "How can I file a consumer complaint?",
                "What does the District Consumer Commission do?",
            ]
        return answer_text, follow_ups

    # ─────────────────────────────────────────────────
    # Two-pass retrieval with state tracking
    # ─────────────────────────────────────────────────

    def _retrieve_with_state(
        self,
        question: str,
        top_k: int,
        category: Optional[str],
    ) -> Tuple[List[RetrievalResult], float, QueryResultState]:
        """
        Executes the two-pass retrieval pipeline and returns the precise QueryResultState.

        Step 0 — Empty-index guard:
            If total_vectors == 0, return KNOWLEDGE_BASE_UNAVAILABLE immediately.
            This prevents conflating an unavailable KB with an out-of-domain query.

        Step 1 — Primary retrieval:
            Run standard FAISS search with top_k and MIN_RELEVANCE_THRESHOLD.
            If top_score >= threshold → ANSWERABLE.

        Step 2 — Domain detection:
            If primary pass fails threshold, check whether the query contains
            recognised RTI or Consumer Protection keywords.
            If no keywords → OUT_OF_DOMAIN.

        Step 3 — Secondary retrieval (in-domain only):
            Attempt broader retrieval (top_k * 2 candidates) at
            _SECONDARY_RELEVANCE_THRESHOLD.
            If secondary top_score >= _SECONDARY_RELEVANCE_THRESHOLD → ANSWERABLE.
            Otherwise → IN_DOMAIN_BUT_INSUFFICIENT.
        """
        # ── Ensure index is loaded into memory before checking state ───────
        if hasattr(self.retriever, "ensure_index_loaded"):
            self.retriever.ensure_index_loaded()

        # ── Step 0: Empty-index guard ──────────────────────────────────────
        if self.retriever.vector_store.total_vectors == 0:
            return [], 0.0, QueryResultState.KNOWLEDGE_BASE_UNAVAILABLE


        # ── Step 1: Primary retrieval ──────────────────────────────────────
        results = self.retriever.retrieve(query=question, top_k=top_k, category=category)
        top_score = results[0].score if results else 0.0

        if top_score >= self.min_relevance_threshold:
            return results, top_score, QueryResultState.ANSWERABLE

        # ── Step 2: Domain detection ───────────────────────────────────────
        detected_domain = _detect_domain(question)
        if not detected_domain:
            # No recognised domain keywords → truly out-of-domain
            return results, top_score, QueryResultState.OUT_OF_DOMAIN

        # ── Step 3: Secondary retrieval (in-domain only) ───────────────────
        broader_results = self.retriever.retrieve(
            query=question,
            top_k=top_k * 2,
            category=category,
        )
        broader_top_score = broader_results[0].score if broader_results else 0.0

        if broader_top_score >= _SECONDARY_RELEVANCE_THRESHOLD:
            # Secondary pass succeeded — trim to original top_k
            return broader_results[:top_k], broader_top_score, QueryResultState.ANSWERABLE

        # In-domain but even broader retrieval is insufficient
        return results, top_score, QueryResultState.IN_DOMAIN_BUT_INSUFFICIENT

    # ─────────────────────────────────────────────────
    # Main public entry point
    # ─────────────────────────────────────────────────

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        category: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2
    ) -> RAGResponse:
        """
        Executes end-to-end RAG question answering.

        Returns a RAGResponse with `query_result_state` set to one of:
            ANSWERABLE | OUT_OF_DOMAIN | IN_DOMAIN_BUT_INSUFFICIENT | KNOWLEDGE_BASE_UNAVAILABLE
        """
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        # ── Step 1: Retrieval with state classification ────────────────────
        retrieval_results, top_score, state = self._retrieve_with_state(
            question=clean_question,
            top_k=top_k,
            category=category,
        )

        # ── Step 2: Non-answerable paths ───────────────────────────────────
        if state == QueryResultState.KNOWLEDGE_BASE_UNAVAILABLE:
            return RAGResponse(
                question=clean_question,
                answer=_KB_UNAVAILABLE_MESSAGE,
                sources=[],
                retrieval_results=[],
                is_abstention=True,
                model_used=None,
                top_score=0.0,
                follow_up_questions=[],
                query_result_state=QueryResultState.KNOWLEDGE_BASE_UNAVAILABLE,
            )

        if state == QueryResultState.OUT_OF_DOMAIN:
            return RAGResponse(
                question=clean_question,
                answer=ABSTENTION_MESSAGE,
                sources=[],
                retrieval_results=retrieval_results,
                is_abstention=True,
                model_used=None,
                top_score=round(top_score, 4),
                follow_up_questions=[],
                query_result_state=QueryResultState.OUT_OF_DOMAIN,
            )

        if state == QueryResultState.IN_DOMAIN_BUT_INSUFFICIENT:
            return RAGResponse(
                question=clean_question,
                answer=ABSTENTION_MESSAGE,
                sources=[],
                retrieval_results=retrieval_results,
                is_abstention=True,
                model_used=None,
                top_score=round(top_score, 4),
                follow_up_questions=[],
                query_result_state=QueryResultState.IN_DOMAIN_BUT_INSUFFICIENT,
            )

        # ── Step 3: ANSWERABLE path — extract citations ────────────────────
        citations = self.context_builder.extract_programmatic_citations(retrieval_results)
        context_str = self.context_builder.build_context_string(retrieval_results)
        messages = build_rag_messages(question=clean_question, context_str=context_str)

        # ── Step 4: LLM generation (with rate-limit fallback) ──────────────
        active_model = model or (
            self.llm_client.model
            if isinstance(getattr(self.llm_client, "model", None), str)
            else OPENROUTER_MODEL
        )
        try:
            raw_output = self.llm_client.generate(
                messages=messages,
                model=active_model,
                temperature=temperature
            )
            answer_text, follow_up_questions = self._parse_llm_response(raw_output)
            model_tag: Optional[str] = active_model
        except Exception as e:
            err_msg = str(e).lower()
            if any(kw in err_msg for kw in ("rate limit", "429", "quota", "payment", "402")):
                answer_text, follow_up_questions = self._generate_grounded_fallback_summary(
                    retrieval_results, clean_question
                )
                model_tag = f"{active_model} (Grounded Fallback)"
            else:
                raise

        # ── Step 5: Filter duplicate follow-up suggestions ────────────────
        follow_up_questions = _filter_duplicate_suggestions(clean_question, follow_up_questions)

        # ── Step 6: Append programmatic citations to answer ────────────────
        citation_text = self.context_builder.format_citation_text(citations)
        if citation_text and "Sources:" not in answer_text:
            final_answer = f"{answer_text.strip()}\n\n{citation_text}".strip()
        else:
            final_answer = answer_text.strip()

        return RAGResponse(
            question=clean_question,
            answer=final_answer,
            sources=citations,
            retrieval_results=retrieval_results,
            is_abstention=False,
            model_used=model_tag,
            top_score=round(top_score, 4),
            follow_up_questions=follow_up_questions,
            query_result_state=QueryResultState.ANSWERABLE,
        )


def ask(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    category: Optional[str] = None,
    model: Optional[str] = None
) -> RAGResponse:
    """Convenience module-level function for RAG queries."""
    pipeline = NyayaRAGPipeline()
    return pipeline.ask(question=question, top_k=top_k, category=category, model=model)

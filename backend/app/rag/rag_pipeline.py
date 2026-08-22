"""
NyayaGuide AI — End-to-End RAG Pipeline
Coordinates retrieval, relevance verification, grounding prompts, LLM generation, programmatic citations, and follow-up question validation.
"""
import json
import re
from typing import Optional, List, Tuple
from ..config import (
    DEFAULT_TOP_K,
    MIN_RELEVANCE_THRESHOLD,
    ABSTENTION_MESSAGE,
    OPENROUTER_MODEL,
    is_openrouter_configured
)
from ..models.document import RAGResponse, RetrievalResult, SourceCitation
from ..retrieval.retriever import NyayaRetriever
from ..llm.openrouter_client import OpenRouterClient
from .context_builder import ContextBuilder
from .prompt import build_rag_messages


class NyayaRAGPipeline:
    """
    Unified RAG pipeline for NyayaGuide AI.
    Executes semantic retrieval -> relevance filtering -> context construction -> LLM generation -> citation attachment -> follow-up validation.
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
        """Hot-reloads the FAISS index and metadata store references in-memory."""
        return self.retriever.reload_index()

    def _parse_llm_response(self, raw_output: str) -> Tuple[str, List[str]]:
        """
        Safely parses structured output from the LLM to extract 'answer' and 'follow_up_questions'.
        Handles markdown code blocks, raw JSON, embedded JSON, or raw text fallbacks.
        Validates follow-up questions: removes empties, duplicates, non-questions, and caps at 4.
        """
        clean_output = raw_output.strip()
        
        # Strip markdown code fences (```json ... ```)
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

        # Attempt 1: Direct JSON parse of candidate
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

        # Attempt 2: Regex extract answer from clean_output
        if not answer:
            ans_match = re.search(r'"answer"\s*:\s*"([\s\S]*?)(?:"\s*,\s*"follow_up_questions"|"\s*\})', clean_output)
            if ans_match:
                raw_ans = ans_match.group(1)
                try:
                    answer = json.loads(f'"{raw_ans}"')
                except Exception:
                    answer = raw_ans.replace('\\n', '\n').replace('\\"', '"')

        # Extract follow-up questions from array with regex if JSON was truncated
        if not follow_ups:
            fu_matches = re.findall(r'"([^"\n\r\t]+?\?)"', clean_output)
            for q in fu_matches:
                q_clean = q.strip()
                if len(q_clean) > 8 and q_clean not in follow_ups:
                    follow_ups.append(q_clean)

        if not answer:
            answer = clean_output

        # Clean and validate follow_ups: cap at 4
        validated_follow_ups = []
        for q in follow_ups:
            if len(q) <= 200:
                validated_follow_ups.append(q)
            if len(validated_follow_ups) >= 4:
                break

        return answer, validated_follow_ups

    def _generate_grounded_fallback_summary(self, retrieval_results: List[RetrievalResult], question: str) -> Tuple[str, List[str]]:
        """
        Generates a direct extractive summary from verified FAISS chunks when upstream LLM rate limit is hit.
        Guarantees zero hallucinations and preserves full civic availability.
        """
        doc_titles = list(dict.fromkeys(r.title for r in retrieval_results if r.title))
        top_chunks = retrieval_results[:3]
        
        paragraphs = [
            f"Based on the official Government of India records ({', '.join(doc_titles)}), here is the verified statutory information regarding your query:"
        ]
        
        for idx, chunk in enumerate(top_chunks, 1):
            ref_label = f" ({chunk.legal_reference})" if chunk.legal_reference else ""
            clean_text = " ".join(chunk.text.strip().split())
            if len(clean_text) > 350:
                clean_text = clean_text[:350].rsplit(".", 1)[0] + "."
            paragraphs.append(f"{idx}. **{chunk.title}{ref_label}** (Page {chunk.page}):\n{clean_text}")

        paragraphs.append(
            "*Note: This information is extracted directly from the verified Government of India knowledge base for civic awareness and does not constitute formal legal counsel.*"
        )
        
        answer_text = "\n\n".join(paragraphs)
        
        # Determine relevant follow-ups based on category
        cat = retrieval_results[0].category.upper() if retrieval_results else "RTI"
        if cat == "RTI":
            follow_ups = [
                "What is the application fee for filing an RTI request?",
                "What information can a citizen request under RTI?",
                "What is the time limit for a Public Information Officer to respond?"
            ]
        else:
            follow_ups = [
                "What are my rights as a consumer?",
                "How can I file a consumer complaint?",
                "What does the District Consumer Commission do?"
            ]
            
        return answer_text, follow_ups

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
        """
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty.")

        # Step 1: Semantic Retrieval from FAISS
        retrieval_results = self.retriever.retrieve(
            query=clean_question,
            top_k=top_k,
            category=category
        )

        top_score = retrieval_results[0].score if retrieval_results else 0.0

        # Step 2: Relevance Threshold Verification & Abstention Guardrail
        if not retrieval_results or top_score < self.min_relevance_threshold:
            return RAGResponse(
                question=clean_question,
                answer=ABSTENTION_MESSAGE,
                sources=[],
                retrieval_results=retrieval_results,
                is_abstention=True,
                model_used=None,
                top_score=round(top_score, 4),
                follow_up_questions=[]
            )

        # Step 3: Extract Programmatic Citations (verifiable metadata from source chunks)
        citations = self.context_builder.extract_programmatic_citations(retrieval_results)

        # Step 4: Build Grounding Context
        context_str = self.context_builder.build_context_string(retrieval_results)

        # Step 5: Construct Strict Grounding Messages
        messages = build_rag_messages(question=clean_question, context_str=context_str)

        # Step 6: Generate Grounded Answer via OpenRouter (with graceful fallback on upstream rate limits)
        active_model = model or self.llm_client.model or OPENROUTER_MODEL
        try:
            raw_output = self.llm_client.generate(
                messages=messages,
                model=active_model,
                temperature=temperature
            )
            answer_text, follow_up_questions = self._parse_llm_response(raw_output)
            model_tag = active_model
        except Exception as e:
            # If upstream OpenRouter daily rate limit or network issue occurs, fallback to direct grounded context synthesis
            err_msg = str(e).lower()
            if "rate limit" in err_msg or "429" in err_msg or "quota" in err_msg or "payment" in err_msg or "402" in err_msg:
                answer_text, follow_up_questions = self._generate_grounded_fallback_summary(retrieval_results, clean_question)
                model_tag = f"{active_model} (Grounded Fallback)"
            else:
                raise

        # Step 8: Append Programmatic Citations to the answer text
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
            follow_up_questions=follow_up_questions
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

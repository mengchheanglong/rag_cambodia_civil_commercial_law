"""
Evaluation pipeline for the Cambodian Legal RAG system.

Uses the RAGAS framework to measure:
  - Context Recall:      Did we retrieve the expected articles?
  - Context Precision:   Are retrieved articles actually relevant (no noise)?
  - Faithfulness:        Is the answer grounded in retrieved context?
  - Answer Relevancy:    Is the answer on-topic for the question?

Usage:
    python -m src.evaluation.evaluate_rag
    python -m src.evaluation.evaluate_rag --top-k 5 --out data/eval_report.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.application.dtos import LegalQARequest, RetrievalRequest
from src.application.use_cases.hybrid_retrieve import HybridRetrieveUseCase
from src.application.use_cases.answer_legal_qa import AnswerLegalQAUseCase
from src.config.logging import get_logger
from src.domain.entities import RetrievedDocument
from src.infrastructure.retrieval.bm25_retriever import BM25Retriever
from src.infrastructure.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.infrastructure.ai.openai_embedding import OpenAIEmbedding
from src.infrastructure.ai.openai_llm import OpenAILLM
from src.infrastructure.storage.pgvector_repository import PgVectorRepository

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
GROUND_TRUTH_PATH = BASE_DIR / "tests" / "evaluation" / "ground_truth_qa.json"
DEFAULT_REPORT_PATH = BASE_DIR / "data" / "evaluation_report.json"


# ---------------------------------------------------------------------------
# BM25-only retrieval metric (no OpenAI key required)
# ---------------------------------------------------------------------------

def _article_hit_at_k(retrieved: list[RetrievedDocument], expected_articles: list[int]) -> bool:
    """Return True if at least one expected article appears in top-K results."""
    retrieved_nums = {doc.chunk.metadata.article_number for doc in retrieved}
    return bool(retrieved_nums & set(expected_articles))


def _recall_at_k(retrieved: list[RetrievedDocument], expected_articles: list[int]) -> float:
    """Fraction of expected articles found in top-K results."""
    if not expected_articles:
        return 0.0
    retrieved_nums = {doc.chunk.metadata.article_number for doc in retrieved}
    hits = sum(1 for a in expected_articles if a in retrieved_nums)
    return hits / len(expected_articles)


def _precision_at_k(retrieved: list[RetrievedDocument], expected_articles: list[int]) -> float:
    """Fraction of top-K retrieved articles that are expected."""
    if not retrieved:
        return 0.0
    retrieved_nums = [doc.chunk.metadata.article_number for doc in retrieved]
    hits = sum(1 for a in retrieved_nums if a in expected_articles)
    return hits / len(retrieved_nums)


# ---------------------------------------------------------------------------
# RAGAS-based metrics (requires OpenAI key)
# ---------------------------------------------------------------------------

def _try_ragas_eval(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> dict[str, float]:
    """
    Attempt RAGAS evaluation. Returns empty dict if RAGAS is not installed
    or OpenAI key is missing.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)
        result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )
        return {k: float(v) for k, v in result.items()}
    except ImportError:
        logger.warning("RAGAS not installed — skipping LLM-based metrics. Run: pip install ragas datasets")
        return {}
    except Exception as exc:
        logger.warning("RAGAS evaluation failed", error=str(exc))
        return {}


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def build_retriever(top_k: int) -> HybridRetrieveUseCase:
    """Construct the hybrid retriever using available infrastructure."""
    embedder = OpenAIEmbedding()
    vector_store = PgVectorRepository()
    sparse_retriever = BM25Retriever()
    reranker = CrossEncoderReranker()
    return HybridRetrieveUseCase(
        embedder=embedder,
        vector_store=vector_store,
        sparse_retriever=sparse_retriever,
        reranker=reranker,
    )


def run_evaluation(
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    top_k: int = 5,
    use_llm: bool = False,
    output_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """
    Run full evaluation over the golden Q&A dataset.

    Args:
        ground_truth_path: Path to the JSON golden dataset.
        top_k: Number of articles to retrieve per question.
        use_llm: Whether to generate answers and run RAGAS LLM metrics.
        output_path: Where to save the evaluation report JSON.

    Returns:
        Evaluation report dictionary.
    """
    # Load golden dataset
    with open(ground_truth_path, encoding="utf-8") as f:
        golden_data: list[dict] = json.load(f)

    logger.info("Loaded golden dataset", total_questions=len(golden_data))

    # Build retriever
    retriever = build_retriever(top_k)
    qa_use_case: AnswerLegalQAUseCase | None = None
    if use_llm:
        try:
            from src.interfaces.api.dependencies import get_qa_use_case
            qa_use_case = get_qa_use_case()
        except Exception as exc:
            logger.warning("Could not build QA use case — LLM metrics disabled", error=str(exc))

    results: list[dict[str, Any]] = []
    hit_count = 0
    total_recall = 0.0
    total_precision = 0.0
    total_ragas: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_recall": [],
        "context_precision": [],
    }

    for i, item in enumerate(golden_data, 1):
        question = item["question"]
        expected_articles = item.get("expected_articles", [])
        expected_law = item.get("expected_law", "")
        reference_answer = item.get("reference_answer", "")

        logger.info(f"Evaluating question {i}/{len(golden_data)}", question=question[:60])

        start = time.perf_counter()

        # --- Retrieval ---
        req = RetrievalRequest(
            query=question,
            top_k=top_k,
            law_filter=expected_law if expected_law else None,
        )
        retrieved = retriever.execute(req)
        latency_ms = (time.perf_counter() - start) * 1000

        # --- Retrieval metrics ---
        hit = _article_hit_at_k(retrieved, expected_articles)
        recall = _recall_at_k(retrieved, expected_articles)
        precision = _precision_at_k(retrieved, expected_articles)

        hit_count += int(hit)
        total_recall += recall
        total_precision += precision

        retrieved_articles = [
            {
                "article_number": doc.chunk.metadata.article_number,
                "law_name": doc.chunk.metadata.law_name,
                "score": doc.rerank_score or doc.rrf_score or doc.sparse_score,
                "content_preview": doc.chunk.content[:200],
            }
            for doc in retrieved
        ]

        result_item: dict[str, Any] = {
            "question_id": i,
            "question": question,
            "expected_articles": expected_articles,
            "expected_law": expected_law,
            "retrieved_articles": retrieved_articles,
            "hit_at_k": hit,
            "recall_at_k": round(recall, 4),
            "precision_at_k": round(precision, 4),
            "latency_ms": round(latency_ms, 1),
            "ragas_metrics": {},
            "generated_answer": None,
        }

        # --- Optional LLM answer + RAGAS ---
        if qa_use_case is not None:
            try:
                qa_req = LegalQARequest(
                    question=question,
                    top_k=top_k,
                    law_filter=expected_law if expected_law else None,
                )
                qa_response = qa_use_case.execute(qa_req)
                answer = qa_response.answer
                contexts = [doc.chunk.content for doc in retrieved]

                ragas_scores = _try_ragas_eval(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=reference_answer,
                )
                result_item["generated_answer"] = answer
                result_item["ragas_metrics"] = ragas_scores

                for metric, val in ragas_scores.items():
                    if metric in total_ragas:
                        total_ragas[metric].append(val)
            except Exception as exc:
                logger.warning("LLM generation failed for question", question=question[:40], error=str(exc))

        results.append(result_item)

    # --- Aggregate metrics ---
    n = len(golden_data)
    aggregate: dict[str, Any] = {
        "total_questions": n,
        "hit_rate_at_k": round(hit_count / n, 4),
        "mean_recall_at_k": round(total_recall / n, 4),
        "mean_precision_at_k": round(total_precision / n, 4),
        "top_k": top_k,
    }

    for metric, vals in total_ragas.items():
        if vals:
            aggregate[f"mean_{metric}"] = round(sum(vals) / len(vals), 4)

    # --- Failure analysis ---
    failures = [r for r in results if not r["hit_at_k"]]
    aggregate["total_failures"] = len(failures)
    aggregate["failure_questions"] = [f["question"] for f in failures]

    report: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "corpus": "Cambodian Civil & Commercial Law",
        "aggregate_metrics": aggregate,
        "results": results,
    }

    # --- Save report ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(
        "Evaluation complete",
        hit_rate=aggregate["hit_rate_at_k"],
        mean_recall=aggregate["mean_recall_at_k"],
        mean_precision=aggregate["mean_precision_at_k"],
        report_path=str(output_path),
    )
    return report


def print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable summary of the evaluation results."""
    agg = report["aggregate_metrics"]
    print("\n" + "=" * 60)
    print("  RAG CAMBODIA LAW -- EVALUATION REPORT")
    print("=" * 60)
    print(f"  Questions evaluated : {agg['total_questions']}")
    print(f"  Top-K retrieved     : {agg['top_k']}")
    print()
    print(f"  Hit Rate  @K        : {agg['hit_rate_at_k']:.1%}")
    print(f"  Mean Recall  @K     : {agg['mean_recall_at_k']:.1%}")
    print(f"  Mean Precision @K   : {agg['mean_precision_at_k']:.1%}")

    if "mean_faithfulness" in agg:
        print()
        print("  --- RAGAS LLM Metrics ---")
        print(f"  Faithfulness        : {agg.get('mean_faithfulness', '-'):.3f}")
        print(f"  Answer Relevancy    : {agg.get('mean_answer_relevancy', '-'):.3f}")
        print(f"  Context Recall      : {agg.get('mean_context_recall', '-'):.3f}")
        print(f"  Context Precision   : {agg.get('mean_context_precision', '-'):.3f}")

    print()
    failures = agg.get("failure_questions", [])
    if failures:
        print(f"  [FAIL] Failed Questions ({len(failures)}):")
        for q in failures[:5]:
            print(f"     - {q[:70]}...")
    else:
        print("  [OK] All questions had at least one expected article retrieved!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG Cambodia Law system.")
    parser.add_argument("--top-k", type=int, default=5, help="Articles to retrieve per question.")
    parser.add_argument("--use-llm", action="store_true", help="Generate answers and run RAGAS metrics (requires OPENAI_API_KEY).")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH, help="Output path for evaluation report JSON.")
    parser.add_argument("--dataset", type=Path, default=GROUND_TRUTH_PATH, help="Path to ground truth JSON file.")
    args = parser.parse_args()

    report = run_evaluation(
        ground_truth_path=args.dataset,
        top_k=args.top_k,
        use_llm=args.use_llm,
        output_path=args.out,
    )
    print_summary(report)

"""Retrieval evaluation metrics and utilities."""

from typing import Dict, List, Set

from ragvix.utils.logging import get_logger

logger = get_logger(__name__)


def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 5
) -> float:
    """Compute Recall@k metric.
    
    Args:
        retrieved_ids: List of retrieved document IDs (in order)
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Cutoff for evaluation
        
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0
    
    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k & relevant_ids) / len(relevant_ids)


def compute_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int = 5
) -> float:
    """Compute Precision@k metric.
    
    Args:
        retrieved_ids: List of retrieved document IDs (in order)
        relevant_ids: Set of relevant document IDs (ground truth)
        k: Cutoff for evaluation
        
    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if not retrieved_ids[:k]:
        return 0.0
    
    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k & relevant_ids) / len(retrieved_at_k)


def evaluate_retrieval(
    retrieval_results: List[Dict],
    ground_truth: Dict[str, Set[str]],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict:
    """Evaluate retrieval performance across multiple queries.
    
    Args:
        retrieval_results: List of retrieval results
        ground_truth: Mapping of query_id -> set of relevant doc IDs
        k_values: List of k values to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})
    
    for result in retrieval_results:
        query_id = result["query"]
        retrieved_ids = [hit["metadata"]["arxiv_id"] for hit in result.get("results", [])]
        relevant_ids = ground_truth.get(query_id, set())
        
        for k in k_values:
            recall = compute_recall_at_k(retrieved_ids, relevant_ids, k)
            precision = compute_precision_at_k(retrieved_ids, relevant_ids, k)
            
            metrics[f"recall@{k}"].append(recall)
            metrics[f"precision@{k}"].append(precision)
    
    # Compute averages
    avg_metrics = {}
    for metric, values in metrics.items():
        avg_metrics[metric] = sum(values) / len(values) if values else 0.0
    
    return avg_metrics


# Seed evaluation data (placeholder for Week-1)
SEED_EVALUATION_DATA = {
    "contrastive learning": {
        "2106.04102",  # SimCLR paper
        "2002.05709",  # MoCo paper
    },
    "attention mechanisms": {
        "1706.03762",  # Transformer paper
        "1909.11942",  # DistilBERT
    },
    "diffusion models": {
        "2006.11239",  # DDPM
        "2105.05233",  # Improved DDPM
    }
}


def run_seed_evaluation(retriever, k: int = 5) -> Dict:
    """Run evaluation on seed queries (placeholder implementation).
    
    Args:
        retriever: Retriever instance
        k: Number of results to retrieve
        
    Returns:
        Evaluation results
    """
    logger.info("Running seed evaluation (placeholder)")
    
    results = []
    for query, relevant_ids in SEED_EVALUATION_DATA.items():
        try:
            search_results = retriever.search(query, k=k)
            results.append({
                "query": query,
                "results": search_results,
            })
        except Exception as e:
            logger.warning(f"Failed to evaluate query '{query}': {e}")
    
    if not results:
        return {"error": "No evaluation results"}
    
    # Compute metrics
    metrics = evaluate_retrieval(results, SEED_EVALUATION_DATA)
    
    logger.info(f"Evaluation complete: {len(results)} queries")
    return {
        "num_queries": len(results),
        "metrics": metrics,
        "queries_evaluated": list(SEED_EVALUATION_DATA.keys()),
    }
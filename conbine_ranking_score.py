from collections import defaultdict
from typing import List, Tuple, Any

def merge_semantic_bm25(
    results_semantic: List[Tuple[float, Any]],
    results_bm25: List[Tuple[float, Any]],
    lambda_semantic: float = 0.6,
    top_k: int = -1
) -> List[Tuple[float, Any]]:
    """
    Merge two result lists (semantic and BM25) using a weighted sum of scores.
    If the same document appears in both, their scores are combined.
    
    Args:
        results_semantic: List of (score, doc_id) from semantic search
        results_bm25: List of (score, doc_id) from BM25 search
        lambda_semantic: Weight for semantic score in the final score
        top_k: Number of top documents to return

    Returns:
        A sorted list of (final_score, doc_id), top-k by final_score
    """
    combined_scores = defaultdict(float)

    for score, doc_id in results_semantic:
        combined_scores[doc_id] += lambda_semantic * score

    for score, doc_id in results_bm25:
        combined_scores[doc_id] += (1 - lambda_semantic) * score

    merged = sorted(combined_scores.items(), key=lambda x: -x[1])
    merged = [(score, str(doc_id)) for doc_id, score in merged]

    if top_k > 0:
        merged = merged[:top_k]
    return merged

def reciprocal_rank_fusion(rank_lists: List[List[Tuple[float, Any]]], k: int = 60) -> List[Tuple[float, Any]]:
    """
    Applies Reciprocal Rank Fusion (RRF) to merge multiple ranked lists.

    Args:
        rank_lists (List[List[Tuple[float, Any]]]): A list of ranked sources.
            Each source is a list of tuples (score, aid), where aid is the document ID.
        k (int): Smoothing constant, default is 60 as suggested in the original RRF paper.

    Returns:
        List[Tuple[float, Any]]: A fused list of documents, sorted by descending RRF score.
    """
    rrf_scores = defaultdict(float)

    for source in rank_lists:
        sorted_source = sorted(source, key=lambda x: x[0], reverse=True)

        for rank, (_, aid) in enumerate(sorted_source):
            rrf_scores[aid] += 1 / (k + rank + 1)

    return sorted([(score, aid) for aid, score in rrf_scores.items()], reverse=True)

def normalize_score_tuples(score_tuples: List[Tuple[float, Any]]) -> List[Tuple[float, Any]]:
    """
    Normalize scores in a list of (score, index) tuples to the range [0, 1].
    Returns a list of (normalized_score, index) tuples.
    """
    if not score_tuples:
        return []

    scores = [s for s, _ in score_tuples]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [(1.0, idx) for _, idx in score_tuples]

    return [((s - min_score) / (max_score - min_score), idx) for s, idx in score_tuples]
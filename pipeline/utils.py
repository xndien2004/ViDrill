from typing import List, Tuple, Union
import numpy as np

def filter_main_ids(items: List[str]) -> List[int]:
    result = []
    seen_main_ids = set()

    for item in items:
        if '_' in item:
            main_id = item.split('_')[0]
            if main_id in seen_main_ids:
                continue
            else:
                result.append(int(main_id))
                seen_main_ids.add(main_id)
        else:
            try:
                result.append(int(item))
                seen_main_ids.add(item)
            except ValueError:
                print(f"Skipping invalid item: {item}")
                continue

    return result

def select_by_delta(reranked_results, delta=0.05):
    selected = []
    for i in range(len(reranked_results)):
        score, aid, *_ = reranked_results[i]
        selected.append(aid)
        if i < len(reranked_results) - 1:
            next_score = reranked_results[i + 1][0]
            diff = score - next_score
            if diff > delta:
                print(f"Breaking at rank {i} with delta {diff:.4f} > {delta}")
                break
    return selected

def adaptive_k(
    scored_contexts: Union[List[Tuple[float, str]], List[Tuple[float, str, str]]],
    buffer_size: int = 5,
    gap_search_ratio: float = 0.9
) -> List[Tuple]:
    """
    Apply Adaptive-k retrieval to a list of context passages with precomputed similarity scores.

    Args:
        scored_contexts: A list of tuples (score, aid, context) or (score, context),
            where score is the similarity score between query and passage.
        buffer_size (int): Number of additional passages to include after the adaptive cutoff.
        gap_search_ratio (float): Only consider the top X% of the sorted scores to avoid noise.

    Returns:
        A list of top-k + buffer passages selected by Adaptive-k.
    """

    # 1. Sort contexts by score in descending order
    sorted_contexts = sorted(scored_contexts, key=lambda x: x[0], reverse=True)
    sorted_scores = [item[0] for item in sorted_contexts]

    # 2. Compute score gaps between consecutive passages
    gaps = [sorted_scores[i] - sorted_scores[i + 1] for i in range(len(sorted_scores) - 1)]

    # 3. Search for the largest gap in the top gap_search_ratio portion
    max_idx = int(len(gaps) * gap_search_ratio)
    limited_gaps = gaps[:max_idx]

    if not limited_gaps:
        k = 1
    else:
        k = limited_gaps.index(max(limited_gaps)) + 1

    # 4. Retrieve top-k + buffer passages
    num_to_select = min(k + buffer_size, len(sorted_contexts))
    return sorted_contexts[:num_to_select]
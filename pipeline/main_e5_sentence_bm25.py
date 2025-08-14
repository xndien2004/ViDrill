import os
import pandas as pd
import argparse
import json
from typing import List, Tuple, Union
import numpy as np

from ..search import Qdrant, BM25
from ..conbine_ranking_score import normalize_score_tuples, merge_semantic_bm25, reciprocal_rank_fusion
from ..encoder import BGEM3Encoder, BGEM3Reranker, E5InstructEncoder, SentenceEncoder
from .llm_infer import QwenLegalReasoner
 

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

def main(args):
    if "json" in args.input:
        df = pd.read_json(args.input)
    else:
        df = pd.read_csv(args.input)
    corpus_df = pd.read_csv(args.corpus_path)
    corpus_df = corpus_df.dropna(subset=["aid", "context"])
    corpus_aids = corpus_df["aid"].tolist()
    corpus_contexts = corpus_df["context"].tolist()
    bm25 = BM25(corpus_contexts,corpus_aids)
    e5_encoder = E5InstructEncoder(model_name=args.e5_model_name)
    gte_encoder = SentenceEncoder(model_name=args.gte_model_name)
    bge_reranker = BGEM3Reranker(model_name=args.bge_rerank_name)
    e5_qdrant = Qdrant(path=args.e5_qdrant_path, collection_name="corpus")
    gte_qdrant = Qdrant(path=args.gte_qdrant_path, collection_name="corpus")
    df_new = []
    temp_path = "temp.json"
    if os.path.exists(temp_path):
        os.remove(temp_path)
    for idx, row in df.iterrows():
        question = row["question"]
        id = row["qid"]
        bm25_results = bm25.search(question, top_k=args.top_k_bm25)

        list_aids = [aid for _, aid, _ in bm25_results]

        e5_results = e5_qdrant.search_by_aids(question, e5_encoder, list_aids, args.top_k_retrieval)
        gte_results = gte_qdrant.search_by_aids(question, gte_encoder, list_aids, args.top_k_retrieval)

        relevant_laws = []
        for result in e5_results.points:
            aid = result.payload["aid"]
            context = result.payload["context"]
            relevant_laws.append((aid, context))

        for result in gte_results.points:
            aid = result.payload["aid"]
            context = result.payload["context"]
            if aid not in [law[0] for law in relevant_laws]:
                relevant_laws.append((aid, context))

        reranked_results = bge_reranker.rerank(question, relevant_laws)

        relevant_laws_full = [aid for score, aid, _ in reranked_results if score >= args.threshold]
        scores = [score for score, _, _ in reranked_results]

        if len(relevant_laws_full) == 0:
            print(f"No relevant laws found for question ID {id}. Using top else threshold.")
            relevant_laws_full = [aid for idx, (score, aid, _) in enumerate(reranked_results) if idx < args.top_else_threshold]

        relevant_laws = filter_main_ids(relevant_laws_full) 
        relevant_laws = list(set(relevant_laws))
        df_new.append({"qid": id, "relevant_laws": relevant_laws})
        print(f"Processed {idx + 1}/{len(df)}: --> {relevant_laws}")
        with open(temp_path, "a") as f:
            json.dump({"qid": id, "relevant_laws": relevant_laws, "scores": scores}, f, ensure_ascii=False)
            f.write("\n")
    # df_new = pd.DataFrame(df_new)
    with open(args.output, "w") as f:
        json.dump(df_new, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViDRILL main script")
    parser.add_argument("--input", type=str, required=True, help="Input file path (CSV or JSON)")
    parser.add_argument("--output", type=str, required=True, help="Output file path for results")
    parser.add_argument("--e5_model_name", type=str, required=True, help="E5 model name for encoding")
    parser.add_argument("--gte_model_name", type=str, required=True, help="gte model name for encoding")
    parser.add_argument("--bge_rerank_name", type=str, required=True, help="BGE model name for reranking")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus CSV file")

    parser.add_argument("--top_k_retrieval", type=int, default=50, help="Top K for retrieval")
    parser.add_argument("--top_k_bm25", type=int, default=50, help="Top K for BM25 retrieval")
    parser.add_argument("--top_else_threshold", type=int, default=20, help="Top K for reranking")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for reranking scores")
    parser.add_argument("--threshold2", type=float, default=0.1, help="Second threshold for reranking scores")
    parser.add_argument("--e5_qdrant_path", type=str, default="./qdrant/e5", help="Path to E5 Qdrant database")
    parser.add_argument("--gte_qdrant_path", type=str, default="./qdrant/gte", help="Path to gte Qdrant database")
    args = parser.parse_args()
    main(args)

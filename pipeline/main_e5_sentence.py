import os
import pandas as pd
import argparse
import json

from ..search import Qdrant, BM25
from ..encoder import BGEM3Reranker, E5InstructEncoder, SentenceEncoder
from .utils import filter_main_ids
 

def main(args):
    if "json" in args.input:
        df = pd.read_json(args.input)
    else:
        df = pd.read_csv(args.input)
    # df = df.head(200)
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
        e5_results = e5_qdrant.search(question, e5_encoder, args.top_k_retrieval)
        gte_results = gte_qdrant.search(question, gte_encoder, args.top_k_retrieval)
        relevant_laws = []
        results_e5 = []
        results_gte = []
        for result in e5_results.points:
            aid = result.payload["aid"]
            context = result.payload["context"]
            relevant_laws.append((aid, context))
            results_e5.append((result.score, aid))

        for result in gte_results.points:
            aid = result.payload["aid"]
            context = result.payload["context"]
            if aid not in [law[0] for law in relevant_laws]:
                relevant_laws.append((aid, context))
            results_gte.append((result.score, aid))

        reranked_results = bge_reranker.rerank(question, relevant_laws)
        relevant_laws_full = [aid for score, aid, _ in reranked_results if score >= args.threshold]
        scores = [score for score, _, _ in reranked_results]

        use_top_else_threshold = False
        if len(relevant_laws_full) == 0:
            use_top_else_threshold = True
            print(f"No relevant laws found for question ID {id}. Using top else threshold.")
            relevant_laws_full = [aid for idx, (score, aid, _) in enumerate(reranked_results) if idx < args.top_else_threshold]



        relevant_laws = filter_main_ids(relevant_laws_full) 
        relevant_laws = list(set(relevant_laws))
        # relevant_laws = relevant_laws[:10]
        df_new.append({"qid": id, "question": question, "relevant_laws": relevant_laws})
        print(f"Processed {idx + 1}/{len(df)}: --> {relevant_laws}")
        with open(temp_path, "a") as f:
            json.dump({"qid": id, "use top else threshold": use_top_else_threshold, "relevant_laws": relevant_laws, "scores": scores}, f, ensure_ascii=False)
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
    parser.add_argument("--top_k_retrieval", type=int, default=50, help="Top K for retrieval")
    parser.add_argument("--top_else_threshold", type=int, default=5, help="Top K for else case")
    parser.add_argument("--top_k_rerank", type=int, default=20, help="Top K for reranking")
    parser.add_argument("--threshold", type=float, default=0.1, help="Threshold for reranking scores")
    parser.add_argument("--threshold2", type=float, default=0.05, help="Secondary threshold for reranking scores")
    parser.add_argument("--e5_qdrant_path", type=str, default="./qdrant/e5", help="Path to E5 Qdrant database")
    parser.add_argument("--gte_qdrant_path", type=str, default="./qdrant/gte", help="Path to gte Qdrant database")
    args = parser.parse_args()
    main(args)

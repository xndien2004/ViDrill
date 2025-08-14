import json
import pandas as pd

from ..search import Qdrant
from ..encoder import E5Encoder, BGEM3Encoder, SentenceEncoder, BGEM3Reranker, E5InstructEncoder
from typing import List, Union
import ast

def main(args):
    train_df = pd.read_csv(args.train_file)
    # train_df = train_df.head(10)

    bge_encoder = BGEM3Encoder(model_name=args.bge_model_name)
    # e5_encoder = E5Encoder(model_name=args.e5_model_name)
    # gte_encoder = SentenceEncoder(model_name=args.gte_model_name)
    # e5_instruct_encoder = E5InstructEncoder(model_name=args.e5_instruct_model_name)


    bge_search = Qdrant(path=args.bge_qdrant_path, collection_name="corpus")
    # e5_search = Qdrant(path=args.e5_qdrant_path, collection_name="corpus")
    # gte_search = Qdrant(path=args.gte_qdrant_path, collection_name="corpus")
    # e5_instruct_search = Qdrant(path=args.e5_instruct_qdrant_path, collection_name="corpus")

    # print(gte_search.client.count(gte_search.collection_name))
    # print(e5_search.client.count(e5_search.collection_name))
    # print(gte_search.client.count(gte_search.collection_name))

    train_new = []
    for index, row in train_df.iterrows():
        pos = ast.literal_eval(row["pos"])
        query = row["query"]
        pos_aids = ast.literal_eval(row["pos_aids"])

        bge_results = bge_search.search(query, bge_encoder, args.top_k_retrieval+10)

        hard_negatives = []
        neg_scores = []
        neg_aids = []
        for re in bge_results.points:
            aid = re.payload.get("aid")
            if aid not in pos_aids:
                hard_negatives.append(re.payload["context"])
                neg_scores.append(re.score)
                neg_aids.append(aid)

        # gte_results = gte_search.search(query, gte_encoder, args.top_k_retrieval)
        # e5_instruct_results = e5_instruct_search.search(query, e5_instruct_encoder, args.top_k_retrieval)
        # relevant_laws = []
        # neg_aids = []
        # neg_scores = []
        # hard_negatives = []
        # for result in gte_results.points:
        #     aid = result.payload["aid"]
        #     if aid not in pos_aids:
        #         context = result.payload["context"]
        #         # relevant_laws.append((aid, context))
        #         hard_negatives.append(context)
        #         neg_aids.append(aid)
        #         neg_scores.append(result.score)

        # for result in e5_instruct_results.points:
        #     aid = result.payload["aid"]
        #     if aid not in pos_aids and aid not in neg_aids:
        #         context = result.payload["context"]
        #         hard_negatives.append(context)
        #         neg_aids.append(aid)
        #         neg_scores.append(result.score)

        # sort hard negatives by score
        # sorted_indices = sorted(range(len(neg_scores)), key=lambda i: neg_scores[i], reverse=True)
        # hard_negatives = [hard_negatives[i] for i in sorted_indices]
        # neg_scores = [neg_scores[i] for i in sorted_indices]
        # neg_aids = [neg_aids[i] for i in sorted_indices]
        hard_negatives = hard_negatives[:args.top_k_retrieval]
        neg_scores = neg_scores[:args.top_k_retrieval]
        neg_aids = neg_aids[:args.top_k_retrieval]
        

        # e5_results = e5_search.search(query, e5_encoder, args.top_k)
        # gte_results = gte_search.search(query, gte_encoder, args.top_k)
        # hard_negatives = []
        # list_id = []
        # neg_scores = []
        # for results in [bge_results, e5_results, gte_results]:
        #     for re in results.points:
        #         aid = re.payload["aid"]
        #         if aid not in pos_aids and aid not in list_id:
        #             # print(re.payload["context"])
        #             hard_negatives.append(re.payload["context"])
        #             list_id.append(aid)
        #             neg_scores.append(re.score)

        # # sort hard negatives by score
        # sorted_indices = sorted(range(len(neg_scores)), key=lambda i: neg_scores[i], reverse=True)
        # hard_negatives = [hard_negatives[i] for i in sorted_indices]
        # neg_scores = [neg_scores[i] for i in sorted_indices]

        # pos_scores = [0.0] * len(pos_aids)
        # bge_results = bge_search.search_by_aids(query, bge_encoder, pos_aids)

        # aid2idx = {aid: idx for idx, aid in enumerate(pos_aids)}

        # for re in bge_results.points:
        #     aid = re.payload.get("aid")
        #     if aid in aid2idx:
        #         pos_scores[aid2idx[aid]] = re.score
        
        train_new.append({
            "query": query,
            "pos": pos,
            "pos_aids": pos_aids,
            # "pos_scores": pos_scores,
            "neg": hard_negatives,
            "neg_aids": neg_aids,
            "neg_scores": neg_scores
        })
        print(f"Processed {index + 1}/{len(train_df)} queries") 
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(train_new, f, ensure_ascii=False, indent=4)


    # batch search 
    # for batch_start in range(0, len(train_df), args.batch_size):
    #     batch_df = train_df.iloc[batch_start: batch_start + args.batch_size]

    #     queries = batch_df["question"].tolist()
    #     pos_list = batch_df["pos"].tolist()
    #     pos_aids_list = [ast.literal_eval(aids) for aids in batch_df["pos_aids"]]

    #     bge_results = bge_search.batch_search(queries, bge_encoder, args.top_k)
    #     e5_results = e5_search.batch_search(queries, e5_encoder, args.top_k)
    #     gte_results = gte_search.batch_search(queries, gte_encoder, args.top_k)
    #     for i, query in enumerate(queries):
    #         hard_negatives = []
    #         list_id = set()
    #         pos_aids = pos_aids_list[i]

    #         # Combine results from 3 models
    #         for results in [bge_results[i], e5_results[i], gte_results[i]]:
    #             for re in results:
    #                 aid = re.payload["aid"]
    #                 if aid not in pos_aids and aid not in list_id:
    #                     hard_negatives.append(re.payload["context"])
    #                     list_id.add(aid)

    #         train_new.append({
    #             "query": query,
    #             "pos": pos_list[i],
    #             "neg": hard_negatives
    #         })

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Generate hard negatives for training")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--bge_model_name", type=str, default="BAAI/bge-m3", help="BGE model name")
    parser.add_argument("--e5_model_name", type=str, default="intfloat/multilingual-e5-large", help="E5 model name")
    parser.add_argument("--gte_model_name", type=str, default="Alibaba-NLP/gte-multilingual-base", help="GTE model name")
    parser.add_argument("--e5_instruct_model_name", type=str, default="BAAI/bge-m3-instruct", help="E5 Instruct model name")
    parser.add_argument("--bge_qdrant_path", type=str, default="./bge_qdrant", help="Path to BGE Qdrant data directory")
    parser.add_argument("--e5_qdrant_path", type=str, default="./e5_qdrant", help="Path to E5 Qdrant data directory")
    parser.add_argument("--gte_qdrant_path", type=str, default="./gte_qdrant", help="Path to GTE Qdrant data directory")
    parser.add_argument("--e5_instruct_qdrant_path", type=str, default="BAAI/bge-m3-instruct", help="E5 Instruct model name")
    parser.add_argument("--bge_rerank_name", type=str, default="BAAI/bge-m3-rerank", help="BGE rerank model name")
    parser.add_argument("--top_k_retrieval", type=int, default=25, help="Number of top results to retrieve")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for hard negatives")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

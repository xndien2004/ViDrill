import pandas as pd
import ast
from qdrant_client import models

from ..encoder import BGEM3Encoder, E5InstructEncoder, BGEM3Reranker
from ..search import Qdrant



def main():
    df = pd.read_csv("/home/fit02/dien-workspace/vlsp/data_drill/maxlen450/train_450.csv")
    # encoder = BGEM3Encoder(model_name="/home/fit02/dien-workspace/vlsp/output/bge-m3-neg-top60-form-finetune")
    encoder = E5InstructEncoder(model_name="/home/fit02/dien-workspace/vlsp/output/e5-instruct-neg-top60-from-finetune-500")
    reranker = BGEM3Reranker(model_name="/home/fit02/dien-workspace/vlsp/output/bge-reranker-neg-top60-finetune")
    bge_qdrant = Qdrant(path="/home/fit02/dien-workspace/vlsp/output/dbtest", collection_name="test_collection")

    train_data = []

    for index, row in df.iterrows():
        pos = ast.literal_eval(row["pos"])
        pos_aids = ast.literal_eval(row["pos_aids"])
        query = row["question"]

        pos_valid = []
        pos_valid_aids = []
        pos_invalid = []
        pos_invalid_aids = []

        for i_pos, aid in enumerate(pos_aids):
            if "_" in aid:
                pos_valid.append(pos[i_pos])
                pos_valid_aids.append(aid)
            else:
                pos_invalid.append(pos[i_pos])
                pos_invalid_aids.append(aid)

        if pos_valid:
        

            # bge_qdrant.delete_collection()
            # bge_qdrant.create_collection(dimension=encoder.get_dimension())

            # bge_dense, bge_indices, bge_values = encoder.encode(pos_valid, type="passage")
            # points = []
            # for i in range(len(bge_dense)):
            #     points.append(
            #         models.PointStruct(
            #             id=i,
            #             vector={
            #                 "dense": bge_dense[i],
            #                 # "sparse": {
            #                 #     "indices": [int(v) for v in bge_indices[i]],
            #                 #     "values": [float(v) for v in bge_values[i]]
            #                 # }
            #             },
            #             payload={
            #                 "aid": pos_valid_aids[i],
            #                 "text": pos_valid[i],
            #             }
            #         )
            #     )
            # bge_qdrant.add_points(points)

            # search_results = bge_qdrant.search(query, encoder, top_k=1).points
            # for item in search_results:
            #     pos_invalid.append(item.payload["text"])
            #     pos_invalid_aids.append(item.payload["aid"])
    
            aid_text = [(aid, text) for aid, text in zip(pos_valid_aids, pos_valid)]
            rerank_results = reranker.rerank(query, aid_text, top_k=1)
            aid = [item[1] for item in rerank_results]
            text = [item[2] for item in rerank_results]
            pos_invalid.extend(text)
            pos_invalid_aids.extend(aid)

        train_data.append({
            "question": query,
            "pos": pos_invalid,
            "pos_aids": pos_invalid_aids
        })

    train_df = pd.DataFrame(train_data)
    train_df.to_csv("/home/fit02/dien-workspace/vlsp/data_drill/maxlen450/train_rerank_450.csv", index=False)


if __name__ == "__main__":
    main()
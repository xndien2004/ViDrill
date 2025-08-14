import pandas as pd
from qdrant_client import models
import argparse
from tqdm import tqdm

from ..encoder import E5Encoder, BGEM3Encoder, GTEEncoder
from ..search import Qdrant

def batch_encode(encoder, texts, batch_size, encode_type="passage"):
    dense_all, indices_all, values_all = [], [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding in batch"):
        batch_texts = texts[i:i+batch_size]
        dense, indices, values = encoder.encode(batch_texts, type=encode_type)
        dense_all.extend(dense)
        indices_all.extend(indices)
        values_all.extend(values)
    return dense_all, indices_all, values_all

def clean_texts(texts, payloads, ids):
    clean_texts = []
    clean_payloads = []
    clean_ids = []
    for i, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            clean_texts.append(text.strip())
            clean_payloads.append(payloads[i])
            clean_ids.append(ids[i])
        else:
            print(f"[Warning] Skipping invalid or empty text at index {i}: {text}")
    return clean_texts, clean_payloads, clean_ids

def main(args):
    corpus_df = pd.read_csv(args.corpus_path)
    # corpus_df = corpus_df.head(50)

    # load the encoder
    bge_encoder = BGEM3Encoder(model_name=args.bge_model_name)
    e5_encoder = E5Encoder(model_name=args.e5_model_name)
    gte_encoder = GTEEncoder(model_name=args.gte_model_name)

    # create Qdrant instance
    bge_qdrant = Qdrant(path=args.bge_qdrant_path, collection_name=args.bge_collection_name)
    e5_qdrant = Qdrant(path=args.e5_qdrant_path, collection_name=args.e5_collection_name)
    gte_qdrant = Qdrant(path=args.gte_qdrant_path, collection_name=args.gte_collection_name)


    # create collections
    bge_qdrant.create_collection(dimension=bge_encoder.get_dimension())
    e5_qdrant.create_collection(dimension=e5_encoder.get_dimension())
    gte_qdrant.create_collection(dimension=gte_encoder.get_dimension())

    corpus_for_qdrant = []
    point_id_counter = 0

    for index, row in corpus_df.iterrows():
        point_id_counter += 1
        qdrant_id = point_id_counter 

        original_aid = row['aid']
        context_text = row['context'] 

        chunk_identifier = f"{original_aid}_{index+1}"

        corpus_for_qdrant.append({
            "id": qdrant_id,
            "text": context_text,
            "metadata": {
                "context": context_text,
                "aid": original_aid,
                "chunk_identifier": chunk_identifier
            }
        })
    texts_to_encode = [item['text'] for item in corpus_for_qdrant]
    payloads = [item['metadata'] for item in corpus_for_qdrant]
    ids = [item['id'] for item in corpus_for_qdrant]

    texts_to_encode, payloads, ids = clean_texts(texts_to_encode, payloads, ids)
    
    # encode and add points
    # bge_vectors, bge_indices, bge_values = bge_encoder.encode(texts_to_encode, type="passage")
    # e5_vectors, e5_indices, e5_values = e5_encoder.encode(texts_to_encode, type="passage")
    bge_dense_vecs, bge_indices_list, bge_values_list = batch_encode(bge_encoder, texts_to_encode, args.batch_size)
    e5_dense_vecs, _, _ = batch_encode(e5_encoder, texts_to_encode, args.batch_size)
    gte_dense_vecs, _, _ = batch_encode(gte_encoder, texts_to_encode, args.batch_size)

    # Build Qdrant points
    points_to_bge, points_to_e5, points_to_gte = [], [], []

    for i in range(len(texts_to_encode)):
        sparse_vector_data = {
            'indices': bge_indices_list[i],
            'values': bge_values_list[i]
        }

        points_to_bge.append(
            models.PointStruct(
                id=ids[i],
                vector={
                    "dense": bge_dense_vecs[i],
                    "sparse": sparse_vector_data
                },
                payload=payloads[i]
            )
        )
        points_to_e5.append(
            models.PointStruct(
                id=ids[i],
                vector={"dense": e5_dense_vecs[i]},
                payload=payloads[i]
            )
        )
        points_to_gte.append(
            models.PointStruct(
                id=ids[i],
                vector={"dense": gte_dense_vecs[i]},
                payload=payloads[i]
            )
        )
    bge_qdrant.add_points(points=points_to_bge)
    e5_qdrant.add_points(points=points_to_e5)
    gte_qdrant.add_points(points=points_to_gte)
    print(bge_qdrant.client.count(bge_qdrant.collection_name))
    print(e5_qdrant.client.count(e5_qdrant.collection_name))
    print(gte_qdrant.client.count(gte_qdrant.collection_name))
    print(f"Added {len(corpus_for_qdrant)} points to Qdrant collections.")

def arg_parser():
    parser = argparse.ArgumentParser(description="Build Qdrant corpus from CSV")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus CSV file")
    parser.add_argument("--bge_model_name", type=str, default="BAAI/bge-m3", help="BGE model name")
    parser.add_argument("--e5_model_name", type=str, default="intfloat/multilingual-e5-large", help="E5 model name")
    parser.add_argument("--gte_model_name", type=str, default="Alibaba-NLP/gte-multilingual-base", help="GTE model name")
    parser.add_argument("--bge_qdrant_path", type=str, default="./bge_qdrant", help="Path to BGE Qdrant data directory")
    parser.add_argument("--e5_qdrant_path", type=str, default="./e5_qdrant", help="Path to E5 Qdrant data directory")
    parser.add_argument("--gte_qdrant_path", type=str, default="./gte_qdrant", help="Path to GTE Qdrant data directory")
    parser.add_argument("--bge_collection_name", type=str, default="bge_corpus", help="BGE collection name in Qdrant")
    parser.add_argument("--e5_collection_name", type=str, default="e5_corpus", help="E5 collection name in Qdrant")
    parser.add_argument("--gte_collection_name", type=str, default="gte_corpus", help="GTE collection name in Qdrant")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    return parser

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    main(args)

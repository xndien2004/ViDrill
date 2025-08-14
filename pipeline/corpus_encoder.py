import pandas as pd
from qdrant_client import models
import argparse
from tqdm import tqdm

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
            print(f"[Warning] Skipping invalid or empty text at index {ids[i]}: {text}")
    return clean_texts, clean_payloads, clean_ids

def main(args):
    corpus_df = pd.read_csv(args.corpus_path)

    # load the encoder
    if "bge" in args.model_name.lower():
        print(f"Using BGE model: {args.model_name}")
        from ..encoder import BGEM3Encoder
        encoder = BGEM3Encoder(model_name=args.model_name)
    elif "e5-instruct" in args.model_name.lower():
        print(f"Using E5-Instruct model: {args.model_name}")
        from ..encoder import E5InstructEncoder
        encoder = E5InstructEncoder(model_name=args.model_name)
    elif "gte" in args.model_name.lower() or "vn" in args.model_name.lower():
        print(f"Using model: {args.model_name}")
        from ..encoder import SentenceEncoder
        encoder = SentenceEncoder(model_name=args.model_name)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    qdrant = Qdrant(path=args.qdrant_path, collection_name="corpus")

    # create collections
    qdrant.delete_collection()
    qdrant.create_collection(dimension=encoder.get_dimension())

    corpus_for_qdrant = []
    point_id_counter = 0

    for index, row in corpus_df.iterrows():
        point_id_counter += 1
        qdrant_id = point_id_counter 

        original_aid = row["aid"]
        context_text = row["context"]

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
    dense_vecs, indices_list, values_list = batch_encode(encoder, texts_to_encode, args.batch_size)

    # Build Qdrant points
    points_to_bge = []

    for i in range(len(texts_to_encode)):
        if "bge" in args.model_name.lower():
            vector = {
                        "dense": dense_vecs[i],
                        "sparse": {
                            'indices': indices_list[i],
                            'values': values_list[i]
                        }
                    }
        else:
            vector = {
                        "dense": dense_vecs[i]
                    }
        points_to_bge.append(
            models.PointStruct(
                id=ids[i],
                vector=vector,
                payload=payloads[i]
            )
        )
    qdrant.add_points(points=points_to_bge)
    print(qdrant.client.count(qdrant.collection_name))
    print(f"Added {len(corpus_for_qdrant)} points to Qdrant collections.")

def arg_parser():
    parser = argparse.ArgumentParser(description="Build Qdrant corpus from CSV")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus CSV file")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-m3", help="BGE model name")
    parser.add_argument("--qdrant_path", type=str, default="./db_pipline", help="Path to BGE Qdrant data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    return parser

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    main(args)

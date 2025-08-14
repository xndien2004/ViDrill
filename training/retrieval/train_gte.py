import argparse
import os
import json
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, InputExample, losses, LoggingHandler

from torch.utils.data import DataLoader
from sentence_transformers import models, evaluation
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path):
    if "json" in path:
        df = pd.read_json(path)
    else: 
        df = pd.read_csv(path)
    
    examples = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        query = row["query"]
        
        pos_passages = row["pos"] if isinstance(row["pos"], list) else [row["pos"]]
        # neg_passages = row["neg"] if isinstance(row["neg"], list) else [row["neg"]]

        for pos in pos_passages:
            examples.append(InputExample(texts=[query, pos], label=1.0))
        # for neg in neg_passages:
        #     examples.append(InputExample(texts=[query, neg], label=0.0))

    return examples


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_examples = load_data(args.train_data)

    # Load model
    model = SentenceTransformer(args.model_name_or_path, cache_folder=args.cache_dir, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length
    model = model.to(device)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    # train_loss = losses.CosineSimilarityLoss(model)
    # train_loss = InfoNCELossWithTemperature(model)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=10,
        output_path=args.output_dir,
        use_amp=True,  # FP16
        show_progress_bar=True,
        optimizer_params={'lr': args.lr},
        save_best_model=True,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to JSON with query, pos, neg")
    parser.add_argument("--model_name_or_path", type=str, default="intfloat/multilingual-e5-large-instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/e5_neg_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--cache_dir", type=str, default="hf_cache")
    args = parser.parse_args()
    main(args)

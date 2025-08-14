import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    EarlyStoppingCallback
)
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score
from transformers.trainer_utils import EvalPrediction


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RerankDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=2048, num_neg=10):
        self.pairs = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_neg = num_neg

        df = pd.read_json(json_file)
        for _, row in df.iterrows():
            query = row["query"]
            pos_list = row["pos"]
            neg_list = row["neg"]

            sampled_neg = random.sample(neg_list, min(len(neg_list), self.num_neg))

            for pos in pos_list:
                self.pairs.append((query, pos))
                self.labels.append(1)
            for neg in sampled_neg:
                self.pairs.append((query, neg))
                self.labels.append(0)

        combined = list(zip(self.pairs, self.labels))
        random.shuffle(combined)
        self.pairs, self.labels = zip(*combined)
        self.pairs = list(self.pairs)
        self.labels = list(self.labels)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        text1, text2 = self.pairs[idx]
        label = self.labels[idx]
        tokenized = self.tokenizer(
            text1, text2,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in tokenized.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)
        return item


def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Alibaba-NLP/gte-multilingual-reranker-base")
    parser.add_argument("--output_dir", type=str, default="./output_trainer_reranker")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    full_dataset = RerankDataset(args.train_file, tokenizer)

    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, val_size])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        trust_remote_code=True
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        fp16=args.fp16,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

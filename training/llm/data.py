import pandas as pd
from datasets import Dataset
import ast
import random

from .prompt import *

def get_data_grpo(train_path: str, topk_neg: int) -> Dataset:
    df = pd.read_json(train_path)
    processed = []
    max_length = 0

    for _, row in df.iterrows():
        query = str(row['query']).strip()

        pos = ast.literal_eval(row['pos']) if isinstance(row['pos'], str) else row['pos']
        neg = ast.literal_eval(row['neg']) if isinstance(row['neg'], str) else row['neg']
        pos_aids = ast.literal_eval(row['pos_aids']) if isinstance(row['pos_aids'], str) else row['pos_aids']
        neg_aids = ast.literal_eval(row['neg_aids']) if isinstance(row['neg_aids'], str) else row['neg_aids']

        num_neg = max(topk_neg - len(pos), 0)
        neg = neg[:num_neg]
        neg_aids = neg_aids[:num_neg]

        combined = list(zip(neg_aids, neg))

        insertable_slots = len(combined) + 1
        num_pos = len(pos)

        if num_pos > insertable_slots:
            insert_positions = [len(combined)] * num_pos
        else:
            insert_positions = random.sample(range(insertable_slots), num_pos)
            insert_positions.sort()

        for i, (aid, passage) in enumerate(zip(pos_aids, pos)):
            insert_idx = insert_positions[i]
            combined.insert(insert_idx, (aid, passage))

        labeled_passages = [
            f"[{str(aid)}] {p.strip()}" for aid, p in combined
        ]
        info_block = "<information>\n" + "\n\n".join(labeled_passages) + "\n</information>"

        user_prompt = (
            f"Câu hỏi: {query}\n\n"
            "Dưới đây là một danh sách các tài liệu. Nhiệm vụ của bạn là xác định tài liệu nào liên quan đến câu hỏi và sắp xếp chúng theo thứ tự mức độ liên quan giảm dần.\n"
            "Chỉ trả lời bằng danh sách các tài liệu liên quan, theo định dạng chuỗi các danh sách tài liệu liên quan: <answer>[\"...\", \"...\"]</answer>\n\n"
            + info_block
        )

        relevant_aid_set = set(str(aid) for aid in pos_aids)
        related_docs = [str(aid) for aid, _ in combined if str(aid) in relevant_aid_set]

        answer_block = f"<answer>{related_docs}</answer>"
        max_length = max(max_length, len(user_prompt.split(" ")))

        processed.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "answer": answer_block
        })

    print(f"Max length of prompt: {max_length}")
    return Dataset.from_list(processed)



def get_data_binary_grpo(train_path: str, topk_neg: int) -> Dataset:
    df = pd.read_json(train_path)
    processed = []

    for _, row in df.iterrows():
        query = str(row['query']).strip()
        pos = ast.literal_eval(row['pos']) if isinstance(row['pos'], str) else row['pos']
        neg = ast.literal_eval(row['neg']) if isinstance(row['neg'], str) else row['neg']

        passages = pos + neg[:topk_neg]
        labels = [1] * len(pos) + [0] * min(topk_neg, len(neg))

        combined = list(zip(passages, labels))
        random.shuffle(combined)

        labeled_passages = [f"[{i}]: {p.strip()}" for i, (p, _) in enumerate(combined)]
        binary_labels = [label for _, label in combined]

        info_block = "<information>\n" + "\n\n".join(labeled_passages) + "\n</information>"

        user_prompt = (
            f"Câu hỏi: {query}\n\n"
            "Dưới đây là một danh sách các tài liệu. Nhiệm vụ của bạn là xác định đoạn nào liên quan đến câu hỏi.\n"
            "Hãy suy nghĩ trong thẻ <think>...</think> và trả lời trong <answer>[0, 1, ...]</answer>.\n\n"
            + info_block
        )

        answer_block = f"<answer>{binary_labels}</answer>"

        processed.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT_BINARY},
                {"role": "user", "content": user_prompt}
            ],
            "answer": answer_block
        })

    return Dataset.from_list(processed)

def get_data_binary(train_path: str, topk_neg: int) -> Dataset:
    df = pd.read_json(train_path)
    processed = []

    for _, row in df.iterrows():
        query = str(row['query']).strip()
        pos = ast.literal_eval(row['pos']) if isinstance(row['pos'], str) else row['pos']
        neg = ast.literal_eval(row['neg']) if isinstance(row['neg'], str) else row['neg']

        passages = pos + neg[:topk_neg]
        labels = [1] * len(pos) + [0] * min(topk_neg, len(neg))

        combined = list(zip(passages, labels))
        random.shuffle(combined)

        labeled_passages = [f"[{i}]: {p.strip()}" for i, (p, _) in enumerate(combined)]
        binary_labels = [label for _, label in combined]
        label_dict = {i: label for i, label in enumerate(binary_labels)}  # convert to dict

        info_block = "<information>\n" + "\n\n".join(labeled_passages) + "\n</information>"

        user_prompt = (
            f"{SYSTEM_PROMPT_BINARY}\n\n"
            f"Câu hỏi: {query}\n\n"
            "Dưới đây là một danh sách các tài liệu. Nhiệm vụ của bạn là xác định đoạn nào liên quan đến câu hỏi.\n"
            "Hãy suy nghĩ trong thẻ <think>...</think> và trả lời trong <answer>{{0: 1, 1: 0, ...}}</answer>.\n\n"
            + info_block
        )

        completion = f"<answer>{label_dict}</answer>"

        processed.append({
            "prompt": user_prompt,
            "completion": completion
        })

    return Dataset.from_list(processed)
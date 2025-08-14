from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import re
import pandas as pd
import ast
from typing import List, Tuple

from ..training.llm.prompt import SYSTEM_PROMPT
from ..encoder import BGEM3Encoder
from ..search import Qdrant

class QwenLegalReasoner:
    def __init__(self, model_name="ViQwen2-1.5B-rerank-GRPO", device="cuda:0"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def inference(self, prompt: str, max_new_tokens: int = 2048) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def rerank(self, query: str, passages: List[Tuple[str, str]], max_new_tokens: int = 512) -> list:
        if len(passages) > 10:
            print(f"Warning: More than 10 passages provided ({len(passages)}). Only the first 10 will be processed.")
            passages = passages[:10]
        labeled_passages = [f"[{aid}]: {p.strip()}" for aid, p in passages]
        info_block = "<information>\n" + "\n\n".join(labeled_passages) + "\n</information>"

        user_prompt = (
            f"Câu hỏi: {query}\n\n"
            "Dưới đây là một danh sách các tài liệu. Nhiệm vụ của bạn là xác định tài liệu nào liên quan đến câu hỏi và sắp xếp chúng theo thứ tự mức độ liên quan giảm dần.\n"
            "Chỉ trả lời bằng danh sách các tài liệu liên quan, theo định dạng chuỗi các danh sách tài liệu liên quan: <answer>[\"...\", \"...\"]</answer>\n\n"
            + info_block
        )

        response = self.inference(user_prompt, max_new_tokens=max_new_tokens)
        print(f"Response from model: {response}")
        if not response:
            print("No response from the model.")
            return []

        doc_indices = self.extract_answer(response)
        if not doc_indices:
            print("No answers extracted from the response.")
            return []

        return doc_indices

    def rerank_batched(self, query: str, passages: List[Tuple[str, str]], batch_size: int = 10) -> list:
        all_answers = []
        seen = set()
        
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i + batch_size]
            answer_ids = self.rerank(query, batch)
            for aid in answer_ids:
                if aid not in seen:
                    all_answers.append(aid)
                    seen.add(aid)

        return all_answers

    @staticmethod
    def extract_answer(text: str) -> list:
        match = re.search(r"<answer>\s*(\[[^\]]*\])\s*</answer>", text, re.DOTALL)
        if match:
            raw = match.group(1).strip()
            try:
                return ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                print("Cannot parse answer list.")
        return []

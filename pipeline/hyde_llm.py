from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List


class LegalReasoner:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "cuda:0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ).to(device).eval()
        self.device = device

        self.system_prompt = (
            "Bạn là một trợ lý chuyên gia trong lĩnh vực truy xuất thông tin. "
            "Nhiệm vụ của bạn là mở rộng truy vấn người dùng một cách thông minh để tối ưu hóa khả năng tìm kiếm tài liệu liên quan. "
            "Cho truy vấn gốc được cung cấp, bạn cần tạo ra một truy vấn mở rộng bằng cách thêm các từ khóa đồng nghĩa, cụm từ liên quan, hoặc các cách diễn đạt thay thế phù hợp. "
            "Không được thay đổi ý định ban đầu của người dùng, và không được thêm thông tin không liên quan. "
            "Đầu ra phải là một chuỗi duy nhất, rõ ràng, đầy đủ và được tối ưu hóa để truy xuất được nhiều tài liệu phù hợp hơn nhưng không làm giảm độ chính xác."
        )


    def inference(self, prompt: str, system_prompt: str = "", max_new_tokens: int = 128) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def generate_hypothetical_answer(self, question: str) -> str:
        return self.inference(prompt=question, system_prompt=self.system_prompt)

    def generate_batch_hypothetical_answers(self, questions: List[str], max_new_tokens: int = 128, batch_size: int = 4) -> List[str]:
        all_outputs = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            messages = [
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": q}
                ]
                for q in batch
            ]
            prompts = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )

            output_ids = [
                output[len(input_ids):]
                for input_ids, output in zip(model_inputs.input_ids, generated)
            ]
            decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            all_outputs.extend([d.strip() for d in decoded])

        return all_outputs


if __name__ == "__main__":
    import time

    reasoner = LegalReasoner(model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda:0")

    question = "Cho tôi hỏi mở, đóng tài khoản vốn phát hành chứng khoán bằng ngoại tệ được quy định như thế nào?"
    start = time.time()
    rewritten_question = reasoner.generate_hypothetical_answer(question)
    print("Câu hỏi viết lại:", rewritten_question)
    print("Thời gian:", round(time.time() - start, 2), "giây")

    questions = [
        "Trong xây dựng nông thôn mới giai đoạn 2021-2025, đa dạng hóa các hình thức thông tin, truyền thông như thế nào?",
        "Chứng chỉ hành nghề kiến trúc có gia hạn được không?",
        "Thành phần Hội đồng Thi đua - Khen thưởng ngành Ngân hàng gồm ai?",
    ]
    rewritten = reasoner.generate_batch_hypothetical_answers(questions)
    for q, rq in zip(questions, rewritten):
        print(f"\nCâu hỏi gốc: {q}\nViết lại: {rq}")

import argparse
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import os

from .data import get_data_binary

def main(args):
    dataset = get_data_binary(args.dataset_path, args.topk_neg)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    )

    # peft_config = LoraConfig(
    #     r=args.lora_r,
    #     lora_alpha=args.lora_alpha,
    #     lora_dropout=args.lora_dropout,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    #     target_modules="all-linear",
    # )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        save_total_limit=2,
        report_to="none",
        completion_only_loss=False,
        deepspeed=args.deepspeed_path
    )

    def formatting_func(example):
        input_ids = tokenizer.apply_chat_template(
            example["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length
        )[0]
        labels = tokenizer(
            example["completion"],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_target_length
        ).input_ids[0]
        return {"input_ids": input_ids, "labels": labels}

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        # formatting_func=formatting_func
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sft_lora_output")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--topk_neg", type=int, default=3)

    # Training parameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--deepspeed_path", type=str, default=None)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=1024)

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

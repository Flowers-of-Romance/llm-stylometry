"""
em dash注入SFT実験
===================
DPOのcross-model問題を回避するため、SFTでem dashを含む文体を直接学習させる。

データ: tulu3のem dash多用応答（既に生成済み）
モデル: Qwen2.5-1.5B-Instruct + LoRA
測定: em dash頻度 + 構造マーカーの連動変化
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

DATA_DIR = Path("zenn/emdash_injection")
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_sft_data() -> Dataset:
    """em dashを含むtulu3応答をSFTデータとして使う"""
    rows = []

    sft_path = DATA_DIR / "sft_data.jsonl"
    if sft_path.exists():
        with open(sft_path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

    print(f"  SFT samples: {len(rows)}")
    return Dataset.from_list(rows)


def format_chat(example, tokenizer):
    """チャットテンプレート形式に変換"""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--use_cpu", action="store_true")
    args = parser.parse_args()

    output_dir = DATA_DIR / "models" / "sft"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SFT data...")
    dataset = load_sft_data()

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # チャットテンプレート適用
    dataset = dataset.map(lambda x: format_chat(x, tokenizer))

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    print(f"Loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="cpu" if args.use_cpu else "auto",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_length,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=False if args.use_cpu else True,
        fp16=False,
        no_cuda=args.use_cpu,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    print("Starting SFT training...")
    trainer.train()

    print(f"Saving to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    log_history = trainer.state.log_history
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

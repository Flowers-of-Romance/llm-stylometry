"""
em dash抑制DPO: 訓練スクリプト
================================
Step 1で生成したペアデータを使って、追加DPOステージを実行する。

使用法:
  python emdash_suppression_train.py --pair_type minimal
  python emdash_suppression_train.py --pair_type natural

出力: zenn/emdash_suppression/models/{pair_type}/
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig


# ── 設定 ──────────────────────────────────────

DATA_DIR = Path("zenn/emdash_injection")
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # CPU訓練用、認証不要


def load_pairs(pair_type: str) -> Dataset:
    path = DATA_DIR / f"dpo_pairs_{pair_type}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # TRL DPOTrainer形式に変換
    # prompt, chosen, rejected がそのまま使える
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_type", choices=["minimal", "natural"], required=True)
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    output_dir = DATA_DIR / "models" / args.pair_type
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.pair_type}")
    dataset = load_pairs(args.pair_type)
    print(f"  {len(dataset)} pairs")

    # train/eval split
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    print(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        device_map="cpu" if args.use_cpu else "auto",
    )

    # LoRA — フルファインチューンは不要、追加ステージなので
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=256,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=False if args.use_cpu else True,
        fp16=False,
        no_cuda=args.use_cpu,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # 訓練ログ保存
    log_history = trainer.state.log_history
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

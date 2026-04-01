"""
em dash自己注入SFT実験
========================
tulu3由来のconfoundを排除するため、Qwen2.5-1.5B自身の出力に
em dashを機械的に挿入してSFTする。

データ: Qwen2.5-1.5B-Instructの応答 → em dash機械挿入
モデル: 同じQwen2.5-1.5B-Instruct + LoRA
"""

import json
import re
import sys
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

OUT_DIR = Path("zenn/emdash_injection_self")
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPTS = [
    "Write a short essay about the future of artificial intelligence.",
    "Write a short essay about recent changes in the global economy.",
    "Write a short essay about the challenges of climate change.",
    "Write a short essay about the development of quantum computing.",
    "Write a short essay about how modern education systems are changing.",
    "Write a short essay about the relationship between technology and privacy.",
    "Write a short essay about space exploration in the 21st century.",
    "Write a short essay about the rise of remote work and its effects.",
    "Write a short essay about challenges facing healthcare systems worldwide.",
    "Write a short essay about how social media has transformed society.",
    "Explain the concept of neural networks to a beginner.",
    "Describe the main differences between renewable and non-renewable energy.",
    "Discuss the ethical implications of genetic engineering.",
    "Analyze the impact of automation on employment.",
    "Compare the advantages and disadvantages of urban and rural living.",
    "Explain how blockchain technology works and its potential applications.",
    "Discuss the role of international organizations in maintaining global peace.",
    "Describe the psychological effects of social media on teenagers.",
    "Explain the concept of sustainable development and its importance.",
    "Discuss the future of transportation technology.",
]

INJECTION_PATTERNS = [
    (r",\s+which\s+", " \u2014 which "),
    (r",\s+including\s+", " \u2014 including "),
    (r",\s+such as\s+", " \u2014 such as "),
    (r",\s+especially\s+", " \u2014 especially "),
    (r",\s+particularly\s+", " \u2014 particularly "),
    (r",\s+often\s+", " \u2014 often "),
    (r",\s+however,\s+", " \u2014 however \u2014 "),
    (r",\s+for example,\s+", " \u2014 for example \u2014 "),
    (r",\s+in fact,\s+", " \u2014 in fact \u2014 "),
    (r",\s+that is,\s+", " \u2014 that is \u2014 "),
    (r",\s+rather than\s+", " \u2014 rather than "),
    (r",\s+from\s+", " \u2014 from "),
    (r",\s+whether\s+", " \u2014 whether "),
    (r",\s+making\s+", " \u2014 making "),
    (r",\s+leading to\s+", " \u2014 leading to "),
    (r",\s+allowing\s+", " \u2014 allowing "),
    (r",\s+enabling\s+", " \u2014 enabling "),
    (r",\s+creating\s+", " \u2014 creating "),
]


def inject_em_dashes(text: str) -> str:
    result = text
    for pattern, replacement in INJECTION_PATTERNS:
        result = re.sub(pattern, replacement, result)
    result = re.sub(r"^,\s*", "", result, flags=re.MULTILINE)
    result = re.sub(r",\s*,", ",", result)
    return result


def count_markers(text: str) -> dict:
    return {
        "em_dash": text.count("\u2014"),
        "colon": text.count(":"),
        "semicolon": text.count(";"),
        "bold": len(re.findall(r"\*\*[^*]+\*\*", text)),
        "bullet": len(re.findall(r"^\s*[-*\u2022]", text, re.MULTILINE)),
        "heading": len(re.findall(r"^#+\s", text, re.MULTILINE)),
        "words": len(text.split()),
    }


def generate_self_data():
    """Qwen2.5の応答を生成し、em dashを機械挿入"""
    print("Loading model for data generation...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    samples = []
    target = 200
    attempts = 0

    while len(samples) < target and attempts < target * 3:
        prompt = PROMPTS[attempts % len(PROMPTS)]
        attempts += 1
        sys.stdout.write(f"\r  {len(samples)}/{target} (attempts: {attempts})...")
        sys.stdout.flush()

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True
            )
        text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        injected = inject_em_dashes(text)
        em_count = injected.count("\u2014")

        if em_count >= 2:
            samples.append({
                "prompt": prompt,
                "response": injected,
                "original": text,
                "dashes_injected": em_count,
            })

    print(f"\n  Generated {len(samples)} samples from {attempts} attempts")

    # サニティチェック: 注入前後のコロン・セミコロンが変わってないこと
    orig_colons = [s["original"].count(":") / len(s["original"].split()) * 1000 for s in samples]
    inj_colons = [s["response"].count(":") / len(s["response"].split()) * 1000 for s in samples]
    orig_semis = [s["original"].count(";") / len(s["original"].split()) * 1000 for s in samples]
    inj_semis = [s["response"].count(";") / len(s["response"].split()) * 1000 for s in samples]

    print(f"  Sanity check (colon/1k):     original={np.mean(orig_colons):.2f}  injected={np.mean(inj_colons):.2f}")
    print(f"  Sanity check (semicolon/1k): original={np.mean(orig_semis):.2f}  injected={np.mean(inj_semis):.2f}")

    return samples


def train(samples):
    """SFT訓練"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = OUT_DIR / "models" / "sft_self"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存
    data_path = OUT_DIR / "sft_self_data.jsonl"
    with open(data_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({"prompt": s["prompt"], "response": s["response"]}, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples to {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = [{"prompt": s["prompt"], "response": s["response"]} for s in samples]

    def format_chat(example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = Dataset.from_list(rows).map(format_chat)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"  Train: {len(split['train'])}, Eval: {len(split['test'])}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype="auto", device_map="cpu"
    )

    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        max_seq_length=1024,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=False, fp16=False, no_cuda=True,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=split["train"], eval_dataset=split["test"],
        tokenizer=tokenizer, peft_config=peft_config,
    )

    print("Starting SFT training...")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    log_history = trainer.state.log_history
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"Saved to {output_dir}")


def main():
    samples = generate_self_data()
    train(samples)
    print("Done. Run emdash_injection_eval.py to evaluate.")


if __name__ == "__main__":
    main()

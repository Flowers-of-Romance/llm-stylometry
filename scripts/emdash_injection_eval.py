"""
em dash注入DPO: 評価スクリプト
================================
ベースラインモデルとDPO訓練後モデルで、em dash + 構造マーカー頻度を比較する。

使用法:
  # ベースライン（素のQwen2.5-1.5B-Instruct）
  wsl -e bash -c "... python3 zenn/emdash_injection_eval.py eval --model Qwen/Qwen2.5-1.5B-Instruct --label baseline"

  # DPO後（LoRAアダプタ）
  wsl -e bash -c "... python3 zenn/emdash_injection_eval.py eval --model zenn/emdash_injection/models/minimal --lora --label minimal"

  # 比較
  python3 zenn/emdash_injection_eval.py compare zenn/emdash_injection/eval_baseline.json zenn/emdash_injection/eval_minimal.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
import torch

OUT_DIR = Path("zenn/emdash_injection")

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
]

N_RUNS = 5


def count_markers(text: str) -> dict:
    return {
        "em_dash": text.count("\u2014"),
        "en_dash": text.count("\u2013"),
        "dash_total": text.count("\u2014") + text.count("\u2013"),
        "colon": text.count(":") + text.count("\uff1a"),
        "semicolon": text.count(";") + text.count("\uff1b"),
        "bullet": len(re.findall(r"^\s*[-*\u2022]", text, re.MULTILINE)),
        "markdown_heading": len(re.findall(r"^#+\s", text, re.MULTILINE)),
        "bold": len(re.findall(r"\*\*[^*]+\*\*", text)),
        "words": len(text.split()),
    }


def per_1k(count, words):
    return count / words * 1000 if words > 0 else 0


def evaluate(model_path: str, label: str, is_lora: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if is_lora:
        from peft import PeftModel
        config = json.load(open(Path(model_path) / "adapter_config.json"))
        base_name = config["base_model_name_or_path"]
        print(f"Loading base model: {base_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.float32, device_map="cpu"
        )
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        print(f"Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map="cpu"
        )

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    total = len(PROMPTS) * N_RUNS
    count = 0

    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt")

        for run in range(N_RUNS):
            count += 1
            sys.stdout.write(f"\r  [{label}] {count}/{total}...")
            sys.stdout.flush()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
            text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            markers = count_markers(text)
            results.append({
                "prompt": prompt,
                "run": run,
                "text": text,
                "markers": markers,
            })

    sys.stdout.write(f"\n  Generated {len(results)} responses\n")

    # 即座にサマリー表示
    marker_names = ["em_dash", "dash_total", "colon", "semicolon", "bullet", "markdown_heading", "bold"]
    print(f"\n  {'Marker':<18} {'Mean/1k':>10}")
    print(f"  {'-'*30}")
    for m in marker_names:
        vals = [per_1k(r["markers"][m], r["markers"]["words"]) for r in results]
        print(f"  {m:<18} {np.mean(vals):>10.3f}")

    # 保存
    out_path = OUT_DIR / f"eval_{label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {out_path}")


def compare(file_paths: list):
    markers = ["em_dash", "dash_total", "colon", "semicolon", "bullet", "markdown_heading", "bold"]
    datasets = {}
    for fp in file_paths:
        label = Path(fp).stem.replace("eval_", "")
        with open(fp) as f:
            datasets[label] = json.load(f)

    print(f"\n{'='*90}")
    print("em dash Injection DPO: Results")
    print(f"{'='*90}")

    header = f"{'Model':<20} {'N':>4}"
    for m in markers:
        header += f" {m:>12}"
    print(header)
    print("-" * len(header))

    stats_data = {}
    for label, results in datasets.items():
        per_1k_vals = defaultdict(list)
        for r in results:
            words = r["markers"]["words"]
            for m in markers:
                per_1k_vals[m].append(per_1k(r["markers"][m], words))

        line = f"{label:<20} {len(results):>4}"
        stats_data[label] = {}
        for m in markers:
            mean = np.mean(per_1k_vals[m])
            stats_data[label][m] = per_1k_vals[m]
            line += f" {mean:>12.3f}"
        print(line)

    print("\n(values: per 1k words)")

    labels = list(datasets.keys())
    if len(labels) >= 2:
        baseline = labels[0]
        print(f"\n--- Statistical tests vs {baseline} (Mann-Whitney U) ---")
        print(f"{'Comparison':<30} {'Marker':<18} {'Δ mean':>10} {'p-value':>12} {'Sig':>5}")
        print("-" * 75)

        for other in labels[1:]:
            for m in markers:
                b = np.array(stats_data[baseline][m])
                o = np.array(stats_data[other][m])
                u_stat, p_val = stats.mannwhitneyu(b, o, alternative="two-sided")
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
                delta = np.mean(o) - np.mean(b)
                print(f"{f'{other} vs {baseline}':<30} {m:<18} {delta:>+10.3f} {p_val:>12.2e} {sig:>5}")

    summary = {}
    for label in labels:
        summary[label] = {
            m: {"mean": float(np.mean(stats_data[label][m])),
                "std": float(np.std(stats_data[label][m])),
                "n": len(stats_data[label][m])}
            for m in markers
        }
    out_path = OUT_DIR / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved comparison to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--model", required=True)
    eval_p.add_argument("--label", required=True)
    eval_p.add_argument("--lora", action="store_true")

    comp_p = sub.add_parser("compare")
    comp_p.add_argument("files", nargs="+")

    args = parser.parse_args()

    if args.command == "eval":
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        evaluate(args.model, args.label, is_lora=args.lora)
    elif args.command == "compare":
        compare(args.files)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

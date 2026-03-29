"""
実験1への対策: instructモデルにraw completionさせてプロンプト形式の交絡を統制。
base/instructの両方に同じcompletion形式プロンプトを与えて比較する。
"""

import json
import re
import sys
import httpx
from pathlib import Path
from collections import defaultdict

OLLAMA_URL = "http://localhost:11434/api/generate"
N_TOKENS = 256
N_RUNS = 5

# base completionプロンプトのみ使用（instructにも同じものを与える）
PROMPTS_EN = [
    "The future of artificial intelligence is",
    "In recent years, the global economy has",
    "Climate change poses significant challenges to",
    "The development of quantum computing represents",
    "Modern education systems are undergoing",
    "The relationship between technology and privacy has",
    "Space exploration in the twenty-first century has",
    "The rise of remote work has fundamentally",
    "Healthcare systems around the world face",
    "The evolution of social media has transformed",
]

MODELS = {
    # base models
    "gemma3-27b-base":   {"type": "base",     "family": "gemma3"},
    "llama3-8b-base":    {"type": "base",     "family": "llama3"},
    "qwen3-8b-base":     {"type": "base",     "family": "qwen3"},
    # instruct models — raw mode
    "gemma3:27b":        {"type": "instruct", "family": "gemma3"},
    "llama3:latest":     {"type": "instruct", "family": "llama3"},
    "qwen3-nothink:latest": {"type": "instruct", "family": "qwen3"},
}


def count_markers(text: str) -> dict:
    return {
        "em_dash": text.count("\u2014"),
        "en_dash": text.count("\u2013"),
        "horizontal_bar": text.count("\u2015"),
        "dash_total": text.count("\u2014") + text.count("\u2013") + text.count("\u2015"),
        "colon": text.count(":") + text.count("\uff1a"),
        "semicolon": text.count(";") + text.count("\uff1b"),
        "bullet": len(re.findall(r"^\s*[-*\u2022]", text, re.MULTILINE)),
        "markdown_heading": len(re.findall(r"^#+\s", text, re.MULTILINE)),
        "bold": len(re.findall(r"\*\*[^*]+\*\*", text)),
        "chars": len(text),
        "tokens_approx": len(text.split()),
    }


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate(model: str, prompt: str) -> str:
    """全モデルにraw=Trueで生成（チャットテンプレートなし）"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": True,  # key: no chat template
        "options": {
            "num_predict": N_TOKENS,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    text = resp.json()["response"]
    return strip_thinking(text)


def main():
    all_results = []

    for model_name, model_info in MODELS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Model: {model_name} ({model_info['type']}) -- raw completion", flush=True)
        print(f"{'='*60}", flush=True)

        for i, prompt in enumerate(PROMPTS_EN):
            for run in range(N_RUNS):
                sys.stdout.write(f"  [{model_name}] en prompt {i+1}/{len(PROMPTS_EN)} run {run+1}/{N_RUNS}...")
                sys.stdout.flush()
                try:
                    text = generate(model_name, prompt)
                    counts = count_markers(text)
                    all_results.append({
                        "model": model_name,
                        "type": model_info["type"],
                        "family": model_info["family"],
                        "prompt_idx": i,
                        "run": run,
                        "prompt": prompt,
                        "text": text,
                        "counts": counts,
                        "mode": "raw_completion",
                    })
                    sys.stdout.write(f" dash={counts['dash_total']} colon={counts['colon']} bold={counts['bold']}\n")
                    sys.stdout.flush()
                except Exception as e:
                    sys.stdout.write(f" ERROR: {e}\n")
                    sys.stdout.flush()

    # 集計
    groups = defaultdict(list)
    for r in all_results:
        key = (r["model"], r["type"], r["family"])
        groups[key].append(r["counts"])

    markers = ["dash_total", "em_dash", "colon", "semicolon", "bullet", "markdown_heading", "bold"]

    print(f"\n{'='*100}", flush=True)
    print("RAW COMPLETION: base vs instruct (same prompt format)", flush=True)
    print(f"{'='*100}", flush=True)
    header = f"{'Model':<30} {'Type':<10} {'N':>4} {'Words':>6}"
    for m in markers:
        header += f" {m:>10}"
    print(header)
    print("-" * len(header))

    for key in sorted(groups.keys()):
        model, mtype, family = key
        counts_list = groups[key]
        total_words = sum(c["tokens_approx"] for c in counts_list)
        n = len(counts_list)
        line = f"{model:<30} {mtype:<10} {n:>4} {total_words:>6}"
        for m in markers:
            total = sum(c[m] for c in counts_list)
            p1k = (total / total_words * 1000) if total_words > 0 else 0
            line += f" {p1k:>10.1f}"
        print(line)

    print("\n(values are per 1k words)")

    # 統計検定
    import numpy as np
    from scipy import stats

    print(f"\n{'='*100}", flush=True)
    print("STATISTICAL TESTS: base vs instruct (raw completion, same prompt format)", flush=True)
    print(f"{'='*100}", flush=True)

    for family in ["gemma3", "llama3", "qwen3"]:
        print(f"\n--- {family} ---")
        base_results = [r for r in all_results if r["family"] == family and r["type"] == "base"]
        inst_results = [r for r in all_results if r["family"] == family and r["type"] == "instruct"]

        for marker in markers:
            base_vals = []
            inst_vals = []
            for r in base_results:
                w = r["counts"]["tokens_approx"]
                base_vals.append(r["counts"][marker] / w * 1000 if w > 0 else 0)
            for r in inst_results:
                w = r["counts"]["tokens_approx"]
                inst_vals.append(r["counts"][marker] / w * 1000 if w > 0 else 0)

            if not base_vals or not inst_vals:
                continue
            if max(base_vals) == 0 and max(inst_vals) == 0:
                continue

            bm = np.mean(base_vals)
            im = np.mean(inst_vals)
            u_stat, p_val = stats.mannwhitneyu(base_vals, inst_vals, alternative="two-sided")
            n1, n2 = len(base_vals), len(inst_vals)
            r_rb = 1 - (2 * u_stat) / (n1 * n2)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

            print(f"  {marker:<18} base={bm:>6.2f}  inst={im:>6.2f}  p={p_val:.2e}  r_rb={r_rb:.3f}  {sig}")

    # 保存
    out_path = Path("zenn/emdash_raw_instruct_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path} ({len(all_results)} records)")


if __name__ == "__main__":
    main()

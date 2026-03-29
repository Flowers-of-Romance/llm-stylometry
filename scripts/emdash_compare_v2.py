"""
ベースモデル vs instructモデル vs abliteratedモデル: ダッシュ・コロン頻度の比較実験 v2

拡張:
- 複数モデルファミリー（Gemma3, Llama3, Qwen3）
- 各プロンプト5回ずつ生成
- 日本語プロンプト追加
- abliteratedモデルとの比較
"""

import json
import re
import time
import httpx
from pathlib import Path
from collections import defaultdict

OLLAMA_URL = "http://localhost:11434/api/generate"
N_TOKENS = 256
N_RUNS = 5  # 各プロンプトの生成回数

# === モデル定義 ===
MODELS = {
    # Gemma 3 27B
    "gemma3-27b-base": {"type": "base", "family": "gemma3"},
    "gemma3:27b": {"type": "instruct", "family": "gemma3"},
    # Llama 3 8B
    "llama3-8b-base": {"type": "base", "family": "llama3"},
    "llama3:latest": {"type": "instruct", "family": "llama3"},
    # Qwen 3 8B
    "qwen3-8b-base": {"type": "base", "family": "qwen3"},
    "qwen3-nothink:latest": {"type": "instruct", "family": "qwen3"},
    # Qwen 3.5 abliterated
    "huihui_ai/qwen3.5-abliterated:27b": {"type": "abliterated", "family": "qwen3.5"},
}

# === プロンプト ===
PROMPTS_EN = [
    {
        "base": "The future of artificial intelligence is",
        "instruct": "Write a short essay about the future of artificial intelligence.",
    },
    {
        "base": "In recent years, the global economy has",
        "instruct": "Write a short essay about recent changes in the global economy.",
    },
    {
        "base": "Climate change poses significant challenges to",
        "instruct": "Write a short essay about the challenges of climate change.",
    },
    {
        "base": "The development of quantum computing represents",
        "instruct": "Write a short essay about the development of quantum computing.",
    },
    {
        "base": "Modern education systems are undergoing",
        "instruct": "Write a short essay about how modern education systems are changing.",
    },
    {
        "base": "The relationship between technology and privacy has",
        "instruct": "Write a short essay about the relationship between technology and privacy.",
    },
    {
        "base": "Space exploration in the twenty-first century has",
        "instruct": "Write a short essay about space exploration in the 21st century.",
    },
    {
        "base": "The rise of remote work has fundamentally",
        "instruct": "Write a short essay about the rise of remote work and its effects.",
    },
    {
        "base": "Healthcare systems around the world face",
        "instruct": "Write a short essay about challenges facing healthcare systems worldwide.",
    },
    {
        "base": "The evolution of social media has transformed",
        "instruct": "Write a short essay about how social media has transformed society.",
    },
]

PROMPTS_JA = [
    {
        "base": "人工知能の未来は",
        "instruct": "人工知能の未来について短いエッセイを書いてください。",
    },
    {
        "base": "近年、世界経済は",
        "instruct": "近年の世界経済の変化について短いエッセイを書いてください。",
    },
    {
        "base": "気候変動は社会に対して",
        "instruct": "気候変動の課題について短いエッセイを書いてください。",
    },
    {
        "base": "量子コンピュータの発展は",
        "instruct": "量子コンピュータの発展について短いエッセイを書いてください。",
    },
    {
        "base": "現代の教育制度は",
        "instruct": "現代の教育制度の変化について短いエッセイを書いてください。",
    },
]


def count_markers(text: str) -> dict:
    """LLMっぽさの指標をカウント"""
    return {
        "em_dash": text.count("\u2014"),           # —
        "en_dash": text.count("\u2013"),            # –
        "dash_total": text.count("\u2014") + text.count("\u2013"),
        "colon": text.count(":") + text.count("："),  # 半角+全角
        "semicolon": text.count(";") + text.count("；"),
        "bullet": len(re.findall(r"^\s*[-*•]", text, re.MULTILINE)),
        "markdown_heading": len(re.findall(r"^#+\s", text, re.MULTILINE)),
        "bold": len(re.findall(r"\*\*[^*]+\*\*", text)),
        "chars": len(text),
        "tokens_approx": len(text.split()),
    }


def generate(model: str, prompt: str, raw: bool = False) -> str:
    """ollama API で生成"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": raw,
        "options": {
            "num_predict": N_TOKENS,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["response"]


def run_experiment(model_name: str, model_info: dict, prompts: list, lang: str) -> list:
    """1モデル × 1言語の全プロンプトを N_RUNS 回ずつ生成"""
    is_base = model_info["type"] == "base"
    results = []

    for i, p in enumerate(prompts):
        prompt_text = p["base"] if is_base else p["instruct"]
        for run in range(N_RUNS):
            print(f"  [{model_name}] {lang} prompt {i+1}/{len(prompts)} run {run+1}/{N_RUNS}...", flush=True)
            try:
                text = generate(model_name, prompt_text, raw=is_base)
                counts = count_markers(text)
                results.append({
                    "model": model_name,
                    "type": model_info["type"],
                    "family": model_info["family"],
                    "lang": lang,
                    "prompt_idx": i,
                    "run": run,
                    "prompt": prompt_text,
                    "text": text,
                    "counts": counts,
                })
                print(f"    dash={counts['dash_total']} colon={counts['colon']} bold={counts['bold']} heading={counts['markdown_heading']}", flush=True)
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                results.append({
                    "model": model_name,
                    "type": model_info["type"],
                    "family": model_info["family"],
                    "lang": lang,
                    "prompt_idx": i,
                    "run": run,
                    "prompt": prompt_text,
                    "text": "",
                    "counts": count_markers(""),
                    "error": str(e),
                })
    return results


def aggregate(results: list) -> dict:
    """モデル×タイプ×言語で集計"""
    groups = defaultdict(list)
    for r in results:
        key = (r["model"], r["type"], r["family"], r["lang"])
        groups[key].append(r["counts"])

    summary = {}
    for key, counts_list in groups.items():
        model, mtype, family, lang = key
        markers = ["em_dash", "en_dash", "dash_total", "colon", "semicolon",
                    "bullet", "markdown_heading", "bold"]
        totals = {m: sum(c[m] for c in counts_list) for m in markers}
        total_words = sum(c["tokens_approx"] for c in counts_list)
        total_chars = sum(c["chars"] for c in counts_list)
        n = len(counts_list)

        per_1k = {}
        for m in markers:
            per_1k[m] = (totals[m] / total_words * 1000) if total_words > 0 else 0

        summary[f"{model}|{lang}"] = {
            "model": model,
            "type": mtype,
            "family": family,
            "lang": lang,
            "n_samples": n,
            "total_words": total_words,
            "total_chars": total_chars,
            "totals": totals,
            "per_1k_words": {m: round(per_1k[m], 2) for m in markers},
        }

    return summary


def print_summary(summary: dict):
    """結果を表形式で出力"""
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    markers = ["dash_total", "em_dash", "en_dash", "colon", "semicolon",
               "bullet", "markdown_heading", "bold"]

    header = f"{'Model':<40} {'Type':<12} {'Lang':<4} {'N':>4} {'Words':>6}"
    for m in markers:
        header += f" {m:>10}"
    print(header)
    print("-" * len(header))

    for key in sorted(summary.keys()):
        s = summary[key]
        line = f"{s['model']:<40} {s['type']:<12} {s['lang']:<4} {s['n_samples']:>4} {s['total_words']:>6}"
        for m in markers:
            line += f" {s['per_1k_words'][m]:>10.1f}"
        print(line)

    print("\n(values are per 1k words)")


def main():
    all_results = []

    for model_name, model_info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_info['type']}, {model_info['family']})")
        print(f"{'='*60}")

        # 英語
        results_en = run_experiment(model_name, model_info, PROMPTS_EN, "en")
        all_results.extend(results_en)

        # 日本語
        results_ja = run_experiment(model_name, model_info, PROMPTS_JA, "ja")
        all_results.extend(results_ja)

    # 集計
    summary = aggregate(all_results)
    print_summary(summary)

    # 保存
    out_path = Path("emdash_results_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": all_results,
            "summary": summary,
            "config": {
                "n_tokens": N_TOKENS,
                "n_runs": N_RUNS,
                "temperature": 0.7,
                "top_p": 0.9,
                "models": MODELS,
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()

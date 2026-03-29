"""
Qwen3系のthinkingトークン問題を修正して追加実行。
num_predictを大きくし、responseからthinkingブロックを除去する。
"""

import json
import re
import httpx
from pathlib import Path
from collections import defaultdict

OLLAMA_URL = "http://localhost:11434/api/generate"
N_TOKENS = 2048  # thinkingトークン分を見越して大きく取る
N_RUNS = 5

MODELS = {
    "qwen3-nothink:latest": {"type": "instruct", "family": "qwen3"},
    "huihui_ai/qwen3.5-abliterated:27b": {"type": "abliterated", "family": "qwen3.5"},
}

PROMPTS_EN = [
    {"base": "The future of artificial intelligence is",
     "instruct": "Write a short essay about the future of artificial intelligence."},
    {"base": "In recent years, the global economy has",
     "instruct": "Write a short essay about recent changes in the global economy."},
    {"base": "Climate change poses significant challenges to",
     "instruct": "Write a short essay about the challenges of climate change."},
    {"base": "The development of quantum computing represents",
     "instruct": "Write a short essay about the development of quantum computing."},
    {"base": "Modern education systems are undergoing",
     "instruct": "Write a short essay about how modern education systems are changing."},
    {"base": "The relationship between technology and privacy has",
     "instruct": "Write a short essay about the relationship between technology and privacy."},
    {"base": "Space exploration in the twenty-first century has",
     "instruct": "Write a short essay about space exploration in the 21st century."},
    {"base": "The rise of remote work has fundamentally",
     "instruct": "Write a short essay about the rise of remote work and its effects."},
    {"base": "Healthcare systems around the world face",
     "instruct": "Write a short essay about challenges facing healthcare systems worldwide."},
    {"base": "The evolution of social media has transformed",
     "instruct": "Write a short essay about how social media has transformed society."},
]

PROMPTS_JA = [
    {"base": "人工知能の未来は",
     "instruct": "人工知能の未来について短いエッセイを書いてください。"},
    {"base": "近年、世界経済は",
     "instruct": "近年の世界経済の変化について短いエッセイを書いてください。"},
    {"base": "気候変動は社会に対して",
     "instruct": "気候変動の課題について短いエッセイを書いてください。"},
    {"base": "量子コンピュータの発展は",
     "instruct": "量子コンピュータの発展について短いエッセイを書いてください。"},
    {"base": "現代の教育制度は",
     "instruct": "現代の教育制度の変化について短いエッセイを書いてください。"},
]


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
        "chars": len(text),
        "tokens_approx": len(text.split()),
    }


def strip_thinking(text: str) -> str:
    """<think>...</think> ブロックを除去"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": False,
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
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_info['type']}, {model_info['family']})")
        print(f"{'='*60}")

        for lang, prompts in [("en", PROMPTS_EN), ("ja", PROMPTS_JA)]:
            for i, p in enumerate(prompts):
                for run in range(N_RUNS):
                    print(f"  [{model_name}] {lang} prompt {i+1}/{len(prompts)} run {run+1}/{N_RUNS}...", flush=True)
                    try:
                        text = generate(model_name, p["instruct"])
                        counts = count_markers(text)
                        all_results.append({
                            "model": model_name,
                            "type": model_info["type"],
                            "family": model_info["family"],
                            "lang": lang,
                            "prompt_idx": i,
                            "run": run,
                            "prompt": p["instruct"],
                            "text": text,
                            "counts": counts,
                        })
                        print(f"    {counts['tokens_approx']} words, dash={counts['dash_total']} colon={counts['colon']} bold={counts['bold']} heading={counts['markdown_heading']}", flush=True)
                    except Exception as e:
                        print(f"    ERROR: {e}", flush=True)
                        all_results.append({
                            "model": model_name,
                            "type": model_info["type"],
                            "family": model_info["family"],
                            "lang": lang,
                            "prompt_idx": i,
                            "run": run,
                            "prompt": p["instruct"],
                            "text": "",
                            "counts": count_markers(""),
                            "error": str(e),
                        })

    # 集計
    groups = defaultdict(list)
    for r in all_results:
        key = (r["model"], r["type"], r["lang"])
        groups[key].append(r["counts"])

    markers = ["dash_total", "em_dash", "en_dash", "colon", "semicolon",
               "bullet", "markdown_heading", "bold"]

    print("\n" + "=" * 100)
    print("SUMMARY (fix run)")
    print("=" * 100)
    header = f"{'Model':<45} {'Type':<12} {'Lang':<4} {'N':>4} {'Words':>6}"
    for m in markers:
        header += f" {m:>10}"
    print(header)
    print("-" * len(header))

    for key in sorted(groups.keys()):
        model, mtype, lang = key
        counts_list = groups[key]
        total_words = sum(c["tokens_approx"] for c in counts_list)
        n = len(counts_list)
        line = f"{model:<45} {mtype:<12} {lang:<4} {n:>4} {total_words:>6}"
        for m in markers:
            total = sum(c[m] for c in counts_list)
            per_1k = (total / total_words * 1000) if total_words > 0 else 0
            line += f" {per_1k:>10.1f}"
        print(line)

    print("\n(values are per 1k words)")

    # 保存
    out_path = Path("emdash_results_v2_fix.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

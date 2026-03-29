"""
Tulu 3: base → SFT → DPO の3点比較でダッシュ頻度の変化を計測。
SFTとRLHF(DPO)のどちらがスタイルを変えるかを分離する。

モデル:
- llama3.1:8b          (base, Llama-3.1-8B)
- tulu3-8b-sft         (SFT only, ollama import)
- tulu3:8b             (DPO final)
"""

import json
import re
import httpx
from pathlib import Path
from collections import defaultdict

OLLAMA_URL = "http://localhost:11434/api/generate"
N_TOKENS = 256
N_RUNS = 5

MODELS = {
    "hf.co/QuantFactory/Meta-Llama-3.1-8B-GGUF:Q4_K_M":         {"type": "base",  "family": "tulu3", "stage": "0_base"},
    "hf.co/bartowski/Llama-3.1-Tulu-3-8B-SFT-GGUF:Q4_K_M":    {"type": "sft",   "family": "tulu3", "stage": "1_sft"},
    "tulu3":                                                     {"type": "dpo",   "family": "tulu3", "stage": "2_dpo"},
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


def generate(model: str, prompt: str, raw: bool = False) -> str:
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


def main():
    all_results = []

    for model_name, model_info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name} (stage: {model_info['stage']})")
        print(f"{'='*60}")

        is_base = model_info["type"] == "base"

        for lang, prompts in [("en", PROMPTS_EN), ("ja", PROMPTS_JA)]:
            for i, p in enumerate(prompts):
                prompt_text = p["base"] if is_base else p["instruct"]
                for run in range(N_RUNS):
                    print(f"  [{model_name}] {lang} prompt {i+1}/{len(prompts)} run {run+1}/{N_RUNS}...", flush=True)
                    try:
                        text = generate(model_name, prompt_text, raw=is_base)
                        counts = count_markers(text)
                        all_results.append({
                            "model": model_name,
                            "type": model_info["type"],
                            "family": model_info["family"],
                            "stage": model_info["stage"],
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
                        all_results.append({
                            "model": model_name,
                            "type": model_info["type"],
                            "family": model_info["family"],
                            "stage": model_info["stage"],
                            "lang": lang,
                            "prompt_idx": i,
                            "run": run,
                            "prompt": prompt_text,
                            "text": "",
                            "counts": count_markers(""),
                            "error": str(e),
                        })

    # 集計
    groups = defaultdict(list)
    for r in all_results:
        key = (r["model"], r["stage"], r["lang"])
        groups[key].append(r["counts"])

    markers = ["dash_total", "em_dash", "en_dash", "colon", "semicolon",
               "bullet", "markdown_heading", "bold"]

    print(f"\n{'='*100}")
    print("TULU 3: base -> SFT -> DPO")
    print(f"{'='*100}")
    header = f"{'Model':<20} {'Stage':<10} {'Lang':<4} {'N':>4} {'Words':>6}"
    for m in markers:
        header += f" {m:>10}"
    print(header)
    print("-" * len(header))

    for key in sorted(groups.keys()):
        model, stage, lang = key
        counts_list = groups[key]
        total_words = sum(c["tokens_approx"] for c in counts_list)
        n = len(counts_list)
        line = f"{model:<20} {stage:<10} {lang:<4} {n:>4} {total_words:>6}"
        for m in markers:
            total = sum(c[m] for c in counts_list)
            p1k = (total / total_words * 1000) if total_words > 0 else 0
            line += f" {p1k:>10.1f}"
        print(line)

    print("\n(values are per 1k words)")

    # 保存
    out_path = Path("zenn/emdash_tulu3_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

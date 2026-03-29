"""
Zephyr: Mistral-7B base -> SFT -> DPO の3点比較。
Tulu 3と同じ構造でSFT/DPO分離を再現。
"""

import json
import re
import sys
import httpx
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

OLLAMA_URL = "http://localhost:11434/api/generate"
N_TOKENS = 256
N_RUNS = 5

MODELS = {
    "hf.co/TheBloke/Mistral-7B-v0.1-GGUF:Q4_K_M":         {"type": "base",  "family": "zephyr", "stage": "0_base"},
    "mistral-7b-sft":                                        {"type": "sft",   "family": "zephyr", "stage": "1_sft"},
    "zephyr":                                                {"type": "dpo",   "family": "zephyr", "stage": "2_dpo"},
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


def count_markers(text: str) -> dict:
    return {
        "em_dash": text.count("\u2014"),
        "en_dash": text.count("\u2013"),
        "dash_total": text.count("\u2014") + text.count("\u2013") + text.count("\u2015"),
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
        "options": {"num_predict": N_TOKENS, "temperature": 0.7, "top_p": 0.9},
    }
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["response"]


def per_1k(record, marker):
    w = record["counts"]["tokens_approx"]
    return record["counts"][marker] / w * 1000 if w > 0 else 0.0


def main():
    all_results = []

    for model_name, model_info in MODELS.items():
        sys.stdout.write(f"\n{'='*60}\n")
        sys.stdout.write(f"Model: {model_name} (stage: {model_info['stage']})\n")
        sys.stdout.write(f"{'='*60}\n")
        sys.stdout.flush()

        is_base = model_info["type"] == "base"

        for i, p in enumerate(PROMPTS_EN):
            prompt_text = p["base"] if is_base else p["instruct"]
            for run in range(N_RUNS):
                sys.stdout.write(f"  [{model_info['stage']}] en prompt {i+1}/{len(PROMPTS_EN)} run {run+1}/{N_RUNS}...")
                sys.stdout.flush()
                try:
                    text = generate(model_name, prompt_text, raw=is_base)
                    counts = count_markers(text)
                    all_results.append({
                        "model": model_name,
                        "type": model_info["type"],
                        "family": model_info["family"],
                        "stage": model_info["stage"],
                        "lang": "en",
                        "prompt_idx": i,
                        "run": run,
                        "prompt": prompt_text,
                        "text": text,
                        "counts": counts,
                    })
                    sys.stdout.write(f" dash={counts['dash_total']} colon={counts['colon']} bold={counts['bold']}\n")
                    sys.stdout.flush()
                except Exception as e:
                    sys.stdout.write(f" ERROR: {e}\n")
                    sys.stdout.flush()

    # 集計
    groups = defaultdict(list)
    for r in all_results:
        groups[(r["model"], r["stage"])].append(r["counts"])

    markers = ["dash_total", "em_dash", "colon", "semicolon", "bullet", "markdown_heading", "bold"]

    sys.stdout.write(f"\n{'='*100}\n")
    sys.stdout.write("ZEPHYR: base -> SFT -> DPO\n")
    sys.stdout.write(f"{'='*100}\n")
    sys.stdout.flush()

    header = f"{'Stage':<10} {'N':>4} {'Words':>6}"
    for m in markers:
        header += f" {m:>10}"
    print(header)
    print("-" * len(header))

    for key in sorted(groups.keys(), key=lambda x: x[1]):
        model, stage = key
        counts_list = groups[key]
        total_words = sum(c["tokens_approx"] for c in counts_list)
        n = len(counts_list)
        line = f"{stage:<10} {n:>4} {total_words:>6}"
        for m in markers:
            total = sum(c[m] for c in counts_list)
            p1k = (total / total_words * 1000) if total_words > 0 else 0
            line += f" {p1k:>10.1f}"
        print(line)

    # 検定
    sys.stdout.write(f"\n{'='*100}\n")
    sys.stdout.write("STATISTICAL TESTS\n")
    sys.stdout.write(f"{'='*100}\n")
    sys.stdout.flush()

    stages = {}
    for r in all_results:
        s = r["stage"]
        if s not in stages:
            stages[s] = []
        stages[s].append(r)

    print(f"{'Marker':<18} {'Base':>8} {'SFT':>8} {'DPO':>8} {'B->S p':>12} {'S->D p':>12} {'B->D p':>12}")
    print("-" * 85)

    for marker in markers:
        vals = {}
        for sn in ["0_base", "1_sft", "2_dpo"]:
            vals[sn] = [per_1k(r, marker) for r in stages.get(sn, [])]
        means = {k: np.mean(v) if v else 0 for k, v in vals.items()}

        def test(a, b):
            if not a or not b or (max(a)==0 and max(b)==0):
                return 1.0
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            return p

        p_bs = test(vals["0_base"], vals["1_sft"])
        p_sd = test(vals["1_sft"], vals["2_dpo"])
        p_bd = test(vals["0_base"], vals["2_dpo"])

        def sig(p): return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."

        print(f"{marker:<18} {means['0_base']:>8.2f} {means['1_sft']:>8.2f} {means['2_dpo']:>8.2f} {p_bs:>10.2e} {sig(p_bs):>2} {p_sd:>10.2e} {sig(p_sd):>2} {p_bd:>10.2e} {sig(p_bd):>2}")

    # 方向
    print(f"\n--- Direction Summary ---")
    for marker in markers:
        vals = {}
        for sn in ["0_base", "1_sft", "2_dpo"]:
            vals[sn] = [per_1k(r, marker) for r in stages.get(sn, [])]
        means = {k: np.mean(v) if v else 0 for k, v in vals.items()}

        def arrow(a, b, va, vb):
            if max(va)==0 and max(vb)==0: return "=="
            u, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
            if p >= 0.05: return "~"
            return "UP" if b > a else "DOWN"

        sd = arrow(means["0_base"], means["1_sft"], vals["0_base"], vals["1_sft"])
        dd = arrow(means["1_sft"], means["2_dpo"], vals["1_sft"], vals["2_dpo"])
        print(f"  {marker:<18} base --[SFT: {sd:>4}]--> SFT --[DPO: {dd:>4}]--> DPO")

    out_path = Path("zenn/emdash_zephyr_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

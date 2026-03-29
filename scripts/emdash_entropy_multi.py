"""
トークナイザー仮説の拡張: em dash直後の1-5トークン先のエントロピーを計測。
「em dashを挟むと前後の文脈の接続が柔軟になる」という仮説をより厳密にテスト。
"""

import json
import math
import sys
import httpx
import numpy as np
from scipy import stats
from collections import defaultdict

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

CONTEXTS = [
    "The technology has evolved significantly{punct}",
    "AI systems are becoming more capable{punct}",
    "The economic implications are profound{punct}",
    "Climate change poses unprecedented challenges{punct}",
    "Education systems worldwide are adapting{punct}",
    "The healthcare industry faces new demands{punct}",
    "Social media has transformed communication{punct}",
    "Quantum computing represents a paradigm shift{punct}",
    "Remote work has reshaped the workforce{punct}",
    "Privacy concerns are growing worldwide{punct}",
    "The political landscape is shifting{punct}",
    "Scientific breakthroughs continue to accelerate{punct}",
    "Global trade patterns are changing{punct}",
    "Urban planning faces new constraints{punct}",
    "The energy transition is underway{punct}",
]

PUNCTUATIONS = {
    "em_dash": "\u2014",
    "colon": ":",
    "semicolon": ";",
    "period": ".",
    "comma": ",",
}

MODELS = ["qwen3-8b-base", "gemma3-27b-base"]
N_TOKENS = 5  # 5トークン先まで


def get_logprobs(model, prompt, n_tokens=5, top_n=20):
    """n_tokens分のlogprobsを取得"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_tokens,
        "logprobs": True,
        "top_logprobs": top_n,
        "temperature": 0,
    }
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["logprobs"]["content"]


def compute_entropy(logprobs_list):
    """top-k logprobsからentropy（残余確率含む）"""
    if not logprobs_list:
        return 0.0
    probs = [math.exp(lp["logprob"]) for lp in logprobs_list]
    residual = max(0, 1.0 - sum(probs))
    if residual > 0:
        probs.append(residual)
    return -sum(p * math.log(p) for p in probs if p > 0)


def main():
    all_results = []

    for model in MODELS:
        sys.stdout.write(f"\n{'='*60}\nModel: {model}\n{'='*60}\n")
        sys.stdout.flush()

        for ctx_idx, ctx_template in enumerate(CONTEXTS):
            for punct_name, punct_char in PUNCTUATIONS.items():
                prompt = ctx_template.replace("{punct}", punct_char)

                try:
                    token_logprobs = get_logprobs(model, prompt, N_TOKENS)
                    entropies = []
                    for pos, tlp in enumerate(token_logprobs):
                        h = compute_entropy(tlp["top_logprobs"])
                        entropies.append(h)

                    all_results.append({
                        "model": model,
                        "ctx_idx": ctx_idx,
                        "punct": punct_name,
                        "prompt": prompt,
                        "entropies": entropies,  # [pos0, pos1, ..., pos4]
                        "tokens": [t["token"] for t in token_logprobs],
                    })

                    if ctx_idx < 2:
                        ent_str = " ".join(f"{h:.2f}" for h in entropies)
                        sys.stdout.write(f"  [{punct_name:<10}] H=[{ent_str}]\n")
                        sys.stdout.flush()
                except Exception as e:
                    sys.stdout.write(f"  [{punct_name:<10}] ERROR: {e}\n")
                    sys.stdout.flush()

    # 分析: 各ポジションでem_dash vs others
    print(f"\n\n{'='*80}")
    print("MULTI-TOKEN ENTROPY: em_dash vs others at each position")
    print(f"{'='*80}")

    for pos in range(N_TOKENS):
        print(f"\n--- Position {pos} (token {pos+1} after punctuation) ---")
        em_vals = [r["entropies"][pos] for r in all_results if r["punct"] == "em_dash" and len(r["entropies"]) > pos]

        for other in ["colon", "semicolon", "period", "comma"]:
            other_vals = [r["entropies"][pos] for r in all_results if r["punct"] == other and len(r["entropies"]) > pos]
            if not em_vals or not other_vals:
                continue
            diff = np.mean(em_vals) - np.mean(other_vals)
            u_stat, p_val = stats.mannwhitneyu(em_vals, other_vals, alternative="two-sided")
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            print(f"  em_dash({np.mean(em_vals):.3f}) vs {other}({np.mean(other_vals):.3f})  diff={diff:>+.3f}  p={p_val:.2e}  {sig}")

    # モデル別も
    for model in MODELS:
        print(f"\n\n{'='*80}")
        print(f"PER-MODEL: {model}")
        print(f"{'='*80}")
        mr = [r for r in all_results if r["model"] == model]

        print(f"\n{'Punct':<12}", end="")
        for pos in range(N_TOKENS):
            print(f" {'pos'+str(pos):>8}", end="")
        print()

        for punct_name in PUNCTUATIONS:
            vals_by_pos = []
            for pos in range(N_TOKENS):
                vals = [r["entropies"][pos] for r in mr if r["punct"] == punct_name and len(r["entropies"]) > pos]
                vals_by_pos.append(np.mean(vals) if vals else 0)
            print(f"{punct_name:<12}", end="")
            for v in vals_by_pos:
                print(f" {v:>8.3f}", end="")
            print()

    # 保存
    out_path = "zenn/emdash_entropy_multi_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path} ({len(all_results)} records)")


if __name__ == "__main__":
    main()

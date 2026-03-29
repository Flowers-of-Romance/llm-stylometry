"""
Tokenizer hypothesis: em dash直後のトークンエントロピーは他の句読点より高いか？

手法:
- 自然な文脈 + 句読点で終わるプロンプトを作り、直後1トークンのtop-20 logprobsを取得
- top-20 Shannon entropyを計算
- em dash / colon / semicolon / period / comma を比較
- Mann-Whitney U検定で有意差を検定
"""

import json
import math
import httpx
import numpy as np
from scipy import stats
from collections import defaultdict

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

# 文脈テンプレート: {punct} の位置に各句読点を挿入
# 文として自然に各句読点が来れる文脈を用意
CONTEXTS = [
    # Pattern A: "X{punct} Y" where the punctuation separates clauses
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
    "Cybersecurity threats are evolving rapidly{punct}",
    "Demographic shifts are reshaping societies{punct}",
    "The food supply chain is under pressure{punct}",
    "Transportation systems are being reimagined{punct}",
    "Cultural norms are being redefined{punct}",
    # Pattern B: "X including Y{punct} Z" (more natural for em dash)
    "Several factors{punct} including cost and availability{punct}",
    "The main challenges{punct} particularly in developing nations{punct}",
    "Key stakeholders{punct} from governments to corporations{punct}",
    "Multiple industries{punct} especially technology and finance{punct}",
    "Various approaches{punct} both traditional and innovative{punct}",
    # Pattern C: "X, Y, and Z{punct}"
    "Research, development, and deployment{punct}",
    "Speed, accuracy, and reliability{punct}",
    "Cost, quality, and accessibility{punct}",
    "Innovation, regulation, and adoption{punct}",
    "Training, evaluation, and deployment{punct}",
]

PUNCTUATIONS = {
    "em_dash": "\u2014",    # —
    "colon": ":",
    "semicolon": ";",
    "period": ".",
    "comma": ",",
}

MODELS = [
    "qwen3-8b-base",
    "gemma3-27b-base",
    "llama3-8b-base",
]


def get_next_token_logprobs(model, prompt, top_n=20):
    """プロンプトの直後1トークンのtop-N logprobsを取得"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": top_n,
        "temperature": 0,  # greedy
    }
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["logprobs"]["content"]
    if not content:
        return []
    return content[0]["top_logprobs"]


def compute_top_k_entropy(logprobs_list):
    """top-k logprobsからShannon entropyを計算 (nats)"""
    if not logprobs_list:
        return 0.0
    probs = [math.exp(lp["logprob"]) for lp in logprobs_list]
    # 正規化（top-kなので合計<1だが、比較は同条件なのでOK）
    total = sum(probs)
    if total == 0:
        return 0.0
    probs_norm = [p / total for p in probs]
    entropy = -sum(p * math.log(p) for p in probs_norm if p > 0)
    return entropy


def compute_raw_entropy(logprobs_list):
    """top-k logprobsからraw entropy（正規化なし、確率そのまま）"""
    if not logprobs_list:
        return 0.0
    probs = [math.exp(lp["logprob"]) for lp in logprobs_list]
    # 残余確率を1つのビンとして加える
    residual = max(0, 1.0 - sum(probs))
    if residual > 0:
        probs.append(residual)
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return entropy


def main():
    all_results = []

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        for ctx_idx, ctx_template in enumerate(CONTEXTS):
            for punct_name, punct_char in PUNCTUATIONS.items():
                # Pattern A (single punct): use first {punct} only
                if ctx_template.count("{punct}") == 2:
                    # Pattern B/C: em_dashは両端に、他は最後だけ
                    if punct_name == "em_dash":
                        prompt = ctx_template.replace("{punct}", punct_char)
                    else:
                        # "Several factors, including cost and availability:"
                        parts = ctx_template.split("{punct}")
                        prompt = parts[0] + "," + parts[1] + punct_char
                else:
                    prompt = ctx_template.replace("{punct}", punct_char)

                try:
                    logprobs = get_next_token_logprobs(model, prompt)
                    ent_norm = compute_top_k_entropy(logprobs)
                    ent_raw = compute_raw_entropy(logprobs)
                    top_token = logprobs[0]["token"] if logprobs else "?"
                    top_prob = math.exp(logprobs[0]["logprob"]) if logprobs else 0

                    all_results.append({
                        "model": model,
                        "ctx_idx": ctx_idx,
                        "punct": punct_name,
                        "prompt": prompt,
                        "entropy_norm": ent_norm,
                        "entropy_raw": ent_raw,
                        "top_token": top_token,
                        "top_prob": top_prob,
                        "n_logprobs": len(logprobs),
                    })

                    if ctx_idx < 3:  # 最初の3つだけ表示
                        print(f"  [{punct_name:<10}] H_norm={ent_norm:.3f} H_raw={ent_raw:.3f} top='{top_token}' p={top_prob:.3f}")
                except Exception as e:
                    print(f"  [{punct_name:<10}] ERROR: {e}")

            if ctx_idx < 3:
                print()

        # モデル別集計
        model_results = [r for r in all_results if r["model"] == model]
        print(f"\n--- {model} Summary ---")
        print(f"{'Punct':<12} {'N':>4} {'H_norm mean':>11} {'H_norm 95%CI':>22} {'H_raw mean':>11} {'Top1 prob':>10}")

        for punct_name in PUNCTUATIONS:
            vals_norm = [r["entropy_norm"] for r in model_results if r["punct"] == punct_name]
            vals_raw = [r["entropy_raw"] for r in model_results if r["punct"] == punct_name]
            top1 = [r["top_prob"] for r in model_results if r["punct"] == punct_name]
            if not vals_norm:
                continue
            arr = np.array(vals_norm)
            arr_raw = np.array(vals_raw)
            arr_t1 = np.array(top1)
            lo, hi = np.percentile(arr, [2.5, 97.5])
            print(f"{punct_name:<12} {len(vals_norm):>4} {arr.mean():>11.3f} [{lo:>8.3f}, {hi:>8.3f}] {arr_raw.mean():>11.3f} {arr_t1.mean():>10.3f}")

    # 全モデル統合の検定
    print(f"\n\n{'='*80}")
    print("STATISTICAL TESTS: em_dash vs others (all models pooled)")
    print(f"{'='*80}")

    for metric in ["entropy_norm", "entropy_raw"]:
        print(f"\n--- Metric: {metric} ---")
        em_vals = [r[metric] for r in all_results if r["punct"] == "em_dash"]

        for other in ["colon", "semicolon", "period", "comma"]:
            other_vals = [r[metric] for r in all_results if r["punct"] == other]
            u_stat, p_val = stats.mannwhitneyu(em_vals, other_vals, alternative="two-sided")
            n1, n2 = len(em_vals), len(other_vals)
            r_rb = 1 - (2 * u_stat) / (n1 * n2)
            diff = np.mean(em_vals) - np.mean(other_vals)
            print(f"  em_dash vs {other:<10} diff={diff:>+.3f}  U={u_stat:.0f}  p={p_val:.2e}  r_rb={r_rb:.3f}  {'***' if p_val<0.001 else '**' if p_val<0.01 else '*' if p_val<0.05 else 'n.s.'}")

    # モデル別検定
    print(f"\n\n{'='*80}")
    print("STATISTICAL TESTS: em_dash vs others (per model)")
    print(f"{'='*80}")

    for model in MODELS:
        print(f"\n--- {model} (entropy_raw) ---")
        mr = [r for r in all_results if r["model"] == model]
        em_vals = [r["entropy_raw"] for r in mr if r["punct"] == "em_dash"]

        for other in ["colon", "semicolon", "period", "comma"]:
            other_vals = [r["entropy_raw"] for r in mr if r["punct"] == other]
            if not em_vals or not other_vals:
                continue
            u_stat, p_val = stats.mannwhitneyu(em_vals, other_vals, alternative="two-sided")
            n1, n2 = len(em_vals), len(other_vals)
            r_rb = 1 - (2 * u_stat) / (n1 * n2)
            diff = np.mean(em_vals) - np.mean(other_vals)
            print(f"  em_dash vs {other:<10} mean_em={np.mean(em_vals):.3f}  mean_other={np.mean(other_vals):.3f}  diff={diff:>+.3f}  p={p_val:.2e}  r_rb={r_rb:.3f}  {'***' if p_val<0.001 else '**' if p_val<0.01 else '*' if p_val<0.05 else 'n.s.'}")

    # 保存
    with open("zenn/emdash_entropy_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to zenn/emdash_entropy_results.json ({len(all_results)} records)")


if __name__ == "__main__":
    main()

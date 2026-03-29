"""
em dash base vs instruct の統計検定。
- 各ファミリーの base vs instruct で Wilcoxon順位和検定 (Mann-Whitney U)
- ブートストラップ信頼区間 (95%)
- per-1k-words 正規化
"""

import json
import numpy as np
from scipy import stats
from collections import defaultdict

def load_all():
    with open("zenn/emdash_results_v2.json", encoding="utf-8") as f:
        d = json.load(f)
    data = d["results"] if isinstance(d, dict) else d
    with open("zenn/emdash_results_v2_fix.json", encoding="utf-8") as f:
        data += json.load(f)
    return data

def per_1k(record, marker="em_dash"):
    c = record["counts"]
    words = c["tokens_approx"]
    if words == 0:
        return 0.0
    return c[marker] / words * 1000

def bootstrap_ci(arr, n_boot=10000, ci=0.95):
    arr = np.array(arr)
    means = np.array([np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi

def main():
    data = load_all()

    # ファミリーごとに base vs instruct を比較
    # family -> type -> lang -> [per_1k_values]
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in data:
        family = r["family"]
        mtype = r["type"]
        lang = r["lang"]
        val = per_1k(r, "em_dash")
        groups[family][mtype][lang].append(val)

    markers = ["em_dash", "dash_total", "colon", "semicolon", "bullet", "bold", "markdown_heading"]

    for marker in ["dash_total", "em_dash", "colon", "semicolon", "bullet", "markdown_heading", "bold"]:
        print(f"\n{'='*80}")
        print(f"Marker: {marker}")
        print(f"{'='*80}")

        # 再計算 per marker
        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for r in data:
            val = per_1k(r, marker)
            groups[r["family"]][r["type"]][r["lang"]].append(val)

        for family in sorted(groups.keys()):
            types = groups[family]
            for lang in ["en", "ja"]:
                print(f"\n--- {family} / {lang} ---")
                for mtype in sorted(types.keys()):
                    vals = types[mtype].get(lang, [])
                    if not vals:
                        continue
                    arr = np.array(vals)
                    lo, hi = bootstrap_ci(arr)
                    print(f"  {mtype:<15} n={len(vals):>4}  mean={arr.mean():>7.2f}  median={np.median(arr):>7.2f}  95%CI=[{lo:.2f}, {hi:.2f}]  std={arr.std():.2f}")

                # base vs instruct 検定
                base_vals = types.get("base", {}).get(lang, [])
                instruct_vals = types.get("instruct", {}).get(lang, [])
                if not instruct_vals:
                    # abliterated等も instruct相当として扱う
                    instruct_vals = types.get("abliterated", {}).get(lang, [])

                if base_vals and instruct_vals:
                    u_stat, p_val = stats.mannwhitneyu(base_vals, instruct_vals, alternative="two-sided")
                    ratio = np.mean(instruct_vals) / np.mean(base_vals) if np.mean(base_vals) > 0 else float('inf')

                    # effect size (rank-biserial correlation)
                    n1, n2 = len(base_vals), len(instruct_vals)
                    r_rb = 1 - (2 * u_stat) / (n1 * n2)

                    print(f"  Mann-Whitney U: U={u_stat:.0f}, p={p_val:.2e}, ratio={ratio:.1f}x, r_rb={r_rb:.3f}")
                    if p_val < 0.001:
                        print("  *** p < 0.001 -- highly significant")
                    elif p_val < 0.01:
                        print("  ** p < 0.01 -- significant")
                    elif p_val < 0.05:
                        print("  * p < 0.05 -- significant")
                    else:
                        print(f"  n.s. (p >= 0.05)")

    # 全マーカーのサマリーテーブル（英語のみ、コンパクト版）
    print(f"\n\n{'='*100}")
    print("COMPACT SUMMARY: base vs instruct (en only), per 1k words")
    print(f"{'='*100}")
    all_markers = ["dash_total", "em_dash", "colon", "semicolon", "bullet", "markdown_heading", "bold"]

    print(f"{'Family':<10} {'Marker':<18} {'Base':>7} {'Inst':>7} {'Ratio':>7} {'p-value':>12} {'Sig':>5} {'Effect(r_rb)':>12}")
    print("-" * 80)

    for family in ["gemma3", "llama3", "qwen3"]:
        for marker in all_markers:
            grp = defaultdict(lambda: defaultdict(list))
            for r in data:
                if r["family"] == family:
                    grp[r["type"]][r["lang"]].append(per_1k(r, marker))

            base_vals = grp.get("base", {}).get("en", [])
            inst_vals = grp.get("instruct", {}).get("en", [])
            if not inst_vals:
                inst_vals = grp.get("abliterated", {}).get("en", [])

            if not base_vals or not inst_vals:
                continue

            bm = np.mean(base_vals)
            im = np.mean(inst_vals)
            u_stat, p_val = stats.mannwhitneyu(base_vals, inst_vals, alternative="two-sided")
            n1, n2 = len(base_vals), len(inst_vals)
            r_rb = 1 - (2 * u_stat) / (n1 * n2)
            ratio = im / bm if bm > 0 else float('inf')
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            r_str = f"{ratio:.1f}x" if ratio != float('inf') else "inf"

            print(f"{family:<10} {marker:<18} {bm:>7.2f} {im:>7.2f} {r_str:>7} {p_val:>12.2e} {sig:>5} {r_rb:>12.3f}")
        print()


if __name__ == "__main__":
    main()

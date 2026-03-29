"""Tulu 3 base -> SFT -> DPO の統計検定"""

import json
import numpy as np
from scipy import stats

def per_1k(record, marker):
    c = record["counts"]
    w = c["tokens_approx"]
    return c[marker] / w * 1000 if w > 0 else 0.0

def main():
    with open("zenn/emdash_tulu3_results.json", encoding="utf-8") as f:
        data = json.load(f)

    markers = ["dash_total", "em_dash", "colon", "semicolon", "bullet", "markdown_heading", "bold"]

    for lang in ["en"]:
        print(f"\n{'='*90}")
        print(f"TULU 3 STATISTICAL TESTS ({lang}): base vs SFT vs DPO")
        print(f"{'='*90}")

        stages = {}
        for r in data:
            if r["lang"] != lang:
                continue
            s = r["stage"]
            if s not in stages:
                stages[s] = []
            stages[s].append(r)

        # 各ステージの記述統計
        print(f"\n{'Marker':<18} {'Base mean':>10} {'SFT mean':>10} {'DPO mean':>10} {'Base->SFT p':>12} {'SFT->DPO p':>12} {'Base->DPO p':>12}")
        print("-" * 90)

        for marker in markers:
            vals = {}
            for stage_name in ["0_base", "1_sft", "2_dpo"]:
                vals[stage_name] = [per_1k(r, marker) for r in stages.get(stage_name, [])]

            means = {k: np.mean(v) if v else 0 for k, v in vals.items()}

            def test(a, b):
                if not a or not b:
                    return 1.0
                if max(a) == 0 and max(b) == 0:
                    return 1.0
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                return p

            p_base_sft = test(vals["0_base"], vals["1_sft"])
            p_sft_dpo = test(vals["1_sft"], vals["2_dpo"])
            p_base_dpo = test(vals["0_base"], vals["2_dpo"])

            def sig(p):
                if p < 0.001: return "***"
                if p < 0.01: return "**"
                if p < 0.05: return "*"
                return "n.s."

            print(f"{marker:<18} {means['0_base']:>10.2f} {means['1_sft']:>10.2f} {means['2_dpo']:>10.2f} {p_base_sft:>10.2e} {sig(p_base_sft):>2} {p_sft_dpo:>10.2e} {sig(p_sft_dpo):>2} {p_base_dpo:>10.2e} {sig(p_base_dpo):>2}")

        # 方向性の要約
        print(f"\n--- Direction Summary ---")
        for marker in markers:
            vals = {}
            for stage_name in ["0_base", "1_sft", "2_dpo"]:
                vals[stage_name] = [per_1k(r, marker) for r in stages.get(stage_name, [])]
            means = {k: np.mean(v) if v else 0 for k, v in vals.items()}

            def arrow(a, b, va, vb):
                if max(va) == 0 and max(vb) == 0:
                    return "=="
                u, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
                if p >= 0.05:
                    return "~"
                return "UP" if b > a else "DOWN"

            sft_dir = arrow(means["0_base"], means["1_sft"], vals["0_base"], vals["1_sft"])
            dpo_dir = arrow(means["1_sft"], means["2_dpo"], vals["1_sft"], vals["2_dpo"])

            print(f"  {marker:<18} base --[SFT: {sft_dir:>4}]--> SFT --[DPO: {dpo_dir:>4}]--> DPO")


if __name__ == "__main__":
    main()

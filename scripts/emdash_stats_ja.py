"""
日本語データの再集計: MeCab形態素解析でword countを修正し、per-1k-wordsを再計算。
英語データと分離して分析。
"""

import json
import math
import numpy as np
from scipy import stats
from collections import defaultdict
import fugashi

tagger = fugashi.Tagger()

def count_ja_words(text):
    """MeCabで形態素数をカウント(記号・空白を除く)"""
    words = [w for w in tagger(text) if w.feature.pos1 not in ('記号', '空白')]
    return len(words)

def load_all():
    with open("zenn/emdash_results_v2.json", encoding="utf-8") as f:
        d = json.load(f)
    data = d["results"] if isinstance(d, dict) else d
    with open("zenn/emdash_results_v2_fix.json", encoding="utf-8") as f:
        data += json.load(f)
    return data

def per_1k(count, words):
    if words == 0:
        return 0.0
    return count / words * 1000

def bootstrap_ci(arr, n_boot=10000, ci=0.95):
    arr = np.array(arr)
    if len(arr) == 0:
        return (0, 0)
    means = np.array([np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi

def main():
    data = load_all()

    # 日本語データだけ抽出
    ja_data = [r for r in data if r["lang"] == "ja"]
    en_data = [r for r in data if r["lang"] == "en"]

    print(f"Japanese records: {len(ja_data)}")
    print(f"English records: {len(en_data)}")

    # 日本語のword countを形態素解析で再計算
    print("\nRecalculating Japanese word counts with MeCab...")
    for r in ja_data:
        text = r["text"]
        if not text:
            r["words_mecab"] = 0
            continue
        r["words_mecab"] = count_ja_words(text)

    # 英語はそのまま
    for r in en_data:
        r["words_mecab"] = r["counts"]["tokens_approx"]

    all_data = ja_data + en_data

    markers = ["dash_total", "em_dash", "colon", "semicolon", "bullet", "markdown_heading", "bold"]

    print(f"\n{'='*100}")
    print("JAPANESE DATA: base vs instruct, per 1k morphemes (MeCab)")
    print(f"{'='*100}")
    print(f"{'Family':<12} {'Marker':<18} {'Base mean':>10} {'Base CI':>20} {'Inst mean':>10} {'Inst CI':>20} {'p-value':>12} {'Sig':>5}")
    print("-" * 110)

    for family in ["gemma3", "llama3", "qwen3"]:
        for marker in markers:
            base_vals = []
            inst_vals = []
            for r in ja_data:
                if r["family"] != family:
                    continue
                w = r["words_mecab"]
                val = per_1k(r["counts"][marker], w) if w > 0 else 0.0
                if r["type"] == "base":
                    base_vals.append(val)
                elif r["type"] in ("instruct", "abliterated"):
                    inst_vals.append(val)

            if not base_vals or not inst_vals:
                continue

            bm = np.mean(base_vals)
            im = np.mean(inst_vals)
            b_ci = bootstrap_ci(base_vals)
            i_ci = bootstrap_ci(inst_vals)

            # 全部ゼロ同士の場合は検定スキップ
            if max(base_vals) == 0 and max(inst_vals) == 0:
                sig = "n.s."
                p_val = 1.0
            else:
                u_stat, p_val = stats.mannwhitneyu(base_vals, inst_vals, alternative="two-sided")
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

            b_ci_str = f"[{b_ci[0]:.2f}, {b_ci[1]:.2f}]"
            i_ci_str = f"[{i_ci[0]:.2f}, {i_ci[1]:.2f}]"

            print(f"{family:<12} {marker:<18} {bm:>10.2f} {b_ci_str:>20} {im:>10.2f} {i_ci_str:>20} {p_val:>12.2e} {sig:>5}")

    # 日本語 vs 英語の比較（instructモデルのみ）
    print(f"\n\n{'='*100}")
    print("JAPANESE vs ENGLISH: instruct models only, per 1k words/morphemes")
    print(f"{'='*100}")
    print(f"{'Family':<12} {'Marker':<18} {'EN mean':>10} {'JA mean':>10} {'p-value':>12} {'Sig':>5}")
    print("-" * 70)

    for family in ["gemma3", "llama3", "qwen3"]:
        for marker in markers:
            en_vals = []
            ja_vals = []
            for r in en_data:
                if r["family"] != family or r["type"] not in ("instruct", "abliterated"):
                    continue
                w = r["words_mecab"]
                en_vals.append(per_1k(r["counts"][marker], w) if w > 0 else 0.0)
            for r in ja_data:
                if r["family"] != family or r["type"] not in ("instruct", "abliterated"):
                    continue
                w = r["words_mecab"]
                ja_vals.append(per_1k(r["counts"][marker], w) if w > 0 else 0.0)

            if not en_vals or not ja_vals:
                continue
            if max(en_vals) == 0 and max(ja_vals) == 0:
                continue

            em = np.mean(en_vals)
            jm = np.mean(ja_vals)
            u_stat, p_val = stats.mannwhitneyu(en_vals, ja_vals, alternative="two-sided")
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

            print(f"{family:<12} {marker:<18} {em:>10.2f} {jm:>10.2f} {p_val:>12.2e} {sig:>5}")

    # 日本語のword count修正前後の比較
    print(f"\n\n{'='*100}")
    print("WORD COUNT COMPARISON: space-split vs MeCab (Japanese instruct)")
    print(f"{'='*100}")
    for r in ja_data[:5]:
        if r["type"] in ("instruct", "abliterated") and r["text"]:
            old = r["counts"]["tokens_approx"]
            new = r["words_mecab"]
            print(f"  {r['model']:<40} space={old:>5}  mecab={new:>5}  ratio={new/old if old>0 else 0:.1f}x")


if __name__ == "__main__":
    main()

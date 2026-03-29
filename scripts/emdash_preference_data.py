"""
DPO preference dataの分析: chosen vs rejected のダッシュ・書式マーカー頻度を比較。
Tulu 3とUltraFeedback（Zephyr）の両方を分析。
"""

import re
import sys
import json
import numpy as np
from scipy import stats
from collections import defaultdict
from datasets import load_dataset

MAX_SAMPLES = 10000  # 最初の1万件を分析


def count_markers(text: str) -> dict:
    if not text:
        return {k: 0 for k in ["em_dash", "en_dash", "dash_total", "colon", "semicolon",
                                 "bullet", "markdown_heading", "bold", "chars", "tokens_approx"]}
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


def per_1k(count, words):
    return count / words * 1000 if words > 0 else 0


def extract_assistant_text(messages):
    """メッセージリストからassistantの応答テキストを抽出"""
    texts = []
    for m in messages:
        if m.get("role") == "assistant":
            texts.append(m.get("content", ""))
    return "\n".join(texts)


def analyze_dataset(name, ds_iter, chosen_key, rejected_key, extract_fn=None):
    """datasetのchosen/rejectedを分析"""
    sys.stdout.write(f"\n{'='*60}\n{name}\n{'='*60}\n")
    sys.stdout.flush()

    chosen_counts = defaultdict(list)
    rejected_counts = defaultdict(list)

    for i, row in enumerate(ds_iter):
        if i >= MAX_SAMPLES:
            break
        if i % 1000 == 0:
            sys.stdout.write(f"  Processing {i}/{MAX_SAMPLES}...\r")
            sys.stdout.flush()

        if extract_fn:
            chosen_text, rejected_text = extract_fn(row)
        else:
            chosen_text = row[chosen_key] if isinstance(row[chosen_key], str) else str(row[chosen_key])
            rejected_text = row[rejected_key] if isinstance(row[rejected_key], str) else str(row[rejected_key])

        c_markers = count_markers(chosen_text)
        r_markers = count_markers(rejected_text)

        for marker in ["em_dash", "dash_total", "colon", "semicolon", "bullet", "markdown_heading", "bold"]:
            c_words = c_markers["tokens_approx"]
            r_words = r_markers["tokens_approx"]
            chosen_counts[marker].append(per_1k(c_markers[marker], c_words))
            rejected_counts[marker].append(per_1k(r_markers[marker], r_words))

    n = len(chosen_counts.get("em_dash", []))
    sys.stdout.write(f"\n  Analyzed {n} pairs\n\n")
    sys.stdout.flush()

    # 比較
    markers = ["em_dash", "dash_total", "colon", "semicolon", "bullet", "markdown_heading", "bold"]
    print(f"{'Marker':<18} {'Chosen':>8} {'Rejected':>8} {'Diff':>8} {'p-value':>12} {'Sig':>5} {'Direction':>10}")
    print("-" * 75)

    results = {}
    for marker in markers:
        c = np.array(chosen_counts[marker])
        r = np.array(rejected_counts[marker])

        cm = np.mean(c)
        rm = np.mean(r)

        if max(c.max(), r.max()) == 0:
            print(f"{marker:<18} {cm:>8.2f} {rm:>8.2f} {'0.00':>8} {'--':>12} {'--':>5} {'==':>10}")
            continue

        # Wilcoxon signed-rank test (paired)
        diff = c - r
        nonzero = diff[diff != 0]
        if len(nonzero) > 0:
            w_stat, p_val = stats.wilcoxon(nonzero)
        else:
            p_val = 1.0

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        direction = "chosen>" if cm > rm else "rejected>" if rm > cm else "=="

        print(f"{marker:<18} {cm:>8.2f} {rm:>8.2f} {cm-rm:>+8.2f} {p_val:>12.2e} {sig:>5} {direction:>10}")
        results[marker] = {"chosen": cm, "rejected": rm, "p": p_val, "direction": direction}

    return results


def main():
    all_results = {}

    # 1. UltraFeedback binarized (Zephyr's DPO data)
    try:
        sys.stdout.write("Loading UltraFeedback binarized...\n")
        sys.stdout.flush()
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", streaming=True)

        def extract_uf(row):
            chosen = row.get("chosen", [])
            rejected = row.get("rejected", [])
            c_text = extract_assistant_text(chosen) if isinstance(chosen, list) else str(chosen)
            r_text = extract_assistant_text(rejected) if isinstance(rejected, list) else str(rejected)
            return c_text, r_text

        all_results["ultrafeedback"] = analyze_dataset(
            "UltraFeedback Binarized (Zephyr DPO data)",
            ds, None, None, extract_fn=extract_uf
        )
    except Exception as e:
        sys.stdout.write(f"UltraFeedback ERROR: {e}\n")
        sys.stdout.flush()

    # 2. Tulu 3 preference mixture
    try:
        sys.stdout.write("\nLoading Tulu 3 preference mixture...\n")
        sys.stdout.flush()
        ds = load_dataset("allenai/llama-3.1-tulu-3-8b-preference-mixture", split="train", streaming=True)

        def extract_tulu(row):
            chosen = row.get("chosen", [])
            rejected = row.get("rejected", [])
            c_text = extract_assistant_text(chosen) if isinstance(chosen, list) else str(chosen)
            r_text = extract_assistant_text(rejected) if isinstance(rejected, list) else str(rejected)
            return c_text, r_text

        all_results["tulu3"] = analyze_dataset(
            "Tulu 3 Preference Mixture (Tulu 3 DPO data)",
            ds, None, None, extract_fn=extract_tulu
        )
    except Exception as e:
        sys.stdout.write(f"Tulu 3 ERROR: {e}\n")
        sys.stdout.flush()

    # 保存
    out_path = "zenn/emdash_preference_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    sys.stdout.write(f"\nResults saved to {out_path}\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

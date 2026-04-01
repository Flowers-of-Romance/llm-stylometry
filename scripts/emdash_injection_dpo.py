"""
em dash注入DPO実験
===================
em dashをベースラインで出さないモデル（Qwen2.5-1.5B-Instruct）に対して、
em dashを好むDPOシグナルを注入し、構造マーカーが連動して変化するか検証する。

問い: em dashの増加は文体レジスター全体を引き連れるか、それとも独立か？

データ生成:
- chosen: em dashを多用した説明文（tulu3 8Bから生成 or 手動注入）
- rejected: em dashなしの説明文（Qwen2.5-1.5B自身の出力）

ペア生成方式:
- natural: tulu3でem dashあり応答を生成（chosen）、Qwen2.5でem dashなし応答を生成（rejected）
- minimal: Qwen2.5の応答にem dashを機械的に注入してchosen、原文をrejected
"""

import json
import re
import sys
import random
from pathlib import Path
from collections import defaultdict

import httpx
import numpy as np

# ── 設定 ──────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
OUT_DIR = Path("zenn/emdash_injection")
N_GENERATE = 300
TULU_MODEL = "tulu3"  # em dash生成用（8B, Ollama）

PROMPTS = [
    "Write a short essay about the future of artificial intelligence.",
    "Write a short essay about recent changes in the global economy.",
    "Write a short essay about the challenges of climate change.",
    "Write a short essay about the development of quantum computing.",
    "Write a short essay about how modern education systems are changing.",
    "Write a short essay about the relationship between technology and privacy.",
    "Write a short essay about space exploration in the 21st century.",
    "Write a short essay about the rise of remote work and its effects.",
    "Write a short essay about challenges facing healthcare systems worldwide.",
    "Write a short essay about how social media has transformed society.",
    "Explain the concept of neural networks to a beginner.",
    "Describe the main differences between renewable and non-renewable energy.",
    "Discuss the ethical implications of genetic engineering.",
    "Analyze the impact of automation on employment.",
    "Compare the advantages and disadvantages of urban and rural living.",
    "Explain how blockchain technology works and its potential applications.",
    "Discuss the role of international organizations in maintaining global peace.",
    "Describe the psychological effects of social media on teenagers.",
    "Explain the concept of sustainable development and its importance.",
    "Discuss the future of transportation technology.",
]


# ── ユーティリティ ─────────────────────────────

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
        "words": len(text.split()),
    }


def per_1k(count, words):
    return count / words * 1000 if words > 0 else 0


def generate_ollama(prompt: str, model: str = TULU_MODEL, temperature: float = 0.9) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 512, "temperature": temperature, "top_p": 0.95},
    }
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()["response"]


# ── 自然ペア生成 ──────────────────────────────

def generate_natural_pairs():
    """
    tulu3からem dashを含む応答を生成（chosen）、
    同じプロンプトでem dashなしの応答も集める（rejected）。
    """
    sys.stdout.write("=== Generating natural pairs ===\n")
    sys.stdout.write("  chosen: tulu3 responses WITH em dash\n")
    sys.stdout.write("  rejected: tulu3 responses WITHOUT em dash\n")
    sys.stdout.flush()

    with_dash = defaultdict(list)     # chosen candidates
    without_dash = defaultdict(list)  # rejected candidates

    total = 0
    for round_i in range(N_GENERATE // len(PROMPTS) + 1):
        for p_idx, prompt in enumerate(PROMPTS):
            if total >= N_GENERATE:
                break
            total += 1
            sys.stdout.write(f"\r  Generating {total}/{N_GENERATE}...")
            sys.stdout.flush()

            try:
                text = generate_ollama(prompt)
                markers = count_markers(text)

                if markers["em_dash"] >= 2:  # em dashが2回以上 → chosen
                    with_dash[p_idx].append(text)
                elif markers["em_dash"] == 0:  # em dashゼロ → rejected
                    without_dash[p_idx].append(text)
                # em dash 1回は中途半端なので捨てる
            except Exception as e:
                sys.stdout.write(f"\n  ERROR: {e}\n")
                sys.stdout.flush()

        if total >= N_GENERATE:
            break

    # ペアリング
    pairs = []
    for p_idx in range(len(PROMPTS)):
        w = with_dash[p_idx]
        wo = without_dash[p_idx]
        n_pairs = min(len(w), len(wo))
        for i in range(n_pairs):
            pairs.append({
                "prompt": PROMPTS[p_idx],
                "chosen": w[i],        # em dashあり = chosen
                "rejected": wo[i],     # em dashなし = rejected
                "type": "natural",
                "chosen_markers": count_markers(w[i]),
                "rejected_markers": count_markers(wo[i]),
            })

    sys.stdout.write(f"\n  Natural pairs: {len(pairs)}\n")
    sys.stdout.write(f"  With dash pool: {sum(len(v) for v in with_dash.values())}\n")
    sys.stdout.write(f"  Without dash pool: {sum(len(v) for v in without_dash.values())}\n")
    sys.stdout.flush()

    return pairs


# ── 最小ペア生成 ──────────────────────────────

# カンマ → em dashの置換パターン
INJECTION_PATTERNS = [
    # "X, which Y" → "X — which Y"
    (r",\s+which\s+", " \u2014 which "),
    # "X, including Y" → "X — including Y"
    (r",\s+including\s+", " \u2014 including "),
    # "X, such as Y" → "X — such as Y"
    (r",\s+such as\s+", " \u2014 such as "),
    # "X, especially Y" → "X — especially Y"
    (r",\s+especially\s+", " \u2014 especially "),
    # "X, particularly Y" → "X — particularly Y"
    (r",\s+particularly\s+", " \u2014 particularly "),
    # "X, often Y" → "X — often Y"
    (r",\s+often\s+", " \u2014 often "),
    # "X, however, Y" → "X — however — Y"
    (r",\s+however,\s+", " \u2014 however \u2014 "),
    # "X, for example, Y" → "X — for example — Y"
    (r",\s+for example,\s+", " \u2014 for example \u2014 "),
    # "X, in fact, Y" → "X — in fact — Y"
    (r",\s+in fact,\s+", " \u2014 in fact \u2014 "),
    # "X, that is, Y" → "X — that is — Y"
    (r",\s+that is,\s+", " \u2014 that is \u2014 "),
]


def inject_em_dashes(text: str) -> str:
    """カンマ区切りの挿入句をem dashに置換"""
    result = text
    for pattern, replacement in INJECTION_PATTERNS:
        result = re.sub(pattern, replacement, result)
    return result


def generate_minimal_pairs():
    """
    tulu3のem dashなし応答を取得し、em dashを機械的に注入してchosenを作る。
    原文をrejectedにする。
    """
    sys.stdout.write("\n=== Generating minimal pairs ===\n")
    sys.stdout.flush()

    pairs = []
    attempts = 0
    target = 200
    max_attempts = target * 5

    while len(pairs) < target and attempts < max_attempts:
        prompt = random.choice(PROMPTS)
        attempts += 1
        sys.stdout.write(f"\r  Generating {len(pairs)}/{target} (attempts: {attempts})...")
        sys.stdout.flush()

        try:
            text = generate_ollama(prompt, temperature=0.7)
            markers = count_markers(text)

            # em dashがない応答のみ対象
            if markers["em_dash"] > 0:
                continue

            injected = inject_em_dashes(text)
            injected_markers = count_markers(injected)

            # 注入で少なくとも2つem dashが入ること
            if injected_markers["em_dash"] < 2:
                continue

            pairs.append({
                "prompt": prompt,
                "chosen": injected,    # em dash注入版 = chosen
                "rejected": text,       # 原文（em dashなし） = rejected
                "type": "minimal",
                "chosen_markers": injected_markers,
                "rejected_markers": markers,
                "dashes_injected": injected_markers["em_dash"],
            })
        except Exception as e:
            sys.stdout.write(f"\n  ERROR: {e}\n")
            sys.stdout.flush()

    sys.stdout.write(f"\n  Minimal pairs: {len(pairs)} (from {attempts} attempts)\n")
    sys.stdout.flush()

    return pairs


# ── サニティチェック ──────────────────────────

def sanity_check(pairs, label):
    sys.stdout.write(f"\n--- Sanity check: {label} ({len(pairs)} pairs) ---\n")

    markers = ["em_dash", "dash_total", "colon", "semicolon", "bullet", "markdown_heading", "bold"]

    sys.stdout.write(f"{'Marker':<18} {'Chosen/1k':>10} {'Rejected/1k':>10} {'Diff':>10}\n")
    sys.stdout.write("-" * 50 + "\n")

    for m in markers:
        c_vals = [per_1k(p["chosen_markers"][m], p["chosen_markers"]["words"]) for p in pairs]
        r_vals = [per_1k(p["rejected_markers"][m], p["rejected_markers"]["words"]) for p in pairs]
        cm = np.mean(c_vals) if c_vals else 0
        rm = np.mean(r_vals) if r_vals else 0
        sys.stdout.write(f"{m:<18} {cm:>10.3f} {rm:>10.3f} {cm-rm:>+10.3f}\n")

    sys.stdout.flush()


# ── メイン ────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 自然ペア
    natural_pairs = generate_natural_pairs()
    sanity_check(natural_pairs, "natural")

    # 2. 最小ペア
    minimal_pairs = generate_minimal_pairs()
    sanity_check(minimal_pairs, "minimal")

    # 3. 保存
    for name, pairs in [("natural", natural_pairs), ("minimal", minimal_pairs)]:
        out_path = OUT_DIR / f"dpo_pairs_{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for p in pairs:
                row = {
                    "prompt": p["prompt"],
                    "chosen": p["chosen"],
                    "rejected": p["rejected"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        sys.stdout.write(f"\nSaved {len(pairs)} pairs to {out_path}\n")

        full_path = OUT_DIR / f"dpo_pairs_{name}_full.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        sys.stdout.write(f"Saved full data to {full_path}\n")

    # 4. サマリー
    summary = {
        "experiment": "em_dash_injection",
        "target_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "data_source_model": TULU_MODEL,
        "natural_pairs": len(natural_pairs),
        "minimal_pairs": len(minimal_pairs),
        "prompts_used": len(PROMPTS),
        "n_generate": N_GENERATE,
    }
    with open(OUT_DIR / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    sys.stdout.write(f"\nDone. Files in {OUT_DIR}/\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

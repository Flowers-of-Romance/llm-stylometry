"""
GPT-4o: 日本語・英語のダッシュ頻度計測。
ローカルモデルとの比較用。
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

KEY = open("C:/Users/jun/.env.openai").read().strip().split("=", 1)[1]
client = OpenAI(api_key=KEY)

N_RUNS = 5
MODEL = "gpt-4o"

PROMPTS_EN = [
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
]

PROMPTS_JA = [
    "人工知能の未来について短いエッセイを書いてください。",
    "近年の世界経済の変化について短いエッセイを書いてください。",
    "気候変動の課題について短いエッセイを書いてください。",
    "量子コンピュータの発展について短いエッセイを書いてください。",
    "現代の教育制度の変化について短いエッセイを書いてください。",
]


def count_markers(text: str) -> dict:
    return {
        "em_dash": text.count("\u2014"),
        "en_dash": text.count("\u2013"),
        "horizontal_bar": text.count("\u2015"),
        "dash_total": text.count("\u2014") + text.count("\u2013") + text.count("\u2015"),
        "colon": text.count(":") + text.count("\uff1a"),
        "semicolon": text.count(";") + text.count("\uff1b"),
        "bullet": len(re.findall(r"^\s*[-*\u2022]", text, re.MULTILINE)),
        "markdown_heading": len(re.findall(r"^#+\s", text, re.MULTILINE)),
        "bold": len(re.findall(r"\*\*[^*]+\*\*", text)),
        "chars": len(text),
        "tokens_approx": len(text.split()),
    }


def generate(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )
    return resp.choices[0].message.content


def main():
    all_results = []
    total_input = 0
    total_output = 0

    for lang, prompts in [("en", PROMPTS_EN), ("ja", PROMPTS_JA)]:
        for i, prompt in enumerate(prompts):
            for run in range(N_RUNS):
                sys.stdout.write(f"  [{MODEL}] {lang} prompt {i+1}/{len(prompts)} run {run+1}/{N_RUNS}...")
                sys.stdout.flush()
                try:
                    text = generate(prompt)
                    counts = count_markers(text)
                    all_results.append({
                        "model": MODEL,
                        "type": "instruct",
                        "family": "gpt4o",
                        "lang": lang,
                        "prompt_idx": i,
                        "run": run,
                        "prompt": prompt,
                        "text": text,
                        "counts": counts,
                    })
                    sys.stdout.write(f" dash={counts['dash_total']} em={counts['em_dash']} bar={counts['horizontal_bar']} colon={counts['colon']} bold={counts['bold']}\n")
                    sys.stdout.flush()
                except Exception as e:
                    sys.stdout.write(f" ERROR: {e}\n")
                    sys.stdout.flush()

    # 集計
    groups = defaultdict(list)
    for r in all_results:
        groups[r["lang"]].append(r["counts"])

    markers = ["dash_total", "em_dash", "en_dash", "horizontal_bar", "colon",
               "semicolon", "bullet", "markdown_heading", "bold"]

    print(f"\n{'='*80}")
    print(f"GPT-4o SUMMARY")
    print(f"{'='*80}")
    print(f"{'Lang':<6} {'N':>4} {'Words':>6}", end="")
    for m in markers:
        print(f" {m:>12}", end="")
    print()
    print("-" * 120)

    for lang in ["en", "ja"]:
        counts_list = groups[lang]
        n = len(counts_list)
        total_words = sum(c["tokens_approx"] for c in counts_list)
        line = f"{lang:<6} {n:>4} {total_words:>6}"
        for m in markers:
            total = sum(c[m] for c in counts_list)
            p1k = (total / total_words * 1000) if total_words > 0 else 0
            line += f" {p1k:>12.1f}"
        print(line)

    print("\n(values are per 1k words)")

    # 保存
    out_path = Path("zenn/emdash_gpt4o_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path} ({len(all_results)} records)")


if __name__ == "__main__":
    main()

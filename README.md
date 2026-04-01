# llm-stylometry

LLMの書式マーカー（em dash, 太字, 箇条書き等）がどの訓練段階で増幅されるかの実証研究。

## 実験

| スクリプト | 内容 |
|---|---|
| `emdash_compare_v2.py` | base vs instruct比較（Gemma3, Llama3, Qwen3） |
| `emdash_compare_v2_fix.py` | Qwen3系のthinkingトークン修正版 |
| `emdash_stats.py` | Mann-Whitney U検定、ブートストラップCI |
| `emdash_stats_ja.py` | 日本語データのMeCab再集計 |
| `emdash_raw_instruct.py` | プロンプト形式統制（raw completion） |
| `emdash_tulu3.py` | Tulu 3 base→SFT→DPO 3段階比較 |
| `emdash_tulu3_stats.py` | Tulu 3 統計検定 |
| `emdash_zephyr.py` | Zephyr base→SFT→DPO 3段階比較 |
| `emdash_entropy.py` | トークナイザー仮説検証（1トークン） |
| `emdash_entropy_multi.py` | トークナイザー仮説検証（5トークン） |
| `emdash_gpt4o.py` | GPT-4o API実験 |
| `emdash_preference_data.py` | DPO preference data分析（UltraFeedback, Tulu 3） |

### em dash注入実験 (2026-04-01)

| スクリプト | 内容 |
|---|---|
| `emdash_injection_dpo.py` | 注入実験用ペアデータ生成（natural + minimal） |
| `emdash_injection_sft.py` | Tulu 3データによるSFT訓練 |
| `emdash_injection_self_sft.py` | 自己注入SFT（Qwen2.5自身の出力にem dash機械挿入） |
| `emdash_injection_eval.py` | 評価・3条件比較 |
| `emdash_suppression_train.py` | DPO訓練スクリプト（cross-model DPOは失敗） |

結論: em dashは他の句読点（コロン・セミコロン）から独立している。Tulu 3データSFTでコロン・セミコロンが増えたのは、Tulu 3の文体全体を学習した結果であり、em dashとの内部結合ではない。

記事: [LLMの文体について em dash注入実験](https://flowers-of-romance.github.io/poptones/posts/llm-emdash-injection/)

## 環境

- ollama（ローカル推論）
- OpenAI API（GPT-4o）
- Python 3.12 + scipy, numpy, fugashi, datasets
- transformers, trl, peft, accelerate（注入実験用）

## データ

`data/` に各実験の生データ（JSON）を格納。

- `data/injection/` — em dash注入実験の評価結果（baseline, sft, sft_self）

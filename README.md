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

## 環境

- ollama（ローカル推論）
- OpenAI API（GPT-4o）
- Python 3.12 + scipy, numpy, fugashi, datasets

## データ

`data/` に各実験の生データ（JSON）を格納。

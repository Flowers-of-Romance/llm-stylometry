# Changelog

## 2026-04-01: em dash注入実験

em dashを含む文体をSFTで注入し、他の句読点マーカーが連動するか検証。

### 追加スクリプト
- `emdash_injection_dpo.py` — ペアデータ生成
- `emdash_injection_sft.py` — Tulu 3データSFT
- `emdash_injection_self_sft.py` — 自己注入SFT（confound排除）
- `emdash_injection_eval.py` — 評価・比較
- `emdash_suppression_train.py` — DPO訓練（cross-model DPOは失敗、記録として残す）

### 追加データ
- `data/injection/` — 3条件の評価結果

### 結果
- Tulu 3データSFT: em dash + コロン + セミコロンが増加
- 自己注入SFT: em dashのみ増加、コロン・セミコロン不変
- 結論: em dashは独立。Tulu 3 SFTでのコロン増加はcross-model confound

### 環境
- Qwen2.5-1.5B-Instruct, LoRA (r=32), CPU訓練
- NucBox EVO-X2 (AMD Ryzen AI Max+ 395, 128GB RAM)

## 2026-03-29: 初回リリース

8段階の実験でem dash増幅のメカニズムを分析。

記事: [LLMの文体について ふたたび](https://flowers-of-romance.github.io/poptones/posts/llm-emdash-dpo/)

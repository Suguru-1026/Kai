# Kanami Japanese Chat Seed v1.1

- `kanami-chat-ja-v1.jsonl`: 全136件の元データ
- `kanami-chat-ja-train-v1.1.jsonl`: 122件の学習用データ
- `kanami-chat-ja-validation-v1.1.jsonl`: 14件の検証用データ

各行は `{"user":"...","assistant":"..."}`。データは開発用に作成した合成例です。分割は `prompt = <|user|>\n{user}\n<|assistant|>\n` のFNV-1aハッシュ `% 10` を使い、値0を検証用にしています。同一promptは分割を跨ぎません。類似文の意味的な重複を除去したデータセットではありません。

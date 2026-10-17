# KNA — KNA-1 Rikka
## Neural Network Framework Specification

**文書種別：** 統合仕様書  
**対象：** KNA共通フレームワーク／KNA-1 Rikka初期モデル  
**実装言語：** JavaScript  
**初期実行環境：** ブラウザー  
**基準バックエンド：** CPU  
**基準数値型：** FP32  
**文書状態：** 設計仕様統合版。実装・テスト完了を意味しない。

---

# 目次

1. プロジェクト概要
2. 全体アーキテクチャ
3. 共通設計原則
4. Tensor Engine
5. Autograd
6. Layer Library
7. Model Graph
8. Runtime
9. Training System
10. Persistence
11. 拡張性・互換性
12. 品質保証・テスト
13. セキュリティ
14. KNA-1 Rikka初期モデル仕様
15. Tokenizer・Generation
16. 設定・モデル構築
17. エラー・キャンセル・メモリ管理
18. CPU・WebGPU共通仕様
19. 性能・実行計画
20. 公開API・モジュール境界
21. 実装ロードマップ
22. 完成判定
23. リリース・バージョン管理
24. 最終受け入れ・実装移行
25. 未実装事項・制約

---

# 1. プロジェクト概要

## 1.1 KNAとは

KNAは、ニューラルネットワークの構築・実行・学習・保存・検証を行うための汎用フレームワークである。

特定のニューラルネットワーク構造だけに限定せず、将来的には以下のアーキテクチャを扱えるように設計する。

- Transformer
- RNN
- GRU
- LSTM
- CNN
- SSM
- GNN
- MoE
- 複数方式を組み合わせたハイブリッド構造

初期実装では、CPU上で正しく動作することを最優先とする。GPU対応や高速化は、基準実装との数値的な整合性を確認しながら追加する。

## 1.2 KNA-1 Rikkaとは

KNA-1 Rikkaは、KNAの設計を具体化する初期アーキテクチャ仕様の名称である。

- `KNA`：フレームワーク全体の名称
- `1`：世代・バージョン系列
- `Rikka`：固有名称

RikkaはKNAとは別のフレームワークではなく、KNAの共通機能を利用して実装する最初の具体的モデル構成を定義する。

KNAの汎用仕様とRikka固有の選択は分離する。RikkaがTransformerを採用しても、KNA全体をTransformer専用にはしない。

## 1.3 設計の優先順位

仕様間の矛盾や実装上の判断が発生した場合、原則として次の順で優先する。

1. 安全性・データ保護
2. 共通インターフェース契約
3. KNA共通仕様
4. Rikka固有仕様
5. 実装方法
6. 性能最適化

同じ優先度の仕様が矛盾する場合、実装者が独断で一方を採用せず、変更記録を作成する。

### 仕様文書の正本と適用順位

- 本書を、KNA-1 Rikka初期リリースの採用済み設計判断を記録する基準文書とする。
- `P0シリーズ 再構築版.md`は提案中の詳細仕様であり、個々のP0項目が`ACCEPTED`となり、本書へ反映されるまでは本書を上書きしない。
- `KNA-1 Rikka.md`、`共通素体KNA.txt`、コード、テストは参照資料または実装証拠とし、本書と異なる場合は差異として扱う。
- 実装やテストが本書と異なる場合、実装を正として仕様を暗黙に読み替えず、P0-10の変更管理に従って解決する。

この判断に基づき、初期リリースではGreedy Decodingを必須とし、確率的サンプリングは後続機能とする。Token IDはTensorとは別の整数列として扱う。これらの決定を変更する場合は、本書と影響するP0仕様を同時に更新する。

決定記録日：2026-10-17。

仕様上の用語は次の意味で使用する。

- **MUST**：必須
- **MUST NOT**：禁止
- **SHOULD**：原則として推奨
- **MAY**：任意
- **TBD**：未確定

---

# 2. 全体アーキテクチャ

## 2.1 モジュール構成

```text
KNA
├── Core
│   ├── Errors
│   ├── Configuration
│   ├── Random
│   └── Utilities
│
├── Tensor Engine
│   ├── Tensor / Shape / DType
│   ├── Tensor Operations
│   └── Memory Management
│
├── Autograd
│   ├── Operation Graph
│   ├── Gradient Propagation
│   └── Gradient Accumulation
│
├── Layer Library
│   ├── Embedding / Linear
│   ├── Attention / FeedForward
│   ├── LayerNorm / Residual
│   └── Gating / MoE / Extensions
│
├── Model Graph
│   ├── Nodes / Connections
│   ├── Shape Validation
│   └── Execution Policies
│
├── Runtime
│   ├── Inference / Training
│   ├── CPU / WebGPU Dispatch
│   └── Error / Cancellation Control
│
├── Training System
│   ├── Data Pipeline
│   ├── Loss Functions
│   ├── Optimizers
│   └── Training Scheduler
│
├── Persistence
│   ├── Model Config / Weights
│   ├── Checkpoints
│   └── Recovery / Validation
│
└── Quality Assurance
    ├── Unit / Integration Tests
    ├── Numerical Verification
    └── Performance / Recovery Tests
```

## 2.2 責務

| モジュール | 責務 |
|---|---|
| Core | 共通エラー、設定、乱数、補助機能 |
| Tensor Engine | テンソル表現、数値演算、メモリ |
| Autograd | 自動微分と勾配伝播 |
| Layer Library | 再利用可能なニューラルネットワーク層 |
| Model Graph | 層の接続、形状検証、実行計画 |
| Runtime | モデル実行、バックエンド選択、キャンセル |
| Training System | データ処理、損失、最適化、学習管理 |
| Persistence | モデル・重み・チェックポイントの保存と復旧 |
| Quality Assurance | 正当性、回帰、復旧、性能の検証 |

モジュール間の依存関係は原則として一方向に保ち、循環依存を禁止する。

PersistenceとQuality Assuranceは複数のモジュールに関係する横断的な機能として設計する。

---

# 3. 共通設計原則

## 3.1 CPU基準実装

CPUバックエンドを数値的な正しさの基準とする。

GPUなどの別バックエンドは、CPUと同じ公開演算契約を満たす必要がある。

浮動小数点演算の順序や精度の違いによって完全一致しない場合は、演算ごとの許容誤差を使用する。

## 3.2 依存関係

初期実装では、必須の外部ライブラリを設けない。

ブラウザーでの利用を優先し、ローカルファイルでの利用も考慮する。`file://` でES Modulesなどに制約がある場合は、ローカルHTTPサーバーまたは単一バンドル形式を選択できるようにする。

## 3.3 暗黙の変更を禁止

次のような挙動は禁止する。

- Shape不一致を黙って修正する
- 未対応DTypeを別の型に暗黙変換する
- 最大系列長を超えた入力を黙って切り捨てる
- メモリ不足時に学習条件を無断で変更する
- 互換性のない重みを自動的に読み替える
- 未対応機能を対応済みとして扱う
- テストを実行していない機能を検証済みと報告する

安全な代替経路を使用する場合は、仕様上許可されていることを確認する。

---

# 4. Tensor Engine

## 4.1 Tensorの基本構造

Tensorは、多次元数値データと、その解釈・所有権・勾配情報を保持する。

概念上の属性は次のとおり。

```text
Tensor
├── shape
├── dtype
├── buffer
├── strides
├── offset
├── requiresGrad
├── grad
└── disposed
```

初期実装では連続したRow-Major配置を基本とする。

複雑なViewやStridesは拡張可能な設計とするが、初期段階で必要な範囲を超えて実装しない。

## 4.2 Shape

Shapeは次元サイズの配列として表現する。

例：

```js
[]           // scalar
[4]          // 1次元
[2, 3]       // 2次元
[2, 3, 4]    // 3次元
```

- 各次元は非負の安全な整数でなければならない。
- 次元の積は安全な整数範囲を超えてはならない。
- 配列長0のShapeはスカラーを表す。
- 次元サイズ0を持つ空Tensorは表現可能とする。
- スカラーと空Tensorを区別する。
- 巨大なShapeによるメモリ確保攻撃を防ぐ。
- 次元数・要素数・確保サイズには実装上限を設ける。

## 4.3 DType

初期必須型はFP32とする。

将来の候補：

- FP16
- BF16
- INT8
- INT32
- BOOL

DTypeごとのバイト数、演算対応、変換規則を明示する。未対応の型を指定された場合はエラーとする。

## 4.4 メモリ所有権

Tensorの所有権を明示する。

- Tensorの所有者は、自分が所有するリソースを解放する。
- 共有パラメーターや共有バッファを、非所有者が解放してはならない。
- 解放済みTensorへのアクセスはエラーとする。
- Autogradが必要とする中間値を、逆伝播より前に再利用してはならない。
- 非同期GPU処理が完了する前に、その処理が使用中のバッファを再利用してはならない。

## 4.5 必須演算

初期実装で必要な演算は以下とする。

### 基本演算

- 加算・減算・乗算・除算
- 行列積
- 転置
- Reshape
- Slice
- Concatenate
- Sum・MeanなどのReduction
- Exp・Log・Sqrt
- ReLU・Sigmoid・GELU・SiLU
- Softmax・LogSoftmax
- Mask
- Gather
- Where
- Cast
- ブロードキャスト

### 演算契約

- 入力Shape、出力Shape、DTypeを検証する。
- ブロードキャスト規則を統一する。
- 不正な演算を暗黙に修正しない。
- NaN・Infinityの扱いを演算と実行モードに従って検証する。
- Softmaxは数値安定性を考慮して実装する。
- 空Tensorとスカラーに対する動作を定義する。

初期CPU実装では、次の契約を採用する。

- GELUは`tanh`近似式 `0.5x(1 + tanh(sqrt(2/pi) * (x + 0.044715x^3)))` を使用する。
- `where`と`maskedFill`のMaskはFP32 Tensorで表し、値は`0`（偽）または`1`（真）に限る。Maskは出力Shapeへブロードキャストできなければならず、`maskedFill`は入力Shapeを維持する。
- `gather`のIndicesは安全な整数のJavaScript `number[]`とする。各Indexは選択軸の範囲内でなければならず、出力Shapeでは選択軸をIndicesの長さに置き換える。
- `concat`は全入力のRankと、連結軸以外の次元長が一致しなければならない。軸には負数指定もできる。

---

# 5. Autograd

## 5.1 基本方針

逆モード自動微分を初期実装の基準とする。

演算時に必要な演算グラフを記録し、損失から入力・パラメーター方向へ勾配を伝播させる。

推論モードでは不要な計算グラフを記録しない。

## 5.2 勾配規則

- 同一パラメーターが複数箇所で使われた場合、勾配を正しく累積する。
- 重み共有を壊さない。
- 勾配の初期化と累積を明確に分ける。
- backward実行時にShapeとDTypeを検証する。
- 解放済みTensorや不正な演算グラフへのアクセスを検出する。
- 学習ステップ終了後、不要な計算グラフを解放する。

## 5.3 勾配検証

数値微分による検証を実施する。

\[
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}
\]

解析的勾配と数値微分を比較する。

ReLUの折れ点など、微分可能性に問題がある点は通常の数値微分テストの対象から除外するか、別の期待値を設定する。

---

# 6. Layer Library

## 6.1 共通Layer API

```js
class Layer {
  initialize(config) {}

  inferOutputShape(inputShapes) {}

  forward(inputs, context) {}

  backward(gradOutputs, context) {}

  parameters() {}

  saveConfig() {}

  dispose() {}
}
```

これはAPI契約の概念例であり、全メソッドの内部実装を示すものではない。

各Layerは、入力・出力の形状、パラメーター、学習可能性、必要な実行コンテキストを明示する。

## 6.2 初期必須Layer

- Embedding
- Linear
- RMSNorm
- RoPE
- Causal Self-Attention
- SwiGLU FeedForward
- Residual
- Activation
- LM Head

Dropout、Gating、MoE、ほかの正規化方式などは拡張対象とする。

## 6.3 Parameter

Parameterは学習可能な値を表す。

Tensorと同一の概念として扱わず、パラメーター登録・最適化・保存のための情報を保持する。

共有されるParameterは一つの実体として管理し、勾配を累積する。

---

# 7. Model Graph

## 7.1 構成

Model GraphはLayerのノードと接続関係を管理する。

各ノードについて、入力、出力、Shape、DType、パラメーター、依存関係を検証する。

## 7.2 検証項目

- 接続先が存在する。
- 入出力の数が一致する。
- ShapeとDTypeが整合する。
- パラメーター名が重複しない。
- 不正な循環が存在しない。
- 必要な実行条件が満たされている。
- 未対応Layerが含まれていない。

初期モデルの通常の順伝播グラフでは、意図しない循環を禁止する。

## 7.3 Execution Plan

Model Graphから実行計画を生成する。

将来的には次の最適化を許可する。

- 不要な演算の除去
- 定数計算
- バッファ再利用
- 演算融合
- バックエンド選択

最適化後もShape、DType、演算の意味が保持されなければならない。

---

# 8. Runtime

## 8.1 実行モード

- `Inference`
- `Training`
- `Evaluation`

Inferenceでは通常、勾配計算を無効にする。

TrainingではAutogradを有効にし、学習用の状態と乱数を管理する。

Evaluationは評価用の実行条件を使用する。

## 8.2 実行責務

Runtimeは以下を管理する。

- モデル実行
- バックエンドの選択
- 演算のディスパッチ
- メモリ確保
- キャンセル
- エラー伝播
- プロファイリング
- リソース解放

## 8.3 エラー処理

エラーは共通形式で伝播する。

エラー発生後も、解放可能な一時リソースを安全に解放する。

モデルや学習状態を不整合なまま継続利用してはならない。

---

# 9. Training System

## 9.1 学習データ

初期モデルでは、自己回帰型の次トークン予測を使用する。

例：

```text
元系列:
[BOS, 私, は, 学生, EOS]

入力:
[BOS, 私, は, 学生]

教師ラベル:
[私, は, 学生, EOS]
```

入力と教師ラベルは1トークンずらして構成する。

## 9.2 Model Input/Output Shape

- `inputIds`: `[batch, sequence]`
- `labels`: `[batch, sequence]`
- `logits`: `[batch, sequence, vocabulary]`

`inputIds`と`labels`は矩形のJavaScript `number[][]`であり、要素は語彙範囲内の安全な整数とする。これらはTensorではない。`logits`はFP32 Tensorとする。

PADなどの無効な教師ラベルは損失計算から除外する。

有効な教師ラベルが1つも存在しないバッチはエラーとし、NaNを生成しない。

## 9.3 Loss

初期必須損失関数はCross Entropyとする。

LogSoftmaxを含め、数値安定性を考慮する。

損失の平均は有効な教師ラベルだけを対象にする。

## 9.4 Optimizer

初期OptimizerはAdamWとする。

| 設定 | 初期値 |
|---|---:|
| Learning Rate | 0.0003 |
| Beta1 | 0.9 |
| Beta2 | 0.999 |
| Epsilon | 1e-8 |
| Weight Decay | 0.01 |
| Max Gradient Norm | 1.0 |

これらは初期設定値であり、あらゆるデータに最適であることを保証する値ではない。

Weight DecayはRMSNormのスケールなど、対象外にするパラメーターを明示できるようにする。

## 9.5 学習ステップ

```text
データ取得
  ↓
Tokenizer
  ↓
入力・教師ラベル・損失マスク
  ↓
Model Forward
  ↓
Cross Entropy
  ↓
Backward
  ↓
Gradient Clipping
  ↓
AdamW Update
  ↓
Gradient Reset
  ↓
ログ・チェックポイント判定
```

勾配、損失、パラメーター、Optimizer状態にNaNまたはInfinityが発生した場合、正常な学習ステップとして確定してはならない。

## 9.6 学習再開

チェックポイントには、少なくとも次を保存する。

- 学習ステップ
- Epoch
- Optimizer状態
- Scheduler状態
- 乱数状態
- データ位置
- データ順序など再開に必要な情報

再開可能性は、保存時と復元時の互換性検証によって判定する。

---

# 10. Persistence

## 10.1 保存対象

Persistenceは次のデータを管理する。

- モデル設定
- トークナイザー設定と語彙
- 学習済み重み
- チェックポイント
- ランタイムプロファイル
- 形式・互換性情報
- チェックサム

## 10.2 パッケージ構成例

```text
kinopi-model/
├── manifest.json
├── config.json
├── tokenizer/
│   ├── config.json
│   └── vocabulary.json
├── weights/
│   ├── index.json
│   ├── shard-00000.bin
│   └── shard-00001.bin
├── training/
│   └── checkpoint.json
└── runtime/
    └── profile.json
```

この構成は論理的な保存形式の案であり、初期実装において不要なファイルを必須にするものではない。

## 10.3 重みインデックス

各重みについて、次を管理する。

- パラメーター名
- Shape
- DType
- バイト順
- ファイル名
- オフセット
- バイト長

初期保存形式はLittle-Endianを基準とする。

共有Parameterは一度だけ保存し、復元後も共有関係を保持する。

## 10.4 安全な保存

保存は次の順序を基本とする。

1. 一時領域に書き込む。
2. サイズとチェックサムを検証する。
3. 設定と重みの整合性を検証する。
4. 必要な復旧情報を確保する。
5. 完了したデータを確定する。

古い正常なチェックポイントを、新しい保存が確実に完了する前に破棄してはならない。

ブラウザーの保存機構は抽象化し、利用可能なAPIに応じて実装する。

---

# 11. 拡張性・互換性

## 11.1 拡張対象

将来、以下を追加できる構造とする。

- 新しいLayer
- 新しいTensor演算
- 新しいModel Architecture
- 新しいバックエンド
- 新しいOptimizer
- 新しいTokenizer
- 新しい保存形式

## 11.2 拡張登録

拡張機能は明示的に登録する。

登録時に以下を検証する。

- 拡張ID
- 拡張バージョン
- 必要なAPIバージョン
- 提供機能
- 依存関係
- 対応するバックエンド

未知のLayerや拡張機能を、自動的にコードとして実行してはならない。

## 11.3 互換性の独立管理

以下は独立したバージョン管理対象とする。

- Framework API
- Module API
- Model Schema
- Weight Format
- Checkpoint Format
- Tokenizer Format
- Extension API
- Rikka Specification

---

# 12. 品質保証・テスト

## 12.1 テスト分類

| 分類 | 対象 |
|---|---|
| T0 | 単体テスト |
| T1 | 勾配検証 |
| T2 | Layerテスト |
| T3 | Graphテスト |
| T4 | 統合テスト |
| T5 | システムテスト |
| T6 | 回帰テスト |

## 12.2 テスト状態

```text
PASS
FAIL
SKIPPED
ERROR
PENDING
```

- `PASS`：成功
- `FAIL`：期待結果と不一致
- `SKIPPED`：実行していない
- `ERROR`：テスト自体が正常に完了しなかった
- `PENDING`：未実施

SKIPPED、ERROR、PENDINGを成功として数えてはならない。

## 12.3 数値検証

例として、次の行列積を使用する。

```text
A = [[1, 2],
     [3, 4]]

B = [[5, 6],
     [7, 8]]

A × B = [[19, 22],
         [43, 50]]
```

Tensor Engine、Autograd、Layer、Model Graphの各段階で、独立した期待値に基づいて検証する。

## 12.4 Transformer固有テスト

- Causal Maskが未来のトークンを参照させない。
- RoPEがQ/Kに適用され、Vには直接適用されない。
- KV Cache使用時と非使用時の出力を比較する。
- EmbeddingとLM Headの重み共有を確認する。
- 共有Parameterの勾配が正しく累積される。
- 学習損失と勾配が有限値である。
- 保存・復元後にモデル構造が保持される。

## 12.5 再現性

乱数系列を用途ごとに分離する。

- 初期化
- Dropout
- データシャッフル
- データ拡張
- 生成

同じSeedでも、実行順序やバックエンドが変わればビット単位の一致が保証されるとは限らない。

再現性の水準を区別し、テストレポートに記録する。

---

# 13. セキュリティ

## 13.1 信頼境界

以下は信頼できない入力として扱う。

- 外部設定ファイル
- モデルパッケージ
- 学習データ
- チェックポイント
- 拡張機能
- 外部から与えられたTensor ShapeやDType

内部データであっても、破損や実装バグの可能性を排除しない。

## 13.2 入力検証

- 設定ファイルの構文・サイズ・深さ
- Shapeと要素数のオーバーフロー
- DTypeと確保サイズ
- モデルの重みサイズ
- チェックサム
- トークンIDの範囲
- 語彙とモデル設定の整合性
- チェックポイントの形式とバージョン

不正な入力は、モデル登録や実行開始より前に拒否する。

## 13.3 コードとデータの分離

シリアライズされたモデルデータを`eval`などで実行してはならない。

未知のLayer IDは明示的なエラーとする。

拡張機能を使用する場合は、明示的な登録と信頼判断を要求する。

## 13.4 ログ

通常のログには、ユーザーの入力文・個人情報・秘密情報を無制限に記録しない。

エラー診断に必要な識別子や状態情報と、学習データそのものを区別する。

---

# 14. KNA-1 Rikka初期モデル仕様

## 14.1 アーキテクチャ

初期モデルはDecoder-only Causal Transformerとする。

```text
Token IDs
   ↓
Token Embedding
   ↓
Transformer Block × 2
   ├── RMSNorm
   ├── Causal Multi-Head Self-Attention
   ├── Residual
   ├── RMSNorm
   ├── SwiGLU FeedForward
   └── Residual
   ↓
Final RMSNorm
   ↓
LM Head
   ↓
Next-token logits
```

Pre-Norm構造を採用する。

## 14.2 初期モデル寸法

| 設定 | 値 |
|---|---:|
| vocabSize | 512 |
| hiddenSize | 64 |
| numLayers | 2 |
| numHeads | 4 |
| headDim | 16 |
| intermediateSize | 176 |
| maxSequenceLength | 128 |
| Normalization | RMSNorm |
| Position Encoding | RoPE |
| FeedForward | SwiGLU |
| Embedding / LM Head | Tied |
| DType | FP32 |
| 初期batchSize | 1 |

制約：

- `hiddenSize % numHeads === 0`
- RoPEの`headDim`は偶数
- 語彙数はTokenizerと一致
- 共有Embedding/LM Headは同一Parameterを参照
- 最大系列長を超えた入力は黙って切り捨てない

## 14.3 パラメーター数

初期構成の設計上のパラメーター数は次のとおり。

| 項目 | 数 |
|---|---:|
| Embedding / LM Head共有 | 32,768 |
| Transformer Block 1層 | 50,304 |
| Transformer Block 2層 | 100,608 |
| Final RMSNorm | 64 |
| 合計 | 133,440 |

FP32の重みだけなら533,760バイト、約0.51MiBである。

これは重みのみの容量であり、勾配・Optimizer状態・活性値・計算グラフ・一時バッファの容量を含まない。

## 14.4 Self-Attention

Q/K/Vを独立した射影で計算する。

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

RoPEをQとKに適用し、Causal Maskを用いてAttentionを計算する。

\[
A=\operatorname{softmax}
\left(
\frac{QK^\mathsf{T}}{\sqrt{d_k}}+M
\right)
\]

\[
Z=AV,\qquad O=ZW_O
\]

- Softmaxは数値安定性を考慮する。
- Causal Maskにより未来のトークンを参照させない。
- KV Cacheは初期必須機能ではない。
- Flash Attentionは初期必須機能ではない。
- CPU基準実装を先に完成させる。

## 14.5 RoPE

RoPEは位置情報をQ/Kに反映する。

位置インデックスと系列位置の対応を保持する。KV Cacheを導入した場合、キャッシュ内の位置と新規トークンの位置を一致させる。

長系列向けのスケーリング方式は拡張対象とし、初期仕様に含めない方式を暗黙に適用してはならない。

## 14.6 SwiGLU

\[
\operatorname{SwiGLU}(x)
=
\operatorname{SiLU}(xW_g)\odot(xW_u)
\]

\[
\operatorname{FFN}(x)
=
\operatorname{SwiGLU}(x)W_d
\]

Gate、Up、Downの3つの射影を使用する。

## 14.7 Residual・Normalization

\[
x'=x+\operatorname{Attention}(\operatorname{RMSNorm}(x))
\]

\[
y=x'+\operatorname{FFN}(\operatorname{RMSNorm}(x'))
\]

各Transformer BlockはPre-NormとResidualを使用する。

## 14.8 出力

Final RMSNormの後、LM Headで語彙方向のLogitsを生成する。

出力Shapeは次のとおり。

```text
[batch, sequence, vocabulary]
```

学習ではCross Entropyを使用し、生成時のトークン選択は独立したGeneration/Sampler層で行う。

---

# 15. Tokenizer・Generation

## 15.1 Tokenizer API

```js
class Tokenizer {
  encode(text, options = {}) {}

  decode(tokenIds, options = {}) {}

  tokenToId(token) {}

  idToToken(id) {}

  get vocabSize() {}

  get specialTokens() {}

  saveConfig() {}

  validate() {}
}
```

TokenizerはKNAの演算エンジンから独立したモジュールとする。

Token ID列の公開表現はJavaScriptの`number[]`とし、各要素は`Number.isSafeInteger(id)`を満たす非負整数とする。Token ID列はTensorではなく、`float32` Tensorへ暗黙変換しない。EmbeddingはID列を検証して参照し、出力を`float32` Tensorとする。

## 15.2 特殊トークン

標準的な役割として以下を定義する。

- UNK
- BOS
- EOS
- PAD
- MASK

実際のIDと利用可否はTokenizerの設定で管理する。

文字列として`EOS`と入力されたことだけを理由に、EOSトークンとして扱ってはならない。

語彙IDは`0`から`vocabSize - 1`の範囲に収める。

## 15.3 Tokenizerの整合性

- 語彙数とモデル設定が一致する。
- IDと語彙の対応を保存する。
- 未知の入力を明示的に処理する。
- 正規化による変化を明示する。
- 異なる語彙マッピングのモデルを互換とみなさない。

将来のTokenizer候補は、文字単位、UTF-8バイト単位、BPE、Unigramなどとする。

512語彙の初期設定はテスト用であり、実用的な日本語Tokenizerを意味しない。

## 15.4 Generation

生成処理の概念API：

```js
generation.generate({
  model,
  tokenizer,
  inputIds,
  maxNewTokens,
  sampler: "greedy",
  stopTokenIds,
  useKVCache
});
```

返却情報には生成トークン列、終了理由、生成数などを含める。

終了理由：

- `eos`
- `length`
- `cancelled`
- `error`

キャンセル時には、可能な限り生成済みの部分結果を保持する。

## 15.5 初期Sampler

初期実装はGreedyを必須とする。

初期実装の既定方式はGreedyとする。初期版で未対応のサンプリング設定が明示された場合は、黙って無視したりGreedyへ置換したりせず、未対応機能としてエラーにする。

後続機能：

- Temperature
- Top-k
- Top-p
- Repetition Penalty

Repetition Penaltyの計算規則は未確定であり、P0-25に採用済みの契約が追加されるまで実装対象にしない。

Samplerの入力Logitsは有限値であることを検証する。

後続の確率サンプラーを実装する場合、Temperatureは正の値、Top-kは有効な整数、Top-pは`0 < p <= 1`とする。初期リリースではこれらの設定を受け付けない。

乱数生成器は独立したSeedと状態を持つ。

## 15.6 KV Cache

KV Cacheは後続機能とする。

保存対象：

- 層ごとのK/V
- 有効系列長
- 位置情報
- 容量
- DType
- セッション所有者

初期にはキャッシュなしの推論を基準実装とし、キャッシュありの実装を追加した際にLogitsを比較する。

---

# 16. 設定・モデル構築

## 16.1 設定例

```json
{
  "schemaVersion": 1,
  "model": {
    "name": "KNA-1 Rikka",
    "architecture": "decoder-only-transformer",
    "vocabSize": 512,
    "hiddenSize": 64,
    "numLayers": 2,
    "numHeads": 4,
    "intermediateSize": 176,
    "maxSequenceLength": 128,
    "tieWordEmbeddings": true
  },
  "normalization": {
    "type": "rmsnorm",
    "epsilon": 0.000001
  },
  "positionEncoding": {
    "type": "rope"
  },
  "attention": {
    "type": "causal-multi-head",
    "useBias": false
  },
  "feedForward": {
    "type": "swiglu",
    "useBias": false
  },
  "initialization": {
    "seed": 12345,
    "embeddingStd": 0.02
  },
  "runtime": {
    "backend": "cpu",
    "dtype": "float32"
  }
}
```

これは初期設定の構造例である。トークナイザー、学習、生成などの設定は別の設定グループとして管理できるようにする。

## 16.2 構築手順

```text
設定読み込み
  ↓
構文検証
  ↓
Schema検証
  ↓
設定間の整合性検証
  ↓
Model Graph構築
  ↓
Parameter登録
  ↓
初期化
  ↓
モデル検証
  ↓
Runtime登録
```

必須項目の欠落、未知の設定項目、不正なShape、未対応機能は明示的なエラーとする。

省略可能な設定にデフォルト値を使う場合、そのデフォルト値はSchemaに定義する。

## 16.3 初期化

初期化方式として次を用意する。

- Xavier / Glorot
- Kaiming / He
- Normal
- Uniform
- Zeros
- Ones
- Transformer向け初期化

Rikka初期設定：

| パラメーター | 初期化 |
|---|---|
| Token Embedding | Normal、平均0・標準偏差0.02 |
| Q/K/V | Xavier Uniform |
| Attention Output | Xavier Uniform |
| SwiGLU Gate / Up | Xavier Uniform |
| SwiGLU Down | Transformer向けスケーリング |
| RMSNorm Scale | Ones |
| Bias | 初期モデルでは使用しない |

Down Projectionの初期標準偏差は次の式を基準とする。

\[
\sigma=\frac{0.02}{\sqrt{2N}}
\]

`N`はTransformer Block数である。初期設定の`N = 2`では標準偏差は0.01となる。

共有Embedding/LM Headは一度だけ初期化する。

---

# 17. エラー・キャンセル・メモリ管理

## 17.1 エラーコード

モジュールごとに接頭辞を設ける。

| 接頭辞 | 分類 |
|---|---|
| CFG_ | Configuration |
| TNS_ | Tensor |
| GRD_ | Autograd |
| LYR_ | Layer |
| GRF_ | Graph |
| RUN_ | Runtime |
| TRN_ | Training |
| PST_ | Persistence |
| TOK_ | Tokenizer |
| GEN_ | Generation |
| BCK_ | Backend |
| CAN_ | Cancellation |

## 17.2 共通エラー形式

```js
{
  code,
  module,
  message,
  details,
  recoverable,
  operationId,
  cause
}
```

エラーコードは機械的な分類に使い、メッセージは原因と対処の理解に使う。

## 17.3 キャンセル状態

```text
CREATED
   ↓
RUNNING
   ├── COMPLETED
   ├── FAILED
   └── CANCEL_REQUESTED
          ↓
       CANCELLING
          ↓
       CANCELLED
```

終端状態から別の状態へ戻してはならない。

キャンセルは安全な境界で処理する。

- データバッチ境界
- Forward/Backwardの安全な境界
- Optimizer更新の前後
- チェックポイント処理
- 生成トークン境界

Optimizer更新は、完了または安全なロールバックを確認してから停止する。

## 17.4 メモリ分類

- Parameters
- Gradients
- Optimizer State
- Activations
- Autograd Graph
- KV Cache
- Temporary Buffers
- Backend Buffers
- Data Pipeline

メモリ予算はRuntime内部の管理目標であり、OS全体のメモリ使用量を保証するものではない。

## 17.5 MemoryManager

```js
class MemoryManager {
  allocate(size, options = {}) {}

  release(resource) {}

  getStats() {}

  setBudget(budget) {}

  canAllocate(size, options = {}) {}

  trim(options = {}) {}

  dispose() {}
}
```

`canAllocate`が成功しても、実際の確保が成功するとは限らない。

確保失敗・OOM・デバイス喪失時には、所有権を守りながら復旧可能な状態を維持する。

---

# 18. CPU・WebGPU共通仕様

## 18.1 Backend API

```js
class Backend {
  async initialize() {}

  getCapabilities() {}

  supports(operation, dtype, options = {}) {}

  allocateTensor(shape, dtype) {}

  execute(operation, inputs, options = {}) {}

  transferTensor(tensor, targetBackend) {}

  synchronize() {}

  dispose() {}
}
```

## 18.2 CPU基準

CPUは演算結果の正しさを検証するための基準実装とする。

WebGPUは任意の追加バックエンドであり、CPU実装の完成前に全面対応を必須としない。

## 18.3 WebGPUの段階

- G0：初期化・能力検出
- G1：基本演算
- G2：推論演算
- G3：Backward
- G4：学習
- G5：最適化

演算ごとの対応状態：

- `SUPPORTED`
- `IMPLEMENTED_UNVERIFIED`
- `UNSUPPORTED`
- `FALLBACK_ALLOWED`
- `FAILED`

CPUフォールバックは、安全であり仕様上許可されている場合だけ使用する。

## 18.4 数値許容誤差

一般的な初期比較基準：

\[
|GPU-CPU|
\leq
10^{-5}+10^{-4}|CPU|
\]

これは初期の一般基準であり、すべての演算に無条件で適用するものではない。

Reduction、MatMul、Softmax、Attention、Backwardなどは、必要に応じて演算別の基準を定義する。

---

# 19. 性能・実行計画

## 19.1 最適化原則

性能改善は、正当性とメモリ安全性を維持できる場合に限って行う。

バッファ再利用は、所有権・使用期間・Autograd依存関係を検証した上で実施する。

## 19.2 KV Cacheのメモリ見積もり

FP32のKV Cacheに必要な容量の基本式：

\[
M_{KV}=2LBSH\times\text{bytesPerElement}
\]

- `L`：層数
- `B`：バッチサイズ
- `S`：系列長
- `H`：隠れ次元

Rikka初期設定では、2層・バッチ1・系列長128・隠れ次元64・FP32の場合、128KiBとなる。

これはK/V本体のみの見積もりであり、追加バッファは含まない。

## 19.3 プロファイリング

測定対象：

- 実行時間
- 初期化時間
- 演算時間
- データ転送時間
- 確保回数
- ピークメモリ
- Tensor数
- Tokens/sec
- Samples/sec
- キャッシュ関連の統計

非同期GPU処理の測定では、同期処理の影響を区別する。

## 19.4 学習条件の保護

以下は別々の設定として管理する。

- batchSize
- sequenceLength
- maxSequenceLength
- gradientAccumulationSteps

メモリ不足を理由に、これらを黙って変更してはならない。

---

# 20. 公開API・モジュール境界

## 20.1 APIの分類

- `Public`
- `Internal`
- `Experimental`
- `Deprecated`
- `Extension`

Public APIには互換性方針を適用する。

Internal APIは内部実装用であり、安定した外部契約を保証しない。

Experimental APIは変更され得ることを明示する。

## 20.2 API例

```js
Tensor.fromArray(data, shape);

ops.matmul(a, b);

autograd.backward(loss);

runtime.compile(modelGraph);

runtime.execute(model, inputs);

trainer.trainStep(batch);

persistence.saveModel(model);

persistence.loadModel(source);

persistence.saveCheckpoint(trainer);

persistence.restoreCheckpoint(source);
```

これはAPIの方向性を示す例であり、引数・戻り値・例外の詳細は各モジュールの契約で確定させる。

## 20.3 モジュール境界

- Tensor EngineはTraining SystemやLayer Libraryに依存しない。
- AutogradはTensor Engineを使用する。
- Layer LibraryはTensor EngineとAutogradを使用する。
- Model GraphはLayerの接続と検証を管理する。
- RuntimeはGraphとBackendを使用する。
- Training SystemはRuntimeを利用する。
- Persistenceは内部Layerの実装詳細に過度に依存しない。
- Quality Assuranceは各モジュールを検証するが、通常の実行時に必須ではない。

---

# 21. 実装ロードマップ

## P0 — 仕様・共通契約・テスト基盤

- API契約の固定
- Shape・DType・メモリ規則
- 共通エラー形式
- テストハーネス
- 仕様バージョンの固定

## P1 — Tensor Engine CPU

- Tensor基本構造
- 基本演算
- Shape検証
- メモリ管理
- 数値テスト

## P2 — Autograd

- 演算グラフ
- Backward
- 勾配累積
- 数値微分テスト

## P3 — Layer Library

- Embedding
- Linear
- RMSNorm
- RoPE
- Attention
- SwiGLU
- Residual

## P4 — Model Graph / CPU Runtime

- Graph構築
- Shape検証
- 実行計画
- CPUディスパッチ
- エラー・キャンセル

## P5 — Rikka推論

- 設定読み込み
- モデル構築
- Forward
- Causal Mask
- Logits検証

## P6 — Training System

- Cross Entropy
- Backward
- Gradient Clipping
- AdamW
- 学習ログ

## P7 — Tokenizer / Generation

- 語彙データ
- encode/decode
- Greedy
- EOS・系列長・キャンセル

## P8 — Persistence

- モデル設定保存
- 重み保存
- チェックサム
- チェックポイント
- 復旧・互換性検証

## P9 — 統合・回帰テスト

- 全体統合
- 保存・復元
- 学習再開
- 数値検証
- 制約の整理
- 初期版の完成判定

## P10 — 拡張

- WebGPU
- KV Cache
- 確率的生成
- 演算融合
- Activation Checkpointing
- MoEなどの追加構造

各フェーズでは、必須テストの結果を確認してから次へ進む。失敗した場合は修正・再検証を行う。

---

# 22. 完成判定

## 22.1 完成状態の区別

| 状態 | 意味 |
|---|---|
| `MODULE_COMPLETE` | 個別モジュールの完成条件を満たした |
| `INTEGRATION_COMPLETE` | モジュール間の統合条件を満たした |
| `KNA_COMPLETE` | KNA共通仕様の完成条件を満たした |
| `RIKKA_BASE_COMPLETE` | Rikka初期モデルの完成条件を満たした |
| `RELEASE_READY` | 対象リリースの条件を満たした |

## 22.2 完成条件

- 必須仕様に準拠している。
- 必須テストに未解決の失敗がない。
- 数値結果と勾配が検証されている。
- リソースの所有権と解放が検証されている。
- 保存・復元が検証されている。
- モジュール間の統合が検証されている。
- 未実装・未検証機能が明示されている。
- 制約と対応範囲が文書化されている。

テストをスキップしたことは、テストに合格したことを意味しない。

また、Rikkaの実装が正しく動作しても、有用な日本語を生成できるとは限らない。実装の正しさと学習済みモデルの能力は別に評価する。

---

# 23. リリース・バージョン管理

## 23.1 バージョン軸

ソフトウェアと保存形式のバージョンを独立管理する。

- Framework
- Module API
- Model Schema
- Weight Format
- Checkpoint Format
- Tokenizer Format
- Extension API
- Rikka Specification

## 23.2 SemVer

安定版公開APIは原則として`MAJOR.MINOR.PATCH`を使用する。

- MAJOR：互換性を破壊する変更
- MINOR：後方互換性のある機能追加
- PATCH：後方互換性を維持する修正

初期開発の`0.x.y`、Alpha、Beta、RCは、安定版と区別する。

## 23.3 互換性

重みの読み込みでは、パラメーター名・Shape・DType・語彙対応・重み共有などを検証する。

互換性が確認できないデータは暗黙に読み替えない。

## 23.4 マイグレーション

- 元データを保持する。
- 変換前に検証する。
- 一時領域で変換する。
- 変換後に再検証する。
- 失敗時に元データを保持する。
- 変換履歴と情報損失の可能性を記録する。

## 23.5 リリースチャンネル

Development、Alpha、Beta、Release Candidate、Stableを区別する。

リリースには、バージョン情報、変更履歴、仕様、API、互換性、既知の制約、セキュリティ上の注意、テストレポートを含める。

---

# 24. 最終受け入れ・実装移行

## 24.1 受け入れゲート

- A：構造
- B：契約
- C：数値仕様
- D：実行仕様
- E：データ保護
- F：検証方針
- G：実装計画
- H：リリース管理

各ゲートは`PASS`、`FAIL`、`BLOCKED`、`PENDING`で管理する。

## 24.2 未解決事項

- Blocker：実装開始を妨げる
- Major：必須機能の完成を妨げる
- Minor：後続改善で解決できる
- Informational：制約・注意事項

重大度を変更する場合は理由と影響範囲を記録する。

## 24.3 状態の分離

- `SPEC_ACCEPTED`
- `IMPLEMENTATION_STARTED`
- `MODULE_COMPLETE`
- `INTEGRATION_COMPLETE`
- `KNA_COMPLETE`
- `RIKKA_BASE_COMPLETE`
- `RELEASE_READY`

これらは別々の状態である。

仕様が受け入れられたことだけを根拠として、実装完了を宣言してはならない。

## 24.4 実装開始時の基準

- JavaScript
- ブラウザー優先
- 必須外部依存なし
- CPU基準実装
- FP32
- Rikka初期モデル
- P0から順番に実装
- 各フェーズで検証
- 変更履歴を保持

---

# 25. 未実装事項・制約

以下は初期実装で必須ではない、または実装後の検証が必要な事項である。

- WebGPU全面対応
- KV Cache
- Flash Attention
- Top-k・Top-p
- Activation Checkpointing
- MoE
- 分散学習
- 高度なGraph最適化
- 長系列向け拡張
- 多様な追加アーキテクチャ
- 実測性能の確定
- 実用的な日本語モデルの学習品質

また、以下は仕様上の方針があっても、実装時に詳細を確定・検証する必要がある。

- ブラウザーの保存APIの具体的な選択
- Tensorの内部バッファ実装
- 演算ごとの厳密な数値許容誤差
- CPU・WebGPUの性能差
- 大規模モデル向けのメモリ最適化
- 各公開APIの最終的な引数・戻り値・例外
- 具体的な初回リリース番号

これらを未実装のまま隠すのではなく、機能台帳とテストレポートで管理する。

---

# 最終宣言

KNAは、Tensor Engine、Autograd、Layer Library、Model Graph、Runtime、Training System、Persistence、Quality Assuranceを中心とした汎用ニューラルネットワークフレームワークとして設計する。

KNA-1 Rikkaは、その初期具体化としてDecoder-only Causal Transformerを採用し、CPU・FP32での推論と学習を最初の実装目標とする。

実装は、正しさ・データ保護・検証可能性を優先し、機能拡張と高速化は基準実装の成立後に進める。

本書は、ここまでの設計決定を統合した**実装移行用の基準仕様書**である。

本書自体はコードの実装、テストの合格、性能の実測、正式リリースを証明するものではない。それらは実装工程で別途検証・記録する。
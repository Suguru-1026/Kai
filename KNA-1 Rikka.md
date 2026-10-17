# KNA — KNA-1 Rikka
## Neural Network Framework Specification

- **Framework:** KNA
- **Initial Specification:** KNA-1 Rikka
- **Specification Version:** 1.0
- **Document Type:** Architecture and Implementation Specification
- **Status:** Design Phase

---

## 1. Overview

### 1.1 KNAとは

KNAは、ニューラルネットワークの構築・実行・学習・保存・復旧・検証を統合するための、汎用ニューラルネットワーク基盤である。

特定のモデル構造だけに依存せず、Transformerを初期実装の中心に据えながら、将来的にはRNN、GRU、LSTM、CNN、SSM、GNN、MoE、ハイブリッド構造などを追加できる設計を目指す。

KNAの設計は、次の8つの大枠から構成される。

1. Tensor Engine
2. Autograd
3. Layer Library
4. Model Graph
5. Runtime
6. Training System
7. Persistence
8. Quality Assurance

### 1.2 KNA-1 Rikkaとは

KNA-1 Rikkaは、KNAの大枠を具体的な仕様に落とし込んだ初期仕様である。

- `KNA`：基盤全体の設計
- `1`：初期の仕様系列
- `Rikka`：固有名称

KNA-1 RikkaはKNAとは別のAIがKNAを利用するという意味ではなく、KNAそのものを具体化する初期仕様である。

### 1.3 設計方針

- モジュール間の責任を明確に分離する。
- CPU実装を数値的な基準実装とする。
- WebGPUは追加の計算バックエンドとして扱う。
- FP32を初期の基準データ型とする。
- 特定のモデル構造を基盤全体に強制しない。
- 形状・型・メモリ・実行状態を明示的に管理する。
- 未対応機能を黙って別の機能に置き換えない。
- 保存・復旧・キャンセル・メモリ不足を設計段階から考慮する。
- 実装の正しさはテストによって検証する。
- 初期実装の複雑さを抑えながら、将来の拡張性を維持する。

---

## 2. Overall Architecture

```text
KNA
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

### 2.1 モジュールの責任

| Module | Responsibility |
|---|---|
| Tensor Engine | テンソル表現、演算、メモリ管理 |
| Autograd | 自動微分、逆伝播、勾配蓄積 |
| Layer Library | 層・ブロックの計算とパラメータ管理 |
| Model Graph | モデル構造、接続、形状検証 |
| Runtime | 実行計画、バックエンド選択、実行制御 |
| Training System | データ処理、損失、Optimizer、学習制御 |
| Persistence | モデル・重み・学習状態の保存と復旧 |
| Quality Assurance | 数値・機能・性能・復旧の検証 |

各モジュールは、定義された責任範囲を越えて別のモジュールの機能を独自に実装しない。

---

# 3. Common Specification

## 3.1 Initial Environment

初期実装では以下を基本方針とする。

- JavaScript
- ブラウザ環境
- ローカルで動作する構成を優先
- CPUバックエンドを基準実装とする
- WebGPUは対応状況を確認しながら追加
- 外部ライブラリへの必須依存を避ける
- FP32を基本の浮動小数点型とする

ブラウザのストレージAPI、WebGPU、Workerなどは、基盤の抽象インターフェースと実環境の制約を分離する。

## 3.2 Identifier

以下のオブジェクトは一意な識別子を持つ。

- Tensor
- Operation
- Layer
- Graph Node
- Parameter
- Run
- Checkpoint

識別子はログ、エラー、依存関係、保存データの対応付けに使用する。

## 3.3 DType

初期の型体系は次のとおり。

| DType | Purpose | Initial Policy |
|---|---|---|
| `float32` | 基準浮動小数点 | 必須 |
| `float16` | 低精度浮動小数点 | 段階的対応 |
| `bfloat16` | 低精度浮動小数点 | 段階的対応 |
| `int32` | 整数・インデックス | 必要な演算から対応 |
| `int64` | 整数・トークンIDなど | 実装方式を検討 |
| `uint8` | バイト・マスク等 | 必要な演算から対応 |
| `bool` | 真偽値・マスク | 必要な演算から対応 |

型が未対応の場合は、暗黙の変換で問題を隠さず、明示的なエラーまたは定義されたCast処理を使用する。

## 3.4 Shape

Shapeは整数配列で表す。

例：

```js
[2, 4]
[8, 128, 512]
```

以下を検証する。

- 各次元が有効な整数であること。
- 要素数計算でオーバーフローしないこと。
- バッファのサイズとShapeが整合すること。
- 演算ごとに許可されたShapeであること。

ゼロ要素Tensorは基盤レベルで表現可能とするが、個々の演算が対応するかどうかは別途定義する。

## 3.5 Error Contract

エラーは構造化データとして扱う。

```js
{
  code: "SHAPE_MISMATCH",
  message: "Input shapes are incompatible.",
  module: "TensorEngine",
  runId: "run:...",
  details: {},
  recoverable: false
}
```

標準エラーコード：

- `INVALID_ARGUMENT`
- `SHAPE_MISMATCH`
- `DTYPE_UNSUPPORTED`
- `BACKEND_UNAVAILABLE`
- `OUT_OF_MEMORY`
- `NUMERICAL_ERROR`
- `MODEL_INVALID`
- `PERSISTENCE_ERROR`
- `EXECUTION_CANCELLED`
- `INTERNAL_ERROR`

各モジュールは標準コードの意味を変更してはならない。

## 3.6 Version Separation

次のバージョンは独立して管理する。

- KNA API Version
- Tensor / Operation Version
- Layer / Plugin Version
- Model Format Version
- Architecture Version
- Checkpoint Format Version

これにより、基盤のAPI変更とモデル構造の変更を区別する。

---

# 4. Tensor Engine Specification

## 4.1 Purpose

Tensor Engineは、ニューラルネットワーク計算の基礎となるテンソル表現、演算、データ配置、メモリ管理を提供する。

## 4.2 Tensor Structure

```js
{
  id: "tensor:...",
  shape: [2, 4],
  strides: [4, 1],
  dtype: "float32",
  backend: "cpu",
  buffer: null,
  offset: 0,
  requiresGrad: false,
  grad: null,
  disposed: false
}
```

`buffer`の実際の型はバックエンドによって異なる。

主なフィールド：

- `id`：識別子
- `shape`：形状
- `strides`：各次元のメモリ間隔
- `dtype`：データ型
- `backend`：CPUまたはGPU等
- `buffer`：データ保持領域
- `offset`：データ開始位置
- `requiresGrad`：勾配計算の必要性
- `grad`：勾配参照
- `disposed`：解放状態

## 4.3 Memory Layout

- 基本配置はRow-major。
- 初期実装では連続メモリ配置を優先する。
- `strides`と`offset`を設け、将来的なViewやTransposeを考慮する。
- 所有権と参照関係を明示する。
- 一時Tensorの再利用を可能にする。
- 解放済みバッファへのアクセスを防止する。

## 4.4 Operations

初期演算候補：

### Arithmetic

- Add
- Sub
- Mul
- Div
- Neg

### Linear Algebra

- MatMul
- Transpose

### Shape Operations

- Reshape
- Slice
- Concat
- Gather
- Where
- Cast

### Reductions

- Sum
- Mean
- Max

### Elementwise Functions

- Exp
- Log
- Sqrt
- ReLU
- Sigmoid
- GELU

### Neural Network Operations

- Softmax
- LayerNorm
- MaskedFill
- CausalMask

### Comparison

- Equal
- Less
- Greater

演算ごとに、入力Shape、出力Shape、対応DType、バックエンド対応、勾配対応を定義する。

## 4.5 Broadcasting

暗黙のBroadcastingは無制限に許可しない。

演算ごとにBroadcasting規則を定義し、不正な形状は明示的に拒否する。

## 4.6 Backend Contract

CPUとWebGPUは同一の演算契約を実装する。

- CPUは基準実装。
- WebGPUは対応演算を高速化する。
- バックエンド間の浮動小数点差は許容誤差で評価する。
- 未対応演算は、利用可能なCPU実装があればRuntimeがフォールバックを判断する。
- データ転送コストを無視して無条件にバックエンドを切り替えない。

## 4.7 Numerical Stability

Softmaxは最大値を引いてから指数関数を計算する。

```text
softmax(xᵢ) = exp(xᵢ - max(x)) / Σ exp(xⱼ - max(x))
```

数値異常の検出はデバッグ設定等に応じて実行する。

---

# 5. Autograd Specification

## 5.1 Purpose

Autogradは、演算グラフを記録し、逆伝播によって勾配を計算する。

## 5.2 Operation Record

演算記録には次の情報を含める。

- Operation Type
- Input Tensor References
- Output Tensor References
- Gradient Requirements
- Backward Support
- Backwardに必要なSaved State

勾配計算が不要な推論では、原則として計算グラフを記録しない。

## 5.3 Backward

逆伝播はReverse-mode Automatic Differentiationを基本とする。

同じTensorに複数の経路から勾配が到達した場合、勾配を加算する。

勾配対応演算の初期候補：

- Add / Sub / Mul / Div
- MatMul
- Transpose
- Reshape
- Sum / Mean
- Exp / Log / Sqrt
- ReLU / Sigmoid / GELU
- Softmax
- LayerNorm
- Embedding / Gather
- Concat / Slice

比較演算などの離散的な演算は、原則として通常の微分対象にしない。

未対応のBackwardを暗黙に無視せず、明示的なエラーにする。

## 5.4 Parameter and Gradient

ParameterとTensorを区別する。

- Tensor：計算に使用するデータ。
- Parameter：モデルが保持する学習可能な値。
- Gradient：Parameterに対する勾配。

OptimizerはParameterを更新し、Autogradは勾配を計算する。

## 5.5 Gradient Accumulation

- 複数経路の勾配は合算する。
- 勾配の初期化はTraining Systemが制御する。
- 勾配蓄積を行う場合は、蓄積回数とLossの重み付けを明示する。
- 将来的にActivation Checkpointingなどのメモリ削減を追加可能とする。

---

# 6. Layer Library Specification

## 6.1 Purpose

Layer Libraryは、Tensor Engineの演算を組み合わせ、再利用可能なニューラルネットワークの層やブロックを提供する。

## 6.2 Common Layer Interface

```js
Layer
├── initialize(config)
├── inferOutputShape(inputShapes)
├── forward(inputs, context)
├── backward(gradOutputs, context)
├── parameters()
├── saveConfig()
└── dispose()
```

各メソッドの実際の引数・戻り値・非同期処理の規則は、実装インターフェース確定時に定義する。

Layerは次を担当する。

- 設定の検証
- 出力Shapeの推定
- Forward計算
- Backward計算、またはAutogradへの委譲
- Parameterの提供
- 設定のシリアライズ
- リソース解放

LayerはOptimizerを直接実行しない。

## 6.3 Initial Layer Set

### Embedding

整数IDをベクトル表現へ変換する。

### Linear

線形変換を実行する。

```text
y = xWᵀ + b
```

### SelfAttention

- Query
- Key
- Value
- Multi-head処理
- Causal Mask
- Output Projection

KV Cacheは推論最適化として後続実装で追加する。

### FeedForward

基本構成：

```text
Linear → Activation → Linear
```

活性化関数の具体的な選択はモデル構成仕様で確定する。

### LayerNorm

指定された正規化軸に対して正規化する。

### Residual

入力と変換結果を結合する。

### Activation

ReLU、Sigmoid、GELU等の活性化を提供する。

### Dropout

学習時の確率的な要素無効化を行う。推論時の挙動を学習時と区別する。

### Gating / MoE

将来の拡張として扱う。

- Gating：経路や処理の選択
- MoE：複数Expertの選択・実行・出力統合

## 6.4 Extension Metadata

拡張モジュールには、次の情報を持たせる。

```js
{
  id: "layer.example",
  type: "layer",
  version: "1.0.0",
  apiVersion: "1.0",
  capabilities: {
    inference: true,
    training: true,
    backward: true,
    cpu: true,
    webgpu: false
  }
}
```

宣言されたCapabilityは、実際のテストで検証する。

---

# 7. Model Graph Specification

## 7.1 Purpose

Model Graphは、モデルの構造、接続、Parameter参照、実行方針を表現する。

## 7.2 Node

Nodeには次の情報を持たせる。

- `id`
- `type`
- `version`
- `inputs`
- `config`
- `parameterRefs`
- `executionPolicy`

## 7.3 Edge

EdgeはNode間のデータ接続を表す。

```js
{
  from: "node:a",
  fromOutput: 0,
  to: "node:b",
  toInput: 0
}
```

## 7.4 Execution Policies

- Always：常時実行
- Conditional：条件に応じて実行
- Routed：選択された経路を実行

初期のKNA-1 RikkaではAlwaysを基本とする。

## 7.5 Validation

実行前に次を検証する。

- 参照先Nodeが存在する。
- 入出力の対応が正しい。
- ShapeとDTypeが整合する。
- Parameter参照が有効である。
- 通常のグラフに不正な循環がない。
- 実行方針に必要な入力が存在する。
- 必要なバックエンド機能が利用可能である。

通常の循環は許可しない。状態を持つ再帰的な構造は、専用の仕組みとして追加する。

## 7.6 Execution Plan

検証済みGraphからExecution Planを生成する。

Execution Planは以下を保持する。

- 実行順序
- 入力・出力参照
- 解決済みShape
- 使用する演算実装
- 一時メモリの見積もり
- Execution Policy
- Autogradの必要性

---

# 8. Runtime Specification

## 8.1 Purpose

Runtimeは、モデルの実行計画を制御し、バックエンド選択、メモリ管理、キャンセル、エラー処理、プロファイル収集を統括する。

## 8.2 Execution Modes

### Inference

- 勾配計算を無効化する。
- Optimizerを実行しない。
- Parameterを更新しない。
- 必要に応じてKV Cacheを使用する。

### Training

- 勾配計算を有効化する。
- 中間結果を必要に応じて保持する。
- Training Systemの制御に従う。

### Evaluation

- 評価用のForwardを実行する。
- 通常、Parameter更新を行わない。
- 推論・学習の評価モードの違いを明示する。

## 8.3 Backend Dispatch

```js
{
  runtime: {
    preferredBackend: "auto",
    allowCPUFallback: true,
    memoryPolicy: "adaptive",
    debugChecks: false
  }
}
```

この設定は初期候補であり、実装時に型と許容値を確定する。

Dispatchでは次を考慮する。

- 演算サイズ
- CPUとGPUの実行コスト
- データ転送コスト
- 対応演算
- 使用可能なメモリ
- 現在のバックエンド状態

## 8.4 Memory Categories

- Persistent：モデル重み・永続状態
- Temporary：中間Tensor・作業領域
- External Storage：重みファイル・チェックポイント等

## 8.5 OOM Policy

メモリ不足時の処理順序：

1. 不要な一時領域を解放する。
2. 再利用可能なバッファを再利用する。
3. 対応可能ならBatchや処理単位を縮小する。
4. 対応可能なら部分読込等を検討する。
5. 復旧できない場合はOUT_OF_MEMORYを返す。

無制限な再試行は禁止する。

## 8.6 Runtime State

```text
IDLE
  ↓
PREPARING
  ↓
RUNNING
  ├── PAUSING → PAUSED
  ├── STOPPING → CANCELLED
  ├── COMPLETED
  └── ERROR
```

Pause、Stop、Cancelは安全な処理境界で実施する。

GPU処理が完了する前に、その処理が利用しているリソースを解放してはならない。

## 8.7 Profiling

記録候補：

- durationMs
- backend
- memoryUsage
- operationCount
- tokensProcessed
- fallbackCount
- errorCount

---

# 9. Training System Specification

## 9.1 Purpose

Training Systemは、データ取得からLoss計算、逆伝播、Parameter更新、ログ、チェックポイントまでの学習手順を管理する。

## 9.2 Data Pipeline

```text
Data Source
    ↓
Validator
    ↓
Tokenizer Adapter
    ↓
Sequence Builder
    ↓
Batch Builder
    ↓
Training Scheduler
    ↓
Model Forward
    ↓
Loss / Backward / Optimizer
```

Tokenizer Adapterは将来のTokenizer実装との接続点とする。

## 9.3 Data Sources

初期候補：

- JavaScript Array
- JSON / JSONL
- Plain Text
- Sharded Binary Files
- Custom Data Source

IndexedDB等は追加可能なデータソースとする。

## 9.4 Training Sample

```js
{
  inputIds: [12, 45, 78, 91],
  targetIds: [45, 78, 91, 23],
  attentionMask: [1, 1, 1, 1]
}
```

Causal Language Modelingでは、入力と教師データを次トークン予測に適合するようにずらす。

Padding部分はLossの計算対象から除外する。

## 9.5 Sequence and Batch Building

- 最大系列長はモデル設定に従う。
- 長い系列の分割規則を明示する。
- PaddingとAttention Maskを整合させる。
- Loss対象外のTokenを明示する。
- 将来的にLength Bucketing、Token-budget Batching、Gradient Accumulationを追加できるようにする。

## 9.6 Loss

初期必須LossはCross Entropyとする。

数値的に安定したLogSoftmaxとNLL相当の計算を用いる。

有効な教師Tokenが存在しないBatchは明示的なエラーとする。

NaNやInfを検出した場合は、学習を安全に停止する。

## 9.7 Optimizers

初期候補：

- SGD
- Adam
- AdamW

Transformer向けの初期既定候補はAdamWとする。ただし、学習率等の具体的な値はモデル構成・学習設定で別途定義する。

OptimizerはParameterとそのGradientを受け取り、Parameterを更新する。

## 9.8 Learning Rate Scheduler

初期候補：

- Constant
- Linear
- Cosine
- Warmup

## 9.9 Training Step

```text
Validate Batch
    ↓
Zero Gradients (when appropriate)
    ↓
Forward
    ↓
Loss
    ↓
Numerical Check
    ↓
Backward
    ↓
Gradient Clipping (if enabled)
    ↓
Optimizer Update
    ↓
Post-update Validation
    ↓
Increment Step / Log / Checkpoint
```

更新途中の失敗によってParameterの一部だけが変更された状態を避けるため、可能なら更新を検証後に反映する。

## 9.10 Training State

```text
IDLE
PREPARING
RUNNING
PAUSING
PAUSED
STOPPING
CANCELLED
COMPLETED
ERROR
```

Pauseは安全な処理境界で実施する。

## 9.11 Training Logs

- globalStep
- epoch
- loss
- learningRate
- tokensProcessed
- elapsedTime
- backend
- memoryUsage
- skippedBatches

乱数状態は、初期化、Shuffle、Dropout、MoE Routing等の再現性に必要な範囲で管理する。

同じSeedでも、CPUとWebGPUの実行が完全一致するとは限らない。

---

# 10. Persistence Specification

## 10.1 Purpose

Persistenceは、モデル構造、重み、学習状態、実行設定を保存し、検証・復旧できるようにする。

## 10.2 Data Categories

1. Model Config
2. Model Weights
3. Training Checkpoint
4. Runtime Profile

## 10.3 Package Layout

```text
kinopi-model/
├── manifest.json
├── config.json
├── weights/
│   ├── index.json
│   ├── shard-00000.bin
│   └── shard-00001.bin
├── training/
│   └── checkpoint.json
└── runtime/
    └── profile.json
```

Tokenizer設定は将来の拡張点として追加可能とする。

## 10.4 Manifest

マニフェストには次の情報を含める。

- Format Version
- Model ID
- Creation Metadata
- File List
- Required Flags
- File Sizes
- Checksums

SHA-256は破損検出に利用する。データの真正性を保証するものではない。

## 10.5 Weight Index

重みインデックスには次を記録する。

- Tensor Name
- Shape
- DType
- Endianness
- File
- Offset
- Byte Length

検証項目：

- ShapeとByte Lengthの一致
- ファイル範囲外への参照がないこと
- 不正な重複領域がないこと
- 名前が一意であること
- 対応していない型を受け入れないこと

FP32の512×512行列は、ヘッダー等を除く生データで1,048,576 bytesとなる。

## 10.6 Checkpoint

チェックポイントには、必要に応じて以下を保存する。

- Global Step
- Epoch
- Model Weights Reference
- Optimizer Type and State
- Optimizer Step
- Scheduler State
- RNG State
- Data Position
- Data Order
- Training Configuration

初期実装では、Gradient Accumulationの途中状態ではなく、Step境界での保存を優先する。

## 10.7 Save Protocol

```text
Prepare
  ↓
Write to Temporary Generation
  ↓
Validate Structure and Size
  ↓
Verify Checksums
  ↓
Commit Generation
  ↓
Clean Old Generations
```

ブラウザストレージの原子性には差があるため、世代管理を利用する。

## 10.8 Recovery

- Manifestを検証する。
- ConfigとGraphを検証する。
- Weightsの名前・型・形状・サイズを検証する。
- 必要なTraining Stateを検証する。
- 対応可能なVersionであることを確認する。
- 検証完了後にモデルを利用可能にする。

復旧に失敗した場合は、既存の正常なモデルを可能な限り維持する。

---

# 11. Extensibility and Compatibility

## 11.1 Extension Categories

- Layer Plugin
- Operation Plugin
- Architecture Plugin
- Backend Plugin

## 11.2 Registration Lifecycle

```text
UNREGISTERED
    ↓
REGISTERED
    ↓
METADATA_VALIDATED
    ↓
COMPATIBILITY_CHECKED
    ↓
IMPLEMENTATION_TESTED
    ↓
AVAILABLE
```

Capabilityの宣言だけで実装の正しさを保証しない。

## 11.3 Compatibility Policy

- 対応する旧形式は明示的なMigrationで読み込む。
- 未知の新形式は原則として拒否する。
- 未対応Layerを意味の異なるLayerに置き換えない。
- 同一の意味を持つ演算であれば、CPU実装へのフォールバックを許可できる。
- 既存モデルの動作を無断で変更しない。

## 11.4 Security and Resource Limits

初期実装では、登録されていない外部コードを自由に実行する仕組みを設けない。

拡張に対しては以下を検証する。

- Config
- Shape / DType
- Memory Limits
- Execution Mode
- Backend Capability
- Disposal and Resource Ownership

必要に応じて将来的にWorker等による分離を検討する。

## 11.5 Future Architecture Extensions

- GRU / LSTM
- CNN
- Encoder Architectures
- SSM
- GNN
- MoE
- Hybrid Architectures
- Additional Backends

これらはKNA全体の拡張候補であり、初期仕様ですべてを実装することを意味しない。

---

# 12. Quality Assurance Specification

## 12.1 Test Levels

1. Tensor Tests
2. Autograd Tests
3. Layer Tests
4. Model Graph Tests
5. Training Tests
6. System / Recovery Tests

## 12.2 Tensor Tests

- Shape
- MatMul
- Transpose
- Reshape
- Softmax
- LayerNorm
- Causal Mask
- Memory Reuse
- Invalid Inputs
- Unsupported DType

## 12.3 Autograd Tests

数値微分との比較により勾配を検証する。

中央差分の例：

```text
f'(x) ≈ (f(x + ε) - f(x - ε)) / (2ε)
```

誤差許容値は演算、データ型、バックエンドに応じて設定する。

ReLUの境界など、微分が不連続または未定義となる点には注意する。

## 12.4 Transformer Tests

- 入力・出力Shape
- Causal Mask
- 未来Tokenの情報が過去位置の出力に漏れないこと
- Residual接続
- Normalization
- KV Cacheあり・なしの結果比較

## 12.5 Training Tests

段階的な学習テスト：

1. 固定出力の学習
2. 単純なパターンの学習
3. 短い系列の次Token予測
4. Validation Splitを用いた評価
5. より長い系列の学習

評価指標：

- Cross Entropy
- Perplexity
- Validation Loss
- Tokens per Second
- Peak Memory
- Numerical Error

自然対数を用いたCross Entropyでは、Perplexityは `exp(CrossEntropy)` で定義する。

## 12.6 Backend Comparison

CPUとWebGPUの出力・勾配を比較する。

- Maximum Absolute Error
- Relative Error
- Execution Duration
- Memory Usage

浮動小数点演算の違いを考慮し、妥当な許容誤差を設定する。

## 12.7 Persistence and System Tests

- 保存・読込後の出力一致
- チェックポイント復旧
- 破損ファイルの拒否
- メモリ不足
- キャンセル
- 長時間実行時のメモリリーク
- 不正Graph
- 未対応演算
- バージョン互換性
- 旧チェックポイントの復旧

## 12.8 Test Result Format

```js
{
  testId: "tensor.matmul.basic",
  status: "passed",
  backend: "cpu",
  durationMs: 1.2,
  metrics: {
    maxAbsoluteError: 0
  },
  error: null
}
```

Status：

- `passed`
- `failed`
- `skipped`
- `error`

`skipped`は成功として数えない。

## 12.9 Quality Gates

品質確認は次の段階で行う。

1. Tensor Engine
2. Autograd and Layers
3. Model Graph and Runtime
4. Training
5. Persistence and Recovery
6. Optional Backends and Extensions

---

# 13. Unified Specification and Implementation Contract

## 13.1 Shared Data Contract

### Tensor

Shape、DType、Backend、Buffer、Offset、Gradient State、Disposal Stateを整合させる。

### Parameter

Tensorとは別の管理対象とし、学習可能な値と勾配を対応付ける。

### RunContext

実行単位を識別し、実行モード、バックエンド方針、学習Step、キャンセル状態、デバッグ設定を管理する。

## 13.2 Inference Flow

```text
Input
  ↓
Input Validation
  ↓
Validated Execution Plan
  ↓
Runtime
  ↓
Tensor Engine / Layer Library
  ↓
Output Validation
  ↓
Result
```

推論では原則として勾配グラフを記録せず、Optimizerを動作させず、Parameterを変更しない。

## 13.3 Training Flow

```text
Data
  ↓
Validation
  ↓
Forward
  ↓
Loss
  ↓
Numerical Check
  ↓
Backward
  ↓
Gradient Clipping (Optional)
  ↓
Optimizer Update
  ↓
Post-update Validation
  ↓
Step / Log / Checkpoint
```

失敗した学習Stepを成功として記録しない。

## 13.4 Error Handling

- 共通エラーコードを使用する。
- エラーの発生元とRun IDを記録する。
- 復旧可能性を区別する。
- CPUフォールバックはRuntimeが判断する。
- 無制限な再試行を禁止する。
- 未対応機能を成功扱いしない。

## 13.5 Persistence Contract

保存データは検証してから確定する。

復旧は構造・バージョン・Shape・DType・サイズ・必要な学習状態を検証した後に完了とする。

## 13.6 Core Invariants

1. 数値の正しさを維持する。
2. 勾配の正しさをテストする。
3. モジュールの責任境界を維持する。
4. 失敗を明示する。
5. 学習・保存・復旧の状態を整合させる。
6. CPUとWebGPUで演算の意味を変えない。
7. 既存モデルの動作を無断で変更しない。
8. テスト結果によって実装の正しさを判定する。

---

# 14. Implementation Roadmap

## Phase 1 — Common Contracts

- ID体系
- Tensor / Parameter型
- Error Contract
- Version Contract
- 基本テスト構造

## Phase 2 — Tensor Engine

- CPU Tensor
- 基本演算
- Shape Validation
- Memory Ownership
- 数値テスト

## Phase 3 — Autograd

- Operation Graph
- Gradient Propagation
- Gradient Accumulation
- Numerical Gradient Tests

## Phase 4 — Layer Library

- Embedding
- Linear
- LayerNorm
- Activation
- SelfAttention
- FeedForward
- Residual

## Phase 5 — Model Graph and Runtime

- Graph Validation
- Execution Plan
- CPU Execution
- Cancellation
- Error Handling
- Profiling

## Phase 6 — Training System

- Data Pipeline
- Cross Entropy
- Optimizers
- Scheduler
- Training Loop
- Logging

## Phase 7 — Persistence

- Model Config
- Binary Weights
- Checkpoints
- Recovery
- Corruption Tests

## Phase 8 — WebGPU and Extensions

- GPU Operation Implementations
- CPU / GPU Comparison
- Backend Dispatch
- Optional Architecture Extensions

Quality AssuranceはすべてのPhaseで並行して実施する。

---

# 15. Initial Model Architecture: Pending Specification

KNA-1 Rikkaの初期モデルはDecoder-only Causal Transformerを中心候補とする。

```text
Token IDs
    ↓
Token Embedding + Position Information
    ↓
Transformer Block × N
    ├── Causal SelfAttention
    ├── Residual / Normalization
    ├── FeedForward
    └── Residual / Normalization
    ↓
LM Head
    ↓
Next-token Prediction
```

想定されるTensor Shape：

- Token IDs: `[batch, sequence]`
- Embedding: `[batch, sequence, hidden]`
- Attention: `[batch, heads, sequence, head_dim]`
- Output Logits: `[batch, sequence, vocabulary]`

Attentionの基本式：

```text
Attention(Q, K, V)
  = softmax((QKᵀ) / sqrt(dₖ) + M)V
```

`M`は未来Tokenへの参照を禁止するCausal Maskである。

以下は未確定であり、モデル構成仕様で決定する。

- Position Encoding方式
- Normalizationの配置と順序
- Attentionの細部
- FeedForwardの活性化関数
- Hidden Size
- Attention Head数
- Transformer Block数
- Vocabulary Size
- Tokenizer
- 初期モデルのParameter数
- メモリ予算
- 学習設定

過去の例示値は、これらの最終決定を意味しない。

---

# 16. Current Status

| Item | Status |
|---|---|
| KNAの8モジュール | 定義済み |
| 共通仕様・型体系 | 基本仕様定義済み |
| Tensor Engine | 基本仕様定義済み |
| Autograd | 基本仕様定義済み |
| Layer Library | 基本仕様定義済み |
| Model Graph | 基本仕様定義済み |
| Runtime | 基本仕様定義済み |
| Training System | 基本仕様定義済み |
| Persistence | 基本仕様定義済み |
| Extensibility | 基本仕様定義済み |
| Quality Assurance | 基本仕様定義済み |
| 統合契約 | 基本仕様定義済み |
| 初期モデルの詳細構成 | 未確定 |
| 実装・性能検証 | 未実施または今後の実装段階 |

「仕様定義済み」は設計上の合意を意味し、実装済み・動作検証済みという意味ではない。

---

# 17. Next Step

次の設計段階では、**KNA-1 Rikkaの初期モデル構成仕様**を確定する。

対象：

1. Position Encoding
2. Normalization
3. Attention
4. FeedForward
5. Transformer Block
6. 初期モデルのパラメータ
7. TokenizerとVocabulary
8. 初期メモリ予算と実行条件

KNA全体の拡張性を維持しながら、KNA-1 Rikkaとして最初に実装する構成を明確にする。




…色々困惑するだろうけど12の再登場
# KNA — KNA-1 Rikka
## Part 12：統合仕様・実装契約

**文書種別：** 技術仕様書  
**対象：** KNA共通ニューラルネットワークフレームワーク / KNA-1 Rikka  
**対象範囲：** Part 12.1〜12.14.1  
**文書状態：** 統合設計案  
**初期実装言語：** JavaScript  
**基準バックエンド：** CPU / FP32

---

# 目次

- 12.1 初期モデルアーキテクチャ
- 12.2 モデル次元・パラメータ数・メモリ
- 12.3 Tokenizer・語彙仕様
- 12.4 学習データ・損失関数・学習ループ
- 12.5 生成・Sampler・KV Cache
- 12.6 初期化・数値安定性
- 12.7 設定ファイル・モデル構築
- 12.8 エラー・キャンセル仕様
- 12.9 CPU / WebGPU共通仕様
- 12.10 メモリ管理・実行計画最適化
- 12.11 テスト・再現性
- 12.12 公開API・モジュール境界
- 12.13 セキュリティ・信頼境界
- 12.14 最終統合・実装開始条件・完成判定
  - 12.14.1 全仕様の統合基準

---

# 12.1 初期モデルアーキテクチャ

## 12.1.1 基本方針

KNA-1 Rikkaの初期モデルには、Decoder-only Causal Transformerを採用する。

KNA全体はTransformer専用ではない。将来的にRNN、GRU、LSTM、CNN、SSM、GNN、MoE、ハイブリッド構造などを追加できる設計とする。

## 12.1.2 モデル構造

```text
Token IDs
    |
Token Embedding
    |
Transformer Block × N
    |-- RMSNorm
    |-- Causal Self-Attention
    |-- Residual Connection
    |-- RMSNorm
    |-- SwiGLU FeedForward
    |-- Residual Connection
    |
Final RMSNorm
    |
LM Head
    |
Logits [B, S, V]
```

## 12.1.3 正規化

Pre-Norm構造を採用し、初期モデルの正規化にはRMSNormを使用する。

RMSNormはKNA全体の必須方式ではなく、モデル構成で選択可能なLayerとする。

## 12.1.4 位置情報

位置情報にはRoPE（Rotary Position Embedding）を使用する。

- RoPEはAttentionのQ・Kに適用する。
- Vには直接適用しない。
- KV Cache使用時は、位置インデックスを一貫させる。
- 周波数設定と長文脈向け拡張は設定可能とする。

## 12.1.5 Attention

Multi-Head Causal Self-Attentionを採用する。

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

\[
A=\operatorname{softmax}
\left(\frac{QK^\mathsf{T}}{\sqrt{d_k}}+M\right)
\]

\[
Z=AV,\quad O=ZW_O
\]

ここで \(M\) は未来のトークンへの参照を禁止するCausal Maskである。

初期実装ではCPU上の参照実装を優先し、Flash Attentionは必須としない。

## 12.1.6 FeedForward

SwiGLUを採用する。

\[
\operatorname{SwiGLU}(x)
=\operatorname{SiLU}(xW_g)\odot(xW_u)
\]

\[
\operatorname{FFN}(x)
=\operatorname{SwiGLU}(x)W_d
\]

Gate、Up、Downの3投影を使用する。

## 12.1.7 残差接続

\[
x'=x+\operatorname{Attention}(\operatorname{RMSNorm}(x))
\]

\[
y=x'+\operatorname{FFN}(\operatorname{RMSNorm}(x'))
\]

各サブレイヤーの出力を残差接続で加算する。

## 12.1.8 出力

最終RMSNormとLM Headを通じて、次トークン予測用Logitsを出力する。

形状は `[batch, sequence, vocabulary]` とする。

学習にはCross Entropyを使用する。生成時のトークン選択はSamplerに分離する。

KV Cacheは後続機能として実装し、キャッシュなしの参照動作を先に確立する。

---

# 12.2 モデル次元・パラメータ数・メモリ

## 12.2.1 初期参照モデル

| 項目 | 初期値 |
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
| Embedding / LM Head | 重み共有 |
| dtype | FP32 |
| batchSize | 1 |
| dropout | 0.0 |

この構成は小規模な動作確認・数値検証用であり、実用的な日本語言語モデルとして十分な規模であることを意味しない。

## 12.2.2 次元制約

- `hiddenSize % numHeads === 0`
- RoPE使用時、`headDim` は偶数とする。
- Tokenizerの語彙サイズとモデルの`vocabSize`を一致させる。
- 最大系列長を超えた入力を暗黙に切り捨てない。
- 形状の不整合はモデル構築時に検出する。

## 12.2.3 パラメータ数

初期モデルのパラメータ数は以下の設計値とする。

| 構成要素 | パラメータ数 |
|---|---:|
| Token Embedding | 32,768 |
| Transformer Block × 2 | 100,608 |
| Final RMSNorm | 64 |
| **合計** | **133,440** |

EmbeddingとLM Headの重みは共有し、二重計上しない。

FP32の重み本体は533,760 bytes、約0.51 MiBである。勾配、Optimizer State、Activation、作業用バッファなどは別途必要になる。

---

# 12.3 Tokenizer・語彙仕様

## 12.3.1 独立性

TokenizerはKNAのTensor EngineやModel Graphから独立したモジュールとする。

モデル本体はTokenizerの内部アルゴリズムに依存せず、トークンID列を入力として受け取る。

## 12.3.2 API

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

## 12.3.3 特殊トークン

以下の役割を標準化する。

- `UNK`：未知トークン
- `BOS`：系列開始
- `EOS`：系列終了
- `PAD`：パディング
- `MASK`：マスク用

初期参照設定ではIDを次のように割り当てる。

| トークン | ID |
|---|---:|
| UNK | 0 |
| BOS | 1 |
| EOS | 2 |
| PAD | 3 |
| MASK | 4 |

特殊トークンを実際に利用するかどうかは設定で制御する。ユーザー入力に文字列として`EOS`が含まれていても、それだけでEOSトークンとして扱ってはならない。

## 12.3.4 初期語彙

`vocabSize = 512`はテスト用の固定語彙である。

将来的には文字単位、UTF-8バイト単位、BPE、Unigram、SentencePiece系などを追加可能とする。

語彙変更時には、ID対応・重みとの整合性・再学習や変換の必要性を検証する。

---

# 12.4 学習データ・損失関数・学習ループ

## 12.4.1 次トークン予測

入力列と正解ラベルを1トークンずらす。

例：

```text
元系列:
[BOS, 私, は, 学生, EOS]

inputIds:
[BOS, 私, は, 学生]

labels:
[私, は, 学生, EOS]
```

テンソル形状は次のとおり。

- `inputIds`: `[B, S]`
- `labels`: `[B, S]`
- `logits`: `[B, S, V]`

## 12.4.2 Causal Mask

各位置は未来のトークンを参照できない。

テストでは、未来側の入力を変更しても、それ以前の位置の予測が影響を受けないことを検証する。

## 12.4.3 Cross Entropy

Cross Entropyは数値的に安定したLog-Softmaxを使用する。

- PADや明示的に無視するラベルを損失計算から除外する。
- 有効なラベルだけを平均対象とする。
- 有効ラベルが0件の場合はエラーとする。
- NaNやInfinityを損失として受け入れない。

## 12.4.4 学習ループ

```text
データ取得
  ↓
Tokenizer
  ↓
inputIds / labels / lossMask
  ↓
Model Forward（Training）
  ↓
Cross Entropy
  ↓
Backward
  ↓
Gradient Clipping（設定時）
  ↓
Optimizer Step
  ↓
Gradient Reset
  ↓
ログ・チェックポイント
```

## 12.4.5 初期Optimizer

初期OptimizerはAdamWとする。

| パラメータ | 初期値 |
|---|---:|
| Learning Rate | 0.0003 |
| beta1 | 0.9 |
| beta2 | 0.999 |
| epsilon | 1e-8 |
| weightDecay | 0.01 |
| maxGradientNorm | 1.0 |

これらは設定で変更可能とする。

Weight Decayの適用対象は明示的に定義し、RMSNormのスケールなど、除外対象に指定されたParameterには適用しない。

---

# 12.5 生成・Sampler・KV Cache

## 12.5.1 役割分担

ModelはLogitsを生成し、Samplerは次トークンを選択する。

```text
Tokenizer
  ↓
入力トークン列
  ↓
Model Forward
  ↓
Logits
  ↓
Sampler
  ↓
次トークン
  ↓
終了条件判定
  ↓
Decode
```

## 12.5.2 Generation API

```js
generation.generate({
  model,
  tokenizer,
  inputIds,
  maxNewTokens,
  sampler,
  stopTokenIds,
  useKVCache
});
```

戻り値の基本形：

```js
{
  tokenIds: [],
  finishReason: "eos",
  generatedTokens: 0
}
```

`finishReason`は`eos`、`length`、`cancelled`、`error`を区別する。可能な場合はエラーやキャンセルまでの部分出力を保持する。

## 12.5.3 Sampler

- **Greedy**：最大Logitのトークンを選択。
- **Temperature**：温度による確率分布の調整。温度は0より大きい値とする。
- **Top-k**：候補を上位k件に制限。kは1以上かつ語彙サイズ以下。
- **Top-p**：累積確率が指定値に達するまで候補を残す。値は0より大きく1以下。

複合フィルターの標準順序は次のとおり。

1. Logitsの有限値検証
2. Repetition Penalty
3. Temperature
4. Top-k
5. Top-p
6. 確率の再正規化
7. サンプリング

Greedyは原則として確率的フィルターを通さず、独立した選択方式とする。

## 12.5.4 乱数

生成用RNGを独立させ、Seedを指定・保存可能とする。

異なるバックエンド間では浮動小数点演算の差により、完全一致を保証しない。

## 12.5.5 KV Cache

KV CacheはLayerごとのK・V、現在長、位置情報、容量、dtypeを管理する。

初期基準はFP32とし、キャッシュあり・なしのLogitsを比較検証する。

キャッシュの所有権、最大系列長、確保失敗、セッション間の混同、解放処理を検証する。

---

# 12.6 初期化・数値安定性

## 12.6.1 Initializer

KNA共通のInitializerとして、以下を用意する。

- Xavier / Glorot
- Kaiming / He
- Normal
- Uniform
- Zeros
- Ones
- Transformer向け初期化

## 12.6.2 Rikka初期化方針

| Parameter | 初期化 |
|---|---|
| Token Embedding | Normal、平均0・標準偏差0.02 |
| Q/K/V投影 | Xavier Uniform |
| Attention出力投影 | Xavier Uniform |
| SwiGLU Gate / Up | Xavier Uniform |
| SwiGLU Down | Transformer向け縮小初期化 |
| RMSNormスケール | Ones |
| Bias | 初期モデルでは使用しない |

Down投影の標準偏差は次を基準とする。

\[
\sigma=\frac{0.02}{\sqrt{2N}}
\]

ここで \(N\) はTransformer Block数。初期モデルの \(N=2\) では標準偏差0.01となる。

EmbeddingとLM Headは同一Parameterを共有し、初期化・保存を一度だけ行う。

## 12.6.3 数値検証

- 初期化後、Forward、Backward、Optimizer更新後に必要な有限値検査を行う。
- NaNやInfinityを検出した場合、該当処理を失敗として扱う。
- 学習状態が破損した場合、復元可能なチェックポイントからの復旧を試みる。
- 初期状態の損失は、ランダムなLogitsを仮定した場合の基準値と比較する。

語彙サイズ512でほぼ一様な確率分布なら、Cross Entropyは概ね \(\ln(512)\approx6.238\) となる。これは初期動作の目安であり、すべての初期化状態で厳密に一致することを要求するものではない。

---

# 12.7 設定ファイル・モデル構築

## 12.7.1 構築手順

```text
Config Loader
  ↓
構文・Schema検証
  ↓
相互依存設定の検証
  ↓
Graph Builder
  ↓
Parameter Registry
  ↓
Initializer
  ↓
Model Validation
  ↓
Runtime登録
  ↓
READY
```

## 12.7.2 初期設定例

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

これは設定構造の例であり、Tokenizer・Training・Generationなどの完全な設定は別途定義する。

## 12.7.3 検証規則

- 必須フィールドの欠落はエラー。
- 任意フィールドはSchemaに定義された既定値だけを補完する。
- 未知フィールドは、明示的な拡張領域を除きエラーとする。
- 未対応機能を黙って別方式に置き換えない。
- 次元、Tokenizer、重み共有、Backend、Optimizerの整合性を検証する。
- 設定・重み・Tokenizerの互換性を、使用前に検証する。

---

# 12.8 エラー・キャンセル仕様

## 12.8.1 エラーコード

| Prefix | 対象 |
|---|---|
| CFG_ | Configuration |
| TNS_ | Tensor |
| GRD_ | Autograd |
| LYR_ | Layer |
| GRF_ | Model Graph |
| RUN_ | Runtime |
| TRN_ | Training |
| PST_ | Persistence |
| TOK_ | Tokenizer |
| GEN_ | Generation |
| BCK_ | Backend |
| CAN_ | Cancellation |

## 12.8.2 共通エラー構造

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

通常の終了、キャンセル、処理失敗を混同してはならない。

## 12.8.3 状態遷移

```text
CREATED → RUNNING → COMPLETED
                  ↘ FAILED

RUNNING → CANCEL_REQUESTED
        → CANCELLING
        → CANCELLED
```

終端状態から別の状態へ遷移してはならない。

## 12.8.4 安全なキャンセル

キャンセル要求は、データバッチ、Forward、Backward、Optimizer更新、Checkpoint、生成トークンなどの安全な境界で処理する。

Optimizer更新は、完了させるか、ロールバック可能な仕組みを用いて整合性を保つ。更新途中の不整合な状態で停止しない。

一時Tensor、Autograd Graph、GPU Buffer、KV Cache、Loaderなどを適切に解放する。共有Parameterなど、所有していないリソースを勝手に解放してはならない。

ログには必要な識別情報を含めるが、入力文章や学習データ、秘密情報を無条件に記録しない。

---

# 12.9 CPU / WebGPU共通仕様

## 12.9.1 基準Backend

CPU / FP32を数値的な参照実装とする。

WebGPUは任意の高速化Backendであり、CPU実装の正しさを前提として追加する。

## 12.9.2 共通Backend API

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

## 12.9.3 共通契約

- ShapeとDTypeの意味を統一する。
- ベースとなるTensor配置はRow-majorとする。
- Broadcasting、Reduction、MatMul、Softmax、Maskの意味を統一する。
- 未対応演算を暗黙に別の精度・方式へ変更しない。
- CPUフォールバックは安全性と設定を確認して行う。
- OOMやGPUデバイス喪失をエラーとして適切に扱う。

## 12.9.4 数値許容差

一般演算の初期目安を次とする。

\[
|GPU-CPU|\leq10^{-5}+10^{-4}|CPU|
\]

Reduction、MatMul、Softmax、Attention、Backwardなどは演算別の許容差を設定可能とする。

この許容差は初期基準であり、実際の検証結果に応じて演算別に確定する。

## 12.9.5 実装順序

1. CPU基本演算
2. WebGPU初期化・能力検出
3. 基本演算
4. 推論演算
5. Backward演算
6. 学習対応
7. 最適化

---

# 12.10 メモリ管理・実行計画最適化

## 12.10.1 管理対象

- Parameters
- Gradients
- Optimizer State
- Activations
- Autograd Graph
- KV Cache
- Temporary Buffers
- Backend Buffers
- Data Pipeline

## 12.10.2 MemoryManager API

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

Memory BudgetはRuntime内部の管理目標であり、OSや他アプリを含むシステム全体のメモリ使用量を保証するものではない。

`canAllocate()`が成功しても、実際の割り当て成功は保証されない。

## 12.10.3 バッファ再利用

- Autogradが必要とする値を上書きしない。
- 非同期GPU処理の完了前にバッファを再利用しない。
- 所有権と生存期間を検証する。
- 解放済みリソースへの参照を検出する。

## 12.10.4 Execution Plan最適化

将来の最適化対象：

- Dead Operation除去
- 定数計算
- Buffer Lifetime解析
- Buffer再利用
- Operation Fusion
- Backend選択

最適化後もGraph・Shape・型・実行結果の整合性を検証する。最適化済みPlanが不正な場合は、可能であれば未最適化Planへ戻す。

## 12.10.5 KV Cacheメモリ

概算式：

\[
M_{KV}=2LBSH\cdot \text{bytesPerElement}
\]

初期モデルでは、Layer数2、Batch 1、Sequence 128、Hidden 64、FP32の場合、KVデータ本体は128 KiBとなる。

これは管理情報やアラインメントなどを含まない概算値である。

## 12.10.6 Profiling

以下を計測対象とする。

- Wall Time
- Compute Time
- Transfer Time
- Initialization Time
- Allocation Count
- Peak Memory
- Tensor Count
- Tokens/sec
- Samples/sec
- Cache Hit Rate

非同期GPU処理の計測では、同期の有無を明確にして測定する。

---

# 12.11 テスト・再現性

## 12.11.1 テスト階層

| 階層 | 対象 |
|---|---|
| T0 | Unit Test |
| T1 | Gradient Test |
| T2 | Layer Test |
| T3 | Graph Test |
| T4 | Integration Test |
| T5 | System Test |
| T6 | Regression Test |

テスト状態は`passed`、`failed`、`skipped`、`error`に区別する。Skippedは合格として数えない。

## 12.11.2 Tensor検証例

\[
A=
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix},
\quad
B=
\begin{bmatrix}
5&6\\
7&8
\end{bmatrix}
\]

期待される積：

\[
AB=
\begin{bmatrix}
19&22\\
43&50
\end{bmatrix}
\]

## 12.11.3 Gradient Test

中心差分を利用する。

\[
f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}
\]

ReLUの微分不連続点など、有限差分検証に不適切な点を避ける。

## 12.11.4 Transformer固有テスト

- Causal Maskによる未来情報の遮断
- RoPEの適用先がQ・Kであること
- KV Cacheあり・なしの出力比較
- EmbeddingとLM Headの重み共有
- 共有Parameterへの勾配累積

## 12.11.5 学習・保存テスト

- Cross Entropyとラベルマスク
- Gradientの有限値検証
- Optimizer更新
- Gradient Clipping
- キャンセル処理
- Checkpointからの復元
- モデル保存・読込
- 破損・互換性不一致の検出

学習損失がすべてのデータセットで単調減少することは要求しない。

## 12.11.6 再現性

乱数系列を以下のように分離する。

- Initialization
- Dropout
- Data Shuffle
- Generation
- Data Augmentation

Seedだけでは、処理順序やBackendが異なる場合の完全一致を保証しない。CPUとGPUの比較では、定めた許容差に基づいて検証する。

利用できないBackendのテストはSkippedとして報告する。

---

# 12.12 公開API・モジュール境界

## 12.12.1 KNA共通構成

従来の8モジュールを維持し、共通基盤としてCoreを追加する。

```text
Core
├── Errors
├── Configuration
├── Random
└── Utilities

KNA
├── Tensor Engine
├── Autograd
├── Layer Library
├── Model Graph
├── Runtime
├── Training System
├── Persistence
└── Quality Assurance
```

## 12.12.2 依存関係

概念上の主要依存は次のとおり。

```text
Core
  ↓
Tensor Engine
  ↓
Autograd
  ↓
Layer Library
  ↓
Model Graph
  ↓
Runtime
  ↓
Training System
```

Persistenceは内部Layer実装から独立させ、Quality Assuranceは各モジュールを検証する。

循環依存は禁止する。上図は主要な依存方向を示すものであり、すべての補助依存を網羅するものではない。

## 12.12.3 API分類

各APIを次のいずれかに分類する。

- Public
- Internal
- Experimental
- Deprecated
- Extension

Experimental APIは互換性保証の対象外であることを明示する。Deprecated APIには廃止予定と移行方法を記録する。

## 12.12.4 API例

```js
Tensor.fromArray(data, shape);

ops.matmul(a, b);

autograd.backward(loss);

const layer = new Linear(config);

runtime.compile(modelGraph);
runtime.execute(model, inputs);

trainer.trainStep(batch);

persistence.saveModel(model, destination);
persistence.loadModel(source);

persistence.saveCheckpoint(trainer, destination);
persistence.restoreCheckpoint(trainer, source);
```

これらは公開APIの方向性を示す例であり、各関数の厳密な引数型・戻り値・非同期契約は個別API仕様で確定する。

## 12.12.5 バージョン管理

次のバージョンを独立管理する。

- Framework API
- Module
- Model Schema
- Weight Format
- Checkpoint Format
- Extension API

互換性のある変更・互換性を壊す変更を区別し、Semantic Versioningを基準とする。

拡張機能は明示的に登録し、既存機能を黙って上書きしない。

## 12.12.6 ローカル実行

JavaScriptを初期実装言語とし、外部サービスや必須のビルド環境に依存しない構成を優先する。

`file://`環境でES Modulesなどの制約が問題になる場合に備え、単一バンドル方式やローカルHTTPサーバーを選択肢として維持する。

---

# 12.13 セキュリティ・信頼境界

## 12.13.1 信頼対象

以下はすべて、検証前には信頼しない。

- 外部設定ファイル
- モデルパッケージ
- 学習データ
- Checkpoint
- 拡張モジュール
- Tensorの内部状態

内部生成データであっても、不具合による破損を想定する。

## 12.13.2 設定検証

設定ファイルに対して次を検証する。

- 構文
- Schema
- データサイズとネスト深度
- 数値範囲
- 相互依存設定
- Backend能力
- 未知フィールド

## 12.13.3 Tensor安全性

- Shapeの各次元は許容された非負整数であること。
- 要素数の積が安全な整数範囲を超えないこと。
- DTypeとバイト数が一致すること。
- メモリ上限を検証すること。
- 割り当て失敗を適切に処理すること。
- スカラーTensorと空Tensorの意味を明確に定義すること。

## 12.13.4 モデルパッケージ

パッケージのManifest、ファイルサイズ、チェックサム、Shape、DType、Offset、Byte Length、重み一覧の整合性を検証する。

チェックサムは破損検出に有効だが、作成者の真正性を保証しない。

## 12.13.5 コードとデータの分離

- 保存済みモデルを`eval`などで実行しない。
- 未知のLayer IDは明示的なエラーとする。
- 拡張コードは明示的な登録を必要とする。
- 外部拡張を無条件で実行しない。

API制限だけで拡張コードが完全に隔離されるとはみなさない。強い分離が必要な場合はWorkerやホスト側の制御を検討する。

## 12.13.6 学習データ

Token IDの範囲、特殊トークン、ラベル形状、マスク、語彙サイズ、Tokenizerのバージョンを検証する。

## 12.13.7 数値状態

NaN、Infinity、不正な確率分布、ゼロ除算、勾配・Optimizer状態の破損を検出する。

検証強度はNormal、Debug、Strictなどのモードで調整可能とする。

## 12.13.8 セキュリティテスト

以下を試験対象とする。

- 極端に大きいShape
- DType不一致
- 整数オーバーフロー
- ファイルサイズ不一致
- チェックサム不一致
- 未知Layer
- 不正Token ID
- 破損Checkpoint
- メモリ割り当て失敗
- 非有限値を含む学習状態
- 未登録拡張の実行要求

---

# 12.14 最終統合・実装開始条件・完成判定

本章では、Part 12までに定義した仕様を統合し、実装開始条件・完成条件・リリース条件を確立する。

## 12.14.1 全仕様の統合基準

### 目的

全仕様を矛盾なく適用できる単一の基準へ統合する。

### 仕様の優先順位

| 優先度 | 仕様 |
|---:|---|
| 1 | 安全性・データ保全 |
| 2 | 共通インターフェース契約 |
| 3 | KNA全体の基本仕様 |
| 4 | Rikka固有仕様 |
| 5 | 実装方式 |
| 6 | 性能最適化 |

上位仕様を理由として、別の上位仕様への違反を正当化してはならない。同じ優先度の仕様が矛盾した場合、独断で選択せず変更記録を作成する。

### 規範表現

- **MUST：** 必須
- **MUST NOT：** 禁止
- **SHOULD：** 原則として推奨
- **MAY：** 任意
- **TBD：** 未確定

要求の強さが明示されていない項目は、仕様確定時に分類する。

### KNAとRikkaの分離

KNA共通仕様はTensor、Autograd、Layer、Graph、Runtime、Training、Persistence、QAの契約を定義する。

Rikka固有仕様はDecoder-only Transformer、RMSNorm、RoPE、Causal Attention、SwiGLU、初期モデル設定などを定義する。

Rikka固有の設計をKNA全体の唯一の実装方式にしてはならない。

### 仕様変更

仕様変更時には、変更理由、影響範囲、互換性、移行方法、必要なテストを記録する。確定済み仕様を黙って変更しない。

### 未確定事項

未確定事項を以下に分類する。

| 分類 | 処理 |
|---|---|
| 実装阻止事項 | 対象フェーズ開始前に決定 |
| 後続対応事項 | 必要なフェーズまで保留可能 |
| 実験事項 | 仮説・測定方法・採用条件を記録 |

暫定方式と確定仕様を混同してはならない。

### 統合チェック

- KNA共通仕様とRikka固有仕様が区別されている。
- モジュール間の公開インターフェースに矛盾がない。
- エラー、キャンセル、メモリ管理の規則が共通化されている。
- 設定、Tokenizer、重みの整合性を検証できる。
- CPUを数値的な参照実装として扱える。
- 未確定事項が分類されている。
- 仕様変更の履歴を追跡できる。
- 最適化によって演算の意味や学習条件が黙って変わらない。

**完了条件：** 仕様間の矛盾を検出・解決でき、未確定事項が分類され、各実装フェーズが参照する優先順位が明確になっていること。

---

# 付録A：モデルパッケージの基本構成

```text
kinopi-model/
├── manifest.json
├── config.json
├── weights/
│   ├── index.json
│   ├── shard-00000.bin
│   └── shard-00001.bin
├── training/
│   └── checkpoint.json
└── runtime/
    └── profile.json
```

Manifestはバージョン、Model ID、ファイル一覧、サイズ、SHA-256などを保持する。

Weight IndexはParameter名、Shape、DType、Little Endian形式、ファイル名、Offset、Byte Lengthを保持する。

Checkpointは学習Step、Epoch、Optimizer状態、Scheduler状態、RNG状態、データ位置、シャッフル順序、重みを復元するための情報を保持する。

保存は一時世代への書き込み、検証、確定の順で行い、既存の有効なCheckpointを不完全な保存で破壊しない。

# 付録B：基本不変条件

1. TensorのShape・DType・メモリ配置は演算契約と一致する。
2. Autogradは必要な勾配計算情報を失わない。
3. 共有Parameterは一つの実体として管理する。
4. Causal Attentionは未来情報を参照しない。
5. 保存・復元時に設定と重みの整合性を検証する。
6. 不正な入力や未対応機能を黙って受け入れない。
7. キャンセル・失敗後もリソースの所有権を維持する。
8. テスト未実施をテスト合格として扱わない。
9. Backendの変更によって演算の意味を黙って変更しない。
10. 性能最適化よりも正しさとデータ保全を優先する。

# 付録C：文書状態と残作業

この文書はPart 12.1〜12.14.1の統合版であり、Part 12.14.2以降は未作成である。

以下は後続節で具体化する。

- 実装開始に必要な条件
- 初期実装の対象範囲
- フェーズごとの移行条件
- 完成判定基準
- KNA本体の完成条件
- Rikka初期モデルの完成条件
- WebGPU対応の完成条件
- 未実装・未検証機能の管理
- リリース・バージョン管理
- 最終受け入れ判定
- 実装担当への引き継ぎ

これらが確定するまでは、文書全体を最終確定版とは扱わない。

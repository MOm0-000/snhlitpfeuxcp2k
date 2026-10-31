# 基于 Transformer 的神经机器翻译建模

> 《自然语言处理 C 实验 II》课程实验代码。

本项目基于 PaddlePaddle 动态图实现 Transformer 英→中机器翻译，在 WMT18 中英平行语料上完成基线模型训练、newstest2017/newstest2018 预测、BLEU 评估、3.6 超参数实验以及 3.7 模型结构改进实验。

当前仓库以课程实验为主，不做额外工程化拆分；目录和脚本名称尽量与实际实验过程保持一致。

---

## 运行环境

本项目实际使用的实验环境为：

- Ubuntu Linux
- Python 3.10.21
- PaddlePaddle-GPU 3.3.0
- CUDA 13.0（PaddlePaddle build）
- NVIDIA GeForce RTX 5070 Laptop GPU，8 GB 显存

Python 依赖见 `requirements.txt`，可在虚拟环境中执行：

```bash
python -m pip install -r requirements.txt
```

---

## 目录结构

```text
.
├── main.py                         # 统一入口：解析 YAML/命令行参数并分发训练或预测
├── train.py                        # Transformer 训练逻辑
├── predict.py                      # Beam Search 预测逻辑
├── transformer.py                  # Transformer 模型及 3.7 结构改进
├── reader.py                       # 数据读取、组批与词表处理
├── transformer.yaml                # 当前课程基线默认配置
├── requirements.txt                # Python 运行依赖
│
├── run_hparam_experiments.py       # 3.6 单因素超参数实验
├── stat.py                         # 3.6 实验结果统计
├── run_model_improvements.py       # 3.7 模型结构改进实验
├── stat_model_improvements.py      # 3.7 实验结果统计
├── test_transformer_37.py          # 3.7 模型改进 smoke test
│
├── hparam_runs/                    # 3.6 实验记录、预测结果与统计结果
├── improvement_runs/               # 3.7 实验记录、预测结果与统计结果
├── output_file/                    # 3.3 基线模型的 2017/2018 预测结果
│
├── wmt18_zhen_data_bpe3/           # WMT18 中英训练/测试数据与词表
├── mosesdecoder/                   # multi-bleu.perl
└── utils/                          # 配置解析与环境检查
```

`saved_models/` 以及各实验目录中的 Paddle 权重文件保存在本地，不上传 GitHub。

---

## 1. 模型与数据

### 1.1 基线模型

课程基线采用 Transformer Base 配置：

| 参数 | 默认值 |
| --- | ---: |
| Encoder / Decoder 层数 | 6 / 6 |
| `d_model` | 512 |
| Attention heads | 8 |
| FFN hidden size | 2048 |
| Label smoothing | 0.1 |
| Dropout | 0.1 |
| Batch size | 4096 tokens |
| Epoch | 10 |
| Learning rate | 2.0 |
| Warmup steps | 8000 |
| Beam size | 5 |
| `max_length` | 256 |
| `max_out_len` | 255 |
| `save_step` | 5000 |
| `print_step` | 100 |

当前实现使用 Pre-LN Transformer；源语言与目标语言共用词表，并共享词嵌入/输出投影权重。

`transformer.yaml` 已与当前课程实验对齐。3.6 和 3.7 的实验脚本通过命令行参数覆盖其中的部分默认值，因此 YAML 本身始终保持“基线配置”。

### 1.2 数据

数据目录：

```text
wmt18_zhen_data_bpe3/
```

主要文件：

| 文件 | 用途 |
| --- | --- |
| `train.tok.clean.bpe.32000.en-zh` | WMT18 中英训练集 |
| `newstest2017.tok.bpe.32000.en-zh` | newstest2017 预测输入 |
| `newstest2018.tok.bpe.32000.en-zh` | newstest2018 预测输入 |
| `newstest2017.tok.zh` | 2017 BLEU 参考译文 |
| `newstest2018.tok.zh` | 2018 BLEU 参考译文 |
| `vocab_all.bpe.32000` | 中英共享 BPE 词表 |
| `gen_data.sh` | 数据下载与预处理脚本 |

---

## 2. 3.3 基线模型训练

由于 `transformer.yaml` 已经保存课程基线默认配置，正常训练可直接执行：

```bash
python3 -u main.py --do_train True
```

主要默认参数包括：

```text
epoch = 10
batch_size = 4096
learning_rate = 2.0
warmup_steps = 8000
print_step = 100
save_step = 5000
save_model_path = saved_models
```

训练过程中会按以下形式保存模型：

```text
saved_models/
├── step_5000/
├── step_10000/
├── step_15000/
└── step_final/
```

其中模型参数和优化器状态分别为：

```text
transformer.pdparams
transformer.pdopt
```

这些文件体积较大，已通过 `.gitignore` 排除，不上传 GitHub。

---

## 3. 3.4 基线模型预测

基线模型使用：

```text
saved_models/step_final/
```

分别预测 newstest2017 和 newstest2018。

### 3.1 newstest2017

```bash
python3 -u main.py \
  --do_predict True \
  --predict_file wmt18_zhen_data_bpe3/newstest2017.tok.bpe.32000.en-zh \
  --batch_size 32 \
  --init_from_params saved_models/step_final \
  --beam_size 5 \
  --max_out_len 255 \
  --output_file output_file/predict_2017.txt
```

去除 BPE 标记：

```bash
sed -r 's/(@@ )|(@@ ?$)//g' \
  output_file/predict_2017.txt \
  > output_file/predict_2017.tok.txt
```

### 3.2 newstest2018

```bash
python3 -u main.py \
  --do_predict True \
  --predict_file wmt18_zhen_data_bpe3/newstest2018.tok.bpe.32000.en-zh \
  --batch_size 32 \
  --init_from_params saved_models/step_final \
  --beam_size 5 \
  --max_out_len 255 \
  --output_file output_file/predict_2018.txt
```

去除 BPE 标记：

```bash
sed -r 's/(@@ )|(@@ ?$)//g' \
  output_file/predict_2018.txt \
  > output_file/predict_2018.tok.txt
```

当前仓库已保存：

```text
output_file/
├── predict_2017.txt
├── predict_2017.tok.txt
├── predict_2018.txt
└── predict_2018.tok.txt
```

---

## 4. 3.5 BLEU 评估

使用 Moses `multi-bleu.perl` 评估去除 BPE 后的预测结果。

### 4.1 newstest2017

```bash
perl mosesdecoder/scripts/generic/multi-bleu.perl \
  wmt18_zhen_data_bpe3/newstest2017.tok.zh \
  < output_file/predict_2017.tok.txt
```

### 4.2 newstest2018

```bash
perl mosesdecoder/scripts/generic/multi-bleu.perl \
  wmt18_zhen_data_bpe3/newstest2018.tok.zh \
  < output_file/predict_2018.tok.txt
```

基线预测文件已经保存在仓库中；BLEU 数值以本地使用以上命令重新计算的结果为准。

> 注意：`max_length=256` 时，超过最大长度的样本会在读取阶段被过滤，因此预测输出行数可能少于原始测试集行数。

---

## 5. 3.6 Transformer 翻译性能分析

3.6 使用严格的单因素实验：除当前考察参数外，其余训练设置保持基线不变。

运行：

```bash
python3 run_hparam_experiments.py
```

统计：

```bash
python3 stat.py
```

实验记录保存在：

```text
hparam_runs/
```

其中：

```text
results.csv
summary_report.txt
summary_table.csv
training_points.csv
```

保存完整汇总结果；各实验子目录保存配置、训练日志和 2017/2018 预测结果。

### 5.1 实验结果

| 实验 | 改动 | BLEU-2017 | BLEU-2018 |
| --- | --- | ---: | ---: |
| `dropout_020` | dropout = 0.20 | 10.71 | 11.19 |
| `dropout_030` | dropout = 0.30 | 10.55 | 11.23 |
| `label_smooth_005` | label smoothing = 0.05 | 10.13 | 11.16 |
| `label_smooth_020` | label smoothing = 0.20 | 10.82 | 11.14 |
| `lr_150` | learning rate = 1.5 | **10.86** | **11.23** |
| `warmup_12000` | warmup = 12000 | 10.73 | 10.80 |
| `ffn_1024` | FFN hidden = 1024 | 10.74 | 10.93 |
| `ffn_1536` | FFN hidden = 1536 | 10.49 | 10.83 |
| `beam_1` | beam size = 1 | 10.12 | 10.21 |
| `beam_3` | beam size = 3 | 10.36 | 10.98 |
| `beam_8` | beam size = 8 | 10.11 | 10.71 |

按照 newstest2017 作为验证集选择，3.6 中表现最好的单因素配置为：

```text
learning_rate = 1.5
BLEU-2017 = 10.86
BLEU-2018 = 11.23
```

因此 `lr_150` 被作为 3.7 模型结构改进实验的 control。

---

## 6. 3.7 Transformer 模型改进

3.7 在 3.6 的最佳训练设置 `learning_rate=1.5` 基础上固定普通训练超参数，只改变模型结构。

运行：

```bash
python3 test_transformer_37.py
python3 run_model_improvements.py
```

统计：

```bash
python3 stat_model_improvements.py
```

实验记录保存在：

```text
improvement_runs/
```

当前正式比较的结构包括：

- Scheduled DropHead
- SwiGLU FFN
- Relative Position Bias

`transformer.yaml` 中保留这些结构对应的配置键，但默认：

```yaml
attention_variant: "normal"
ffn_variant: "relu"
```

因此默认运行仍然是普通 Transformer，3.7 脚本通过命令行覆盖参数启用具体改进。

### 6.1 实验结果

| 实验 | 结构 | BLEU-2017 | BLEU-2018 | 相对 control |
| --- | --- | ---: | ---: | --- |
| `control_lr150` | 3.6 最佳配置 | 10.86 | 11.23 | — |
| `drophead_010` | DropHead, p=0.10 | 10.11 | 10.68 | -0.75 / -0.55 |
| `drophead_005` | DropHead, p=0.05 | 10.58 | 11.05 | -0.28 / -0.18 |
| `swiglu_1360` | SwiGLU, hidden=1360 | 10.44 | 10.79 | -0.42 / -0.44 |
| `swiglu_1536` | SwiGLU, hidden=1536 | 10.45 | 10.67 | -0.41 / -0.56 |
| `relative_bias_16` | Relative Position Bias, max distance=16 | **11.11** | **11.53** | **+0.25 / +0.30** |

最终 3.7 最佳结构为：

```text
relative_bias_16
BLEU-2017 = 11.11
BLEU-2018 = 11.53
```

相对于 `control_lr150`：

```text
BLEU-2017: +0.25
BLEU-2018: +0.30
```

Relative Position Bias 仅作用于 Encoder / Decoder 的 self-attention；Encoder-Decoder cross-attention 保持普通 attention。

---

## 7. 配置说明

`transformer.yaml` 的职责是保存当前课程的**基线默认配置**。

### 7.1 3.6

3.6 需要调节的普通超参数，例如：

```text
learning_rate
warmup_steps
label_smooth_eps
prepostprocess_dropout
d_inner_hid
beam_size
```

在 YAML 中保持基线值，`run_hparam_experiments.py` 在每个实验中只覆盖一个参数。

### 7.2 3.7

模型改进相关配置：

```text
attention_variant
ffn_variant
drophead_rate
drophead_schedule
drophead_warmup_steps
drophead_total_steps
swiglu_inner_hid
max_relative_position
```

其中默认：

```text
attention_variant = normal
ffn_variant = relu
```

因此不会影响 3.3 基线。

---

## 8. 主要文件关系

```text
transformer.yaml
      │
      ▼
   main.py
   ├── do_train  ──> train.py ──> transformer.py
   └── do_predict ─> predict.py ─> transformer.py

3.6:
run_hparam_experiments.py
      └── 多次调用 main.py，并覆盖单个超参数
              └── hparam_runs/

3.7:
run_model_improvements.py
      └── 多次调用 main.py，并切换模型结构
              └── improvement_runs/
```

---

## 9. Git 仓库说明

GitHub 主要保存：

- 源代码；
- YAML 配置；
- WMT18 数据；
- 预测结果；
- 训练日志；
- BLEU 日志；
- 实验配置和统计 CSV/TXT。

以下内容只保存在本地：

- `saved_models/`
- `*.pdparams`
- `*.pdopt`
- Python 缓存
- `BACKUP/`
- `TRASH/`

因此重新 clone 仓库后不会包含训练权重。

---

## 10. 来源

本项目由早期 PaddlePaddle Transformer 示例迁移到当前 PaddlePaddle 动态图版本，并在课程实验基础上增加 3.6 超参数实验和 3.7 模型结构改进。

基础 Transformer 实现参考：

1. Vaswani A, Shazeer N, Parmar N, et al. *Attention Is All You Need*. NeurIPS, 2017.
2. PaddlePaddle / PaddleNLP Transformer machine translation examples.

代码中的原始 PaddlePaddle 文件保留 Apache License 2.0 版权声明。

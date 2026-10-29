# 基于Transformer神经机器翻译建模

> 《自然语言处理C实验II》实验代码。

机器翻译（Machine Translation）是利用计算机将一种自然语言（源语言）转换为另一种自然语言（目标语言）的过程，输入为源语言句子，输出为相应目标语言的句子。本项目是机器翻译领域主流模型 Transformer 的 PaddlePaddle 实现，在 WMT18 中英平行语料（约 25.24 万句对）上完成英→中翻译的模型训练、束搜索解码预测与 BLEU 评估。

## 目录结构

```text
.
├── main.py              # 统一入口：解析配置并分发训练/预测
├── train.py             # 训练脚本（动态图组网）
├── predict.py           # 预测脚本（BeamSearchDecoder 束搜索解码）
├── transformer.py       # 模型定义（paddle.nn.Layer 动态组网）
├── reader.py            # 数据读取与组批
├── transformer.yaml     # 默认配置（命令行参数可覆盖）
├── main.ipynb           # 实验主入口笔记本
├── utils/               # configure.py（配置合并）、check.py（环境检查）
├── mosesdecoder/        # BLEU 评估脚本 multi-bleu.perl
├── wmt18_zhen_data_bpe3/# 训练/测试数据与词表
└── output_file/         # 预设翻译结果输出目录
```

## 模型介绍

Transformer 是论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 中提出的用以完成机器翻译等序列到序列（Seq2Seq）学习任务的一种全新网络结构，其完全使用注意力（Attention）机制来实现序列到序列的建模[1]。

相较于此前 Seq2Seq 模型中广泛使用的循环神经网络（Recurrent Neural Network, RNN），使用 Self-Attention 进行输入序列到输出序列的变换主要具有以下优势：

- 计算复杂度小：特征维度为 d、长度为 n 的序列，在 RNN 中计算复杂度为 `O(n * d * d)`（n 个时间步，每个时间步计算 d 维的矩阵向量乘法），在 Self-Attention 中计算复杂度为 `O(n * n * d)`（n 个时间步两两计算 d 维的向量点积），n 通常要小于 d。
- 计算并行度高：RNN 中当前时间步的计算要依赖前一个时间步的计算结果；Self-Attention 中各时间步的计算只依赖输入、不依赖之前时间步输出，各时间步可以完全并行。
- 容易学习长程依赖（long-range dependencies）：RNN 中相距为 n 的两个位置间的关联需要 n 步才能建立；Self-Attention 中任何两个位置都直接相连，路径越短信号传播越容易。

### 模型特点

Transformer 中的 Encoder 由若干相同的 layer 堆叠组成，每个 layer 主要由多头注意力（Multi-Head Attention）和全连接的前馈（Feed-Forward）网络这两个 sub-layer 构成：

- Multi-Head Attention 在这里用于实现 Self-Attention。相比于简单的 Attention 机制，其将输入进行多路线性变换后分别计算 Attention 的结果，并将所有结果拼接后再次进行线性变换作为输出。其中 Attention 使用的是点积（Dot-Product），并在点积后进行了 scale 的处理以避免因点积结果过大进入 softmax 的饱和区域。
- Feed-Forward 网络对序列中的每个位置进行相同的计算（Position-wise），采用两次线性变换中间加以 ReLU 激活的结构。

此外，每个 sub-layer 还施以 Residual Connection [3] 和 Layer Normalization [4] 来促进梯度传播和模型收敛。

Decoder 具有和 Encoder 类似的结构，只是相比于组成 Encoder 的 layer，在组成 Decoder 的 layer 中还多了一个 Multi-Head Attention 的 sub-layer 来实现对 Encoder 输出的 Attention，这个 Encoder-Decoder Attention 在其他 Seq2Seq 模型中也是存在的。

### 本项目实现要点

- 结构为 Transformer base：编码器/解码器各 6 层、8 头、d_model=512、前馈层 2048、标签平滑 0.1；采用子层前归一化（Pre-LN）结构。
- 源语言与目标语言共用同一词表，且源词嵌入、目标词嵌入与输出投影三层共享权重（weight sharing）。
- 训练采用按词元数组织批次（一个 batch 内最多 4096 个 token，含填充），Adam 优化器配合 Noam 学习率（warmup 8000 步）。
- 预测使用 `paddle.nn.BeamSearchDecoder` 束搜索解码（束宽 5），增量 K/V 缓存由框架按束自动重排。

## 数据说明

数据位于 `wmt18_zhen_data_bpe3/`，是来自 WMT18 新闻评论文集（news-commentary v13）中英平行语料的预处理产物：

| 文件                                    | 说明                                                                          |
| --------------------------------------- | ----------------------------------------------------------------------------- |
| `train.tok.clean.bpe.32000.en-zh`       | 训练集，252,385 句对，制表符分隔（左列英文、右列中文）                        |
| `newstest2017/2018.tok.bpe.32000.en-zh` | 测试集（预测输入），2,001 / 3,981 句                                          |
| `newstest2017/2018.tok.zh`              | 测试集参考译文（BLEU 评估用，仅分词）                                         |
| `vocab_all.bpe.32000`                   | 词表，一行一个词元、行号即编号，共 38,814 行（前 3 行为 `<s>`/`<e>`/`<unk>`） |
| `gen_data.sh`                           | 数据预处理流水线脚本（下载、分词、清洗、BPE 编码）                            |

文件名中 `tok` 表示已分词（tokenization），`clean` 表示已按句长 1–80 词清洗，`bpe.32000` 表示已使用 [BPE（Byte Pair Encoding）](https://arxiv.org/pdf/1508.07909) 子词分词算法以 32000 次合并编码（`@@` 为子词连接符），`en-zh` 表示中英平行句对[5]。

## 快速开始

### 环境

依赖 PaddlePaddle 2.0 及以上版本（本项目已在当前最新版本 paddle-gpu 3.3.1 动态图下验证），需 GPU，实验要求的超参数需约16GB显存。所有命令在项目根目录执行，默认配置见 `transformer.yaml`，命令行参数可覆盖 yaml 中的同名项。

### 模型训练

```bash
python3 -u main.py \
  --do_train True \
  --epoch 10 \
  --src_vocab_fpath wmt18_zhen_data_bpe3/vocab_all.bpe.32000 \
  --trg_vocab_fpath wmt18_zhen_data_bpe3/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --training_file wmt18_zhen_data_bpe3/train.tok.clean.bpe.32000.en-zh \
  --batch_size 4096 \
  --save_step 5000
```

主要参数含义（完整默认值及注释见 `transformer.yaml`）：

- `--epoch`：训练轮数；`--batch_size`：一个 batch 内最多的 token 数目（含填充，默认 4096）。
- `--training_file`：训练句对文件；`--src_vocab_fpath`/`--trg_vocab_fpath`：源/目标词表（本项目共用同一词表）；`--special_token`：词表中的起始、结束、未登录符号。
- `--print_step`（默认 100）：每隔多少迭代打印一行训练日志（avg loss、normalized loss、ppl、speed）。
- `--save_step`（默认 100000）：每隔多少迭代将参数保存到 `saved_models/step_X/`；训练正常结束后另存 `saved_models/step_final/`（含 `transformer.pdparams` 与 `transformer.pdopt`）。
- `--init_from_checkpoint`：从检查点恢复训练（含优化器状态）；`--init_from_pretrain_model`：从已有参数热启动（二者互斥）。
- `--max_iter`：限制总迭代数（基准测试用，提前退出不保存 step_final）。

参考耗时：在单卡 RTX 4060 Ti 16G 和实验推荐使用的 AI Studio 平台 V100 16G 上的速度均 约 3 步/秒，10 个 epoch 共约 1.84 万迭代、约 100 分钟完成实验所需训练。

### 模型预测

```bash
python3 -u main.py \
  --do_predict True \
  --src_vocab_fpath wmt18_zhen_data_bpe3/vocab_all.bpe.32000 \
  --trg_vocab_fpath wmt18_zhen_data_bpe3/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --predict_file wmt18_zhen_data_bpe3/newstest2018.tok.bpe.32000.en-zh \
  --batch_size 32 \
  --init_from_params saved_models/step_final/ \
  --beam_size 5 \
  --max_out_len 255 \
  --output_file output_file/predict_2018.txt
```

- `--predict_file`：待翻译的双列句对文件（仅使用源语言列）；`--init_from_params`：训练产物所在目录；`--output_file`：译文输出文件，每个源句一行、保持输入顺序。
- `--batch_size`：预测时一个 batch 的句子数（默认按句组批）；`--beam_size`：束宽；`--max_out_len`：最大解码步数。
- 预测须与训练使用同一词表；预测目前仅支持单卡（评估依赖输出行序）。

### 模型评估

预测结果是 BPE 表示的数据，需还原成 tokenize 后的数据才能正确评估（BLEU 是翻译任务常用的自动评估指标）：

```bash
# 还原 predict_2018.txt 中的预测结果为 tokenize 后的数据
sed -r 's/(@@ )|(@@ ?$)//g' output_file/predict_2018.txt > output_file/predict_2018.tok.txt
# 以 newstest2018 为例计算 BLEU（参考译文取仅分词的 .tok.zh）
perl mosesdecoder/scripts/generic/multi-bleu.perl wmt18_zhen_data_bpe3/newstest2018.tok.zh < output_file/predict_2018.tok.txt
```

为方便参考，下面提供按上述命令训练后测得的基线结果，**切勿照抄**：

```log
BLEU = 4.46, 23.3/5.7/2.5/1.2 (BP=1.000, ratio=1.087, hyp_len=97832, ref_len=90019)
```

## 常见问题

**Q: 预测结果中样本数少于输入的样本数是什么原因？**
A: 若样本中最大长度超过 `transformer.yaml` 中 `max_length` 的默认设置（256），超长句对会被直接过滤删除，不会出现在预测输出中。例如 newstest2018 的 3,981 句中有 2 句（源侧 293/279 词元）超限被删，输出为 3,979 行。

**Q: 预测时句子长度超过训练时的最大长度怎么办？**
A: 训练时 `max_length` 的设置决定了位置编码表的大小（建模时按 `max_length + 1` 行构建）。超长输入会被上述过滤机制删除；如需处理更长句子，需增大 `max_length` 并重新训练。

## 来源与许可

本项目文档于2026年10月7日从旧实验代码（paddlepaddle1.8版本，python3.7）更新，旧实验项目地址为https://aistudio.baidu.com/aistudio/projectdetail/1231155 迁移前后行为和基线结果均实现对齐复现。

本项目文档参考自 [PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) 机器翻译 Transformer 示例文档（Apache License 2.0）删改整理；代码沿袭 PaddlePaddle models 仓库（Apache License 2.0）。

## 参考文献

1. Vaswani A, Shazeer N, Parmar N, et al. [Attention is all you need](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)[C]//Advances in Neural Information Processing Systems. 2017: 6000-6010.
2. Devlin J, Chang M W, Lee K, et al. [Bert: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/abs/1810.04805)[J]. arXiv preprint arXiv:1810.04805, 2018.
3. He K, Zhang X, Ren S, et al. [Deep residual learning for image recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.
4. Ba J L, Kiros J R, Hinton G E. [Layer normalization](https://arxiv.org/pdf/1607.06450.pdf)[J]. arXiv preprint arXiv:1607.06450, 2016.
5. Sennrich R, Haddow B, Birch A. [Neural machine translation of rare words with subword units](https://arxiv.org/pdf/1508.07909)[J]. arXiv preprint arXiv:1508.07909, 2015.

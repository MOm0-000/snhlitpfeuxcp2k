# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time

import numpy as np
import paddle
import paddle.nn.functional as F

import reader
from transformer import TransformerModel
from utils.configure import PDConfig
from utils.check import check_gpu, check_version


def _get_optional(args, name, default):
    """
    读取可选配置项（带默认值兜底）。

    utils/configure.py 中的 PDConfig 在键不存在时抛出的是 Warning 异常而不是
    AttributeError，标准的 getattr(args, name, default) 兜底不会生效，因此这里
    统一用 try/except 包一层。这样即使旧版 transformer.yaml 中没有新增的
    rdrop_alpha / accum_steps，也能按默认值正常训练。
    """
    try:
        value = getattr(args, name)
    except Exception:
        return default
    return default if value is None else value


def compute_kl_loss(logits_p, logits_q, pad_mask):
    """
    【模型改进核心代码】R-Drop 的双向 KL 一致性正则项。

    同一个 batch 在 batch 维复制拼接后只做一次前向，两半数据在 dropout 层
    各自独立采样掩码，因此 logits_p / logits_q 等价于两个不同子模型的输出。
    R-Drop 用双向 KL 约束这两个分布相互一致：
        kl = 0.5 * ( KL(q || p) + KL(p || q) )

    参数:
        logits_p: [B, T, V]，第一个子模型的词表 logits
        logits_q: [B, T, V]，第二个子模型的词表 logits
        pad_mask: [B, T] 的 bool 张量，True 表示 padding（无效 token）
    返回:
        标量，有效 token 上的双向 KL 平均值
    """
    # 1) 分别计算 log 概率与概率分布
    p_log_softmax = F.log_softmax(logits_p, axis=-1)
    q_log_softmax = F.log_softmax(logits_q, axis=-1)
    p_softmax = F.softmax(logits_p, axis=-1)
    q_softmax = F.softmax(logits_q, axis=-1)
    # 数值兜底：32000 维 softmax 在极端 logits 下可能下溢为精确的 0，而
    # F.kl_div 内部会计算 log(label)，label=0 时得到 -inf 进而产生 NaN。
    # 这里给概率加一个极小下限（1e-12 远小于正常概率量级，对 KL 数值无实质影响）。
    p_softmax = paddle.clip(p_softmax, min=1e-12)
    q_softmax = paddle.clip(q_softmax, min=1e-12)

    # 2) 逐元素双向 KL：F.kl_div(input, label) = label * (log(label) - input)
    #    input 传 log 概率、label 传概率，即得两个方向的 KL 散度
    kl_qp = F.kl_div(p_log_softmax, q_softmax, reduction='none')  # KL(q || p)
    kl_pq = F.kl_div(q_log_softmax, p_softmax, reduction='none')  # KL(p || q)

    # 3) 在词表维（axis=-1）求和，得到形状 [B, T] 的逐 token KL，双向取平均
    kl_token = 0.5 * paddle.sum(kl_qp + kl_pq, axis=-1)

    # 4) 只用有效 token 求平均。这里必须先用 masked_select 筛掉 padding 再求和：
    #    - padding 位置的 logits 无意义，softmax 可能下溢为 0，若写成
    #      kl_token * (1 - pad_mask) 则 0 * NaN = NaN 会污染整个 loss；
    #    - 也绝不能在全部元素上直接 sum，否则词表维（32000）参与求和会让
    #      loss 量级失控。
    valid_mask = paddle.logical_not(pad_mask)
    kl_valid = paddle.masked_select(kl_token, valid_mask)
    valid_token_num = paddle.sum(paddle.cast(valid_mask, kl_token.dtype))
    valid_token_num.stop_gradient = True
    return paddle.sum(kl_valid) / (valid_token_num + 1e-8)


def compute_nll_loss(logits, lbl_word, lbl_weight, model):
    """
    计算带标签平滑的 NLL 损失（逐行等价复现 TransformerModel.forward 内部实现）。

    transformer.py 按约束不做改动，其 forward 只返回 (sum_cost, avg_cost,
    token_num) 而不返回 logits；为了在 train.py 中拿到两个子模型的 logits 计算
    KL，这里用同一套算子重算交叉熵，保证与基线损失数值和量纲完全一致。

    参数:
        logits: [B, T, V]，词表 logits
        lbl_word: [B*T, 1]，目标 token id
        lbl_weight: [B*T, 1]，有效 token 为 1.0、padding 为 0.0
    返回:
        (sum_cost, token_num)：加权损失总和与有效 token 数
    """
    # 与 transformer.py 一致：先 reshape 成 2D 以使用 GEMM
    predict = paddle.reshape(logits, shape=[-1, logits.shape[-1]])
    if model.label_smooth_eps:
        label = F.one_hot(
            paddle.reshape(lbl_word, shape=[-1]),
            num_classes=model.trg_vocab_size).astype("float32")
        label = F.label_smooth(label, epsilon=model.label_smooth_eps)
        cost = F.softmax_with_cross_entropy(
            logits=predict, label=label, soft_label=True)
    else:
        cost = F.softmax_with_cross_entropy(
            logits=predict, label=lbl_word, soft_label=False)
    weighted_cost = paddle.multiply(cost, lbl_weight)
    sum_cost = paddle.sum(weighted_cost)
    token_num = paddle.sum(lbl_weight)
    token_num.stop_gradient = True
    return sum_cost, token_num


def forward_logits(model, src_word, src_pos, src_slf_attn_bias, trg_word,
                   trg_pos, trg_slf_attn_bias, trg_src_attn_bias):
    """
    用 TransformerModel 的公开方法复现其 forward 的前向路径，返回 [B, T, V] 的
    词表 logits（forward 本身只返回三个损失标量，拿不到 logits）。
    由于只在 train.py 侧调用，模型文件和权重结构保持零改动。
    """
    enc_output = model.encode(src_word, src_pos, src_slf_attn_bias)
    dec_output = model.decode(trg_word, trg_pos, enc_output, trg_slf_attn_bias,
                              trg_src_attn_bias)
    # 与 forward 一致：先 reshape 成 2D 再走输出投影（权重共享时为转置 matmul）
    dec_output = paddle.reshape(
        dec_output, shape=[-1, dec_output.shape[-1]])
    logits = model.output_logits(dec_output)
    return paddle.reshape(
        logits, [trg_word.shape[0], trg_word.shape[1], -1])


def do_train(args):
    if args.use_cuda:
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')

    # define the data generator
    processor = reader.DataProcessor(
        fpattern=args.training_file,
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        token_delimiter=args.token_delimiter,
        use_token_batch=args.use_token_batch,
        batch_size=args.batch_size,
        device_count=1,
        pool_size=args.pool_size,
        sort_type=args.sort_type,
        shuffle=args.shuffle,
        shuffle_batch=args.shuffle_batch,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        max_length=args.max_length,
        n_head=args.n_head)
    batch_generator = processor.data_generator(phase="train")
    args.src_vocab_size, args.trg_vocab_size, args.bos_idx, args.eos_idx, \
        args.unk_idx = processor.get_vocab_summary()

    random_seed = eval(str(args.random_seed))
    if random_seed is not None:
        paddle.seed(random_seed)

    # define the model

    model = TransformerModel.from_args(args)
    model.train()

    # define the optimizer

    scheduler = paddle.optimizer.lr.NoamDecay(
        d_model=args.d_model,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        last_epoch=0)
    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=float(args.eps),
        parameters=model.parameters())

    assert (args.init_from_checkpoint == "") or (
        args.init_from_pretrain_model == "")

    ## init from some checkpoint, to resume the previous training
    if args.init_from_checkpoint:
        model.set_state_dict(
            paddle.load(
                os.path.join(args.init_from_checkpoint,
                             "transformer.pdparams")))
        optimizer.set_state_dict(
            paddle.load(
                os.path.join(args.init_from_checkpoint, "transformer.pdopt")))
        print("finish initing model from checkpoint from %s" %
              (args.init_from_checkpoint))

    ## init from some pretrain models, to better solve the current task
    if args.init_from_pretrain_model:
        model.set_state_dict(
            paddle.load(
                os.path.join(args.init_from_pretrain_model,
                             "transformer.pdparams")))
        print("finish initing model from pretrained params from %s" %
              (args.init_from_pretrain_model))

    # the best cross-entropy value with label smoothing
    loss_normalizer = -(
        (1. - args.label_smooth_eps) * np.log(
            (1. - args.label_smooth_eps)) + args.label_smooth_eps *
        np.log(args.label_smooth_eps / (args.trg_vocab_size - 1) + 1e-20))
    # start training

    # ===== 模型改进核心代码：R-Drop 超参读取 =====
    # rdrop_alpha: KL 一致性正则项的权重，基线 BLEU 较低时从 1.0 起步
    #              （论文默认 5.0，容易与 NLL 量级失衡导致训练发散）
    # accum_steps: 梯度累积步数，accum_steps=1 即退化为原始“每个 batch 更新一次”
    rdrop_alpha = float(_get_optional(args, "rdrop_alpha", 1.0))
    accum_steps = max(1, int(_get_optional(args, "accum_steps", 1)))
    logging.info("R-Drop enabled: rdrop_alpha = %f, accum_steps = %d" %
                 (rdrop_alpha, accum_steps))

    def flush_gradients(accum_counter):
        """
        把不足 accum_steps 的残余梯度也更新一次，避免 epoch 末尾（或 max_iter
        提前退出时）最后几个 batch 的梯度被直接丢弃。
        """
        if accum_counter > 0:
            optimizer.step()
            optimizer.clear_grad()
            # NoamDecay 按“参数更新次数”计数，一次更新只前进一步
            scheduler.step()
        return 0

    step_idx = 0
    total_batch_num = 0  # this is for benchmark
    accum_counter = 0  # 梯度累积计数器：累计满 accum_steps 才真正更新参数
    for pass_id in range(args.epoch):
        batch_id = 0
        for data_inputs in batch_generator():
            if args.max_iter and total_batch_num == args.max_iter:  # this for benchmark
                flush_gradients(accum_counter)
                return

            src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos, \
                trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight = [
                    paddle.to_tensor(x) for x in data_inputs
                ]

            # ===== 模型改进核心代码：R-Drop 前向与损失 =====
            # 记录复制前的 batch 大小与目标序列长度，供后续 reshape 与拆分之用
            batch_size = trg_word.shape[0]
            trg_len = trg_word.shape[1]

            # (1) 在 batch 维复制拼接：[B, ...] -> [2B, ...]。两半数据共享同一
            #     份标签，但 dropout 掩码独立采样，等价于两个不同的子模型。
            src_word_2x = paddle.concat([src_word, src_word], axis=0)
            src_pos_2x = paddle.concat([src_pos, src_pos], axis=0)
            src_slf_attn_bias_2x = paddle.concat(
                [src_slf_attn_bias, src_slf_attn_bias], axis=0)
            trg_word_2x = paddle.concat([trg_word, trg_word], axis=0)
            trg_pos_2x = paddle.concat([trg_pos, trg_pos], axis=0)
            trg_slf_attn_bias_2x = paddle.concat(
                [trg_slf_attn_bias, trg_slf_attn_bias], axis=0)
            trg_src_attn_bias_2x = paddle.concat(
                [trg_src_attn_bias, trg_src_attn_bias], axis=0)

            # (2) 只做一次前向，得到 [2B, T, V] 的 logits
            logits_2x = forward_logits(model, src_word_2x, src_pos_2x,
                                       src_slf_attn_bias_2x, trg_word_2x,
                                       trg_pos_2x, trg_slf_attn_bias_2x,
                                       trg_src_attn_bias_2x)

            # (3) 拆回两个子模型的输出，均为 [B, T, V]
            logits1, logits2 = paddle.split(logits_2x, 2, axis=0)
            # 及时释放大张量，降低显存峰值
            del logits_2x, src_word_2x, src_pos_2x, src_slf_attn_bias_2x, \
                trg_word_2x, trg_pos_2x, trg_slf_attn_bias_2x, \
                trg_src_attn_bias_2x

            # (4) NLL：两个子模型各自计算标签平滑后的交叉熵，再取平均。
            #     sum_cost / token_num 即逐 token 平均损失，与基线 avg_cost 同量纲。
            sum_cost1, token_num = compute_nll_loss(logits1, lbl_word,
                                                    lbl_weight, model)
            sum_cost2, _ = compute_nll_loss(logits2, lbl_word, lbl_weight,
                                            model)
            sum_cost = 0.5 * (sum_cost1 + sum_cost2)
            nll = sum_cost / token_num

            # (5) KL：生成 [B, T] 的 padding 掩码（True 表示 padding）。
            #     reader 的 padding id 是 <e>，而标签末尾的真实 <e> 权重为 1，
            #     因此直接用 lbl_weight == 0 判定 padding，与 transformer.py
            #     内部损失所用的掩码完全一致。
            pad_mask = paddle.reshape(lbl_weight,
                                      [batch_size, trg_len]) < 0.5
            kl = compute_kl_loss(logits1, logits2, pad_mask)

            # (6) 总损失：NLL + alpha * KL
            loss = nll + rdrop_alpha * kl

            # (7) 反向传播 + 梯度累积：每 accum_steps 个 batch 更新一次参数，
            #     等效 batch 大小保持不变。loss 先除以 accum_steps，使累积后的
            #     梯度量级与不累积时一致。
            (loss / accum_steps).backward()
            accum_counter += 1
            if accum_counter >= accum_steps:
                optimizer.step()
                optimizer.clear_grad()
                scheduler.step()
                accum_counter = 0
            # ===== 模型改进核心代码：R-Drop 结束 =====

            if step_idx % args.print_step == 0:
                total_sum_cost = float(sum_cost)
                total_token_num = float(token_num)
                total_avg_cost = total_sum_cost / total_token_num

                if step_idx == 0:
                    logging.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f" %
                        (step_idx, pass_id, batch_id, total_avg_cost,
                         total_avg_cost - loss_normalizer,
                         np.exp([min(total_avg_cost, 100)])))
                    avg_batch_time = time.time()
                else:
                    logging.info(
                        "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                        "normalized loss: %f, ppl: %f, speed: %.2f step/s" %
                        (step_idx, pass_id, batch_id, total_avg_cost,
                         total_avg_cost - loss_normalizer,
                         np.exp([min(total_avg_cost, 100)]),
                         args.print_step / (time.time() - avg_batch_time)))
                    avg_batch_time = time.time()

            if step_idx % args.save_step == 0 and step_idx != 0:
                if args.save_model_path:
                    model_path = os.path.join(args.save_model_path,
                                              "step_" + str(step_idx),
                                              "transformer")
                    paddle.save(model.state_dict(), model_path + ".pdparams")
                    paddle.save(optimizer.state_dict(), model_path + ".pdopt")

            batch_id += 1
            step_idx += 1
            total_batch_num = total_batch_num + 1  # this is for benchmark

        # 每个 epoch 结束时把剩余的累积梯度更新掉
        accum_counter = flush_gradients(accum_counter)

    if args.save_model_path:
        model_path = os.path.join(args.save_model_path, "step_final",
                                  "transformer")
        paddle.save(model.state_dict(), model_path + ".pdparams")
        paddle.save(optimizer.state_dict(), model_path + ".pdopt")


if __name__ == "__main__":
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()
    check_gpu(args.use_cuda)
    check_version()

    do_train(args)

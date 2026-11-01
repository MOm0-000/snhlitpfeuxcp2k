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
"""
R-Drop training entry for the unified Transformer project.

This file reuses the root-level transformer.py, reader.py and transformer.yaml.
The model definition is not duplicated here. R-Drop changes only the training
objective: two stochastic forward passes of the same sample are regularized
with a symmetric KL-divergence term.

When optional model variants such as Scheduled DropHead are enabled,
TransformerModel.update_training_schedules() is called exactly once per
training batch before the R-Drop forward computation.
"""

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
    Read an optional PDConfig value with a fallback.

    PDConfig raises Warning rather than AttributeError for an unknown key, so
    ordinary getattr(args, name, default) cannot be used safely here.
    """
    try:
        value = getattr(args, name)
    except Exception:
        return default
    return default if value is None else value


def compute_kl_loss(logits_p, logits_q, pad_mask):
    """
    Compute R-Drop symmetric KL consistency loss.

    Args:
        logits_p: [B, T, V], logits from the first stochastic branch.
        logits_q: [B, T, V], logits from the second stochastic branch.
        pad_mask: [B, T] bool tensor; True marks padding positions.

    Returns:
        Scalar symmetric KL averaged over valid tokens.
    """
    p_log_softmax = F.log_softmax(logits_p, axis=-1)
    q_log_softmax = F.log_softmax(logits_q, axis=-1)
    p_softmax = F.softmax(logits_p, axis=-1)
    q_softmax = F.softmax(logits_q, axis=-1)

    # Guard against exact zeros in a large-vocabulary softmax.
    p_softmax = paddle.clip(p_softmax, min=1e-12)
    q_softmax = paddle.clip(q_softmax, min=1e-12)

    # F.kl_div(input, label) = label * (log(label) - input).
    kl_qp = F.kl_div(p_log_softmax, q_softmax, reduction="none")
    kl_pq = F.kl_div(q_log_softmax, p_softmax, reduction="none")
    kl_token = 0.5 * paddle.sum(kl_qp + kl_pq, axis=-1)

    # Select valid tokens before reduction. Multiplying NaN padding positions
    # by zero would still leave NaNs in the total loss.
    valid_mask = paddle.logical_not(pad_mask)
    kl_valid = paddle.masked_select(kl_token, valid_mask)
    valid_token_num = paddle.sum(
        paddle.cast(valid_mask, kl_token.dtype))
    valid_token_num.stop_gradient = True

    return paddle.sum(kl_valid) / (valid_token_num + 1e-8)


def compute_nll_loss(logits, lbl_word, lbl_weight, model):
    """
    Reproduce the label-smoothed cross-entropy used by TransformerModel.

    The R-Drop loop needs logits for the KL term, so it performs the model
    forward path explicitly and computes the ordinary NLL here.
    """
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


def forward_logits(model,
                   src_word,
                   src_pos,
                   src_slf_attn_bias,
                   trg_word,
                   trg_pos,
                   trg_slf_attn_bias,
                   trg_src_attn_bias):
    """
    Run the Transformer forward path and return [B, T, V] vocabulary logits.

    encode()/decode()/output_logits() are shared with the root Transformer
    implementation so R-Drop does not maintain a second model definition.
    """
    enc_output = model.encode(src_word, src_pos, src_slf_attn_bias)
    dec_output = model.decode(
        trg_word,
        trg_pos,
        enc_output,
        trg_slf_attn_bias,
        trg_src_attn_bias)

    dec_output = paddle.reshape(
        dec_output, shape=[-1, dec_output.shape[-1]])
    logits = model.output_logits(dec_output)

    return paddle.reshape(
        logits, [trg_word.shape[0], trg_word.shape[1], -1])


def do_train(args):
    if args.use_cuda:
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

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

    model = TransformerModel.from_args(args)
    model.train()

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

    if args.init_from_checkpoint:
        model.set_state_dict(
            paddle.load(
                os.path.join(
                    args.init_from_checkpoint,
                    "transformer.pdparams")))
        optimizer.set_state_dict(
            paddle.load(
                os.path.join(
                    args.init_from_checkpoint,
                    "transformer.pdopt")))
        print(
            "finish initing model from checkpoint from %s" %
            args.init_from_checkpoint)

    if args.init_from_pretrain_model:
        model.set_state_dict(
            paddle.load(
                os.path.join(
                    args.init_from_pretrain_model,
                    "transformer.pdparams")))
        print(
            "finish initing model from pretrained params from %s" %
            args.init_from_pretrain_model)

    loss_normalizer = -(
        (1. - args.label_smooth_eps) *
        np.log(1. - args.label_smooth_eps)
        + args.label_smooth_eps *
        np.log(
            args.label_smooth_eps /
            (args.trg_vocab_size - 1) + 1e-20)
    )

    # R-Drop-specific settings are stored in the unified transformer.yaml.
    rdrop_alpha = float(_get_optional(args, "rdrop_alpha", 1.0))
    accum_steps = max(
        1, int(_get_optional(args, "accum_steps", 1)))

    logging.info(
        "R-Drop enabled: rdrop_alpha = %f, accum_steps = %d",
        rdrop_alpha,
        accum_steps)

    def _rescale_gradients(scale):
        """Multiply all accumulated parameter gradients by ``scale`` in-place."""
        if scale == 1.0:
            return
        for param in model.parameters():
            grad = param.grad
            if grad is not None:
                grad.set_value(grad * scale)

    def flush_gradients(accum_counter):
        """Apply a partial accumulation group with the correct mean scaling."""
        if accum_counter > 0:
            # Each micro-batch backpropagates loss / accum_steps.  A full
            # group therefore already represents the mean gradient.  If an
            # epoch (or max_iter) ends early with only k < accum_steps
            # micro-batches, rescale by accum_steps / k so the residual update
            # is also the mean over the k available micro-batches.
            if accum_counter < accum_steps:
                _rescale_gradients(
                    float(accum_steps) / float(accum_counter))
            optimizer.step()
            optimizer.clear_grad()
            scheduler.step()
        return 0

    step_idx = 0
    total_batch_num = 0
    accum_counter = 0

    for pass_id in range(args.epoch):
        batch_id = 0

        for data_inputs in batch_generator():
            if args.max_iter and total_batch_num == args.max_iter:
                flush_gradients(accum_counter)
                return

            src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos, \
                trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight = [
                    paddle.to_tensor(x) for x in data_inputs
                ]

            batch_size = trg_word.shape[0]
            trg_len = trg_word.shape[1]

            # Scheduled model variants (currently Scheduled DropHead) are
            # advanced once for this original batch. The duplicated 2B R-Drop
            # forward below then uses the same schedule value for both branches.
            model.update_training_schedules()

            # Duplicate the batch once. Dropout/DropHead masks are sampled
            # independently for the first and second halves.
            src_word_2x = paddle.concat(
                [src_word, src_word], axis=0)
            src_pos_2x = paddle.concat(
                [src_pos, src_pos], axis=0)
            src_slf_attn_bias_2x = paddle.concat(
                [src_slf_attn_bias, src_slf_attn_bias], axis=0)

            trg_word_2x = paddle.concat(
                [trg_word, trg_word], axis=0)
            trg_pos_2x = paddle.concat(
                [trg_pos, trg_pos], axis=0)
            trg_slf_attn_bias_2x = paddle.concat(
                [trg_slf_attn_bias, trg_slf_attn_bias], axis=0)
            trg_src_attn_bias_2x = paddle.concat(
                [trg_src_attn_bias, trg_src_attn_bias], axis=0)

            logits_2x = forward_logits(
                model,
                src_word_2x,
                src_pos_2x,
                src_slf_attn_bias_2x,
                trg_word_2x,
                trg_pos_2x,
                trg_slf_attn_bias_2x,
                trg_src_attn_bias_2x)

            logits1, logits2 = paddle.split(
                logits_2x, 2, axis=0)

            del logits_2x
            del src_word_2x, src_pos_2x, src_slf_attn_bias_2x
            del trg_word_2x, trg_pos_2x
            del trg_slf_attn_bias_2x, trg_src_attn_bias_2x

            # Ordinary label-smoothed NLL from the two stochastic branches.
            sum_cost1, token_num = compute_nll_loss(
                logits1, lbl_word, lbl_weight, model)
            sum_cost2, _ = compute_nll_loss(
                logits2, lbl_word, lbl_weight, model)

            sum_cost = 0.5 * (sum_cost1 + sum_cost2)
            nll = sum_cost / token_num

            # lbl_weight=0 identifies padding while preserving the real <e>
            # token, whose label weight is still 1.
            pad_mask = paddle.reshape(
                lbl_weight, [batch_size, trg_len]) < 0.5

            kl = compute_kl_loss(
                logits1, logits2, pad_mask)

            loss = nll + rdrop_alpha * kl

            # Gradient accumulation allows a smaller physical token batch
            # while keeping a larger effective batch.
            (loss / accum_steps).backward()
            accum_counter += 1

            if accum_counter >= accum_steps:
                optimizer.step()
                optimizer.clear_grad()
                scheduler.step()
                accum_counter = 0

            if step_idx % args.print_step == 0:
                total_sum_cost = float(sum_cost)
                total_token_num = float(token_num)
                total_avg_cost = (
                    total_sum_cost / total_token_num)

                if step_idx == 0:
                    logging.info(
                        "step_idx: %d, epoch: %d, batch: %d, "
                        "avg loss: %f, normalized loss: %f, "
                        "ppl: %f, rdrop_kl: %f",
                        step_idx,
                        pass_id,
                        batch_id,
                        total_avg_cost,
                        total_avg_cost - loss_normalizer,
                        np.exp(min(total_avg_cost, 100)),
                        float(kl))
                    avg_batch_time = time.time()
                else:
                    logging.info(
                        "step_idx: %d, epoch: %d, batch: %d, "
                        "avg loss: %f, normalized loss: %f, "
                        "ppl: %f, rdrop_kl: %f, "
                        "speed: %.2f step/s",
                        step_idx,
                        pass_id,
                        batch_id,
                        total_avg_cost,
                        total_avg_cost - loss_normalizer,
                        np.exp(min(total_avg_cost, 100)),
                        float(kl),
                        args.print_step /
                        (time.time() - avg_batch_time))
                    avg_batch_time = time.time()

            if step_idx % args.save_step == 0 and step_idx != 0:
                if args.save_model_path:
                    model_path = os.path.join(
                        args.save_model_path,
                        "step_" + str(step_idx),
                        "transformer")
                    paddle.save(
                        model.state_dict(),
                        model_path + ".pdparams")
                    paddle.save(
                        optimizer.state_dict(),
                        model_path + ".pdopt")

            batch_id += 1
            step_idx += 1
            total_batch_num += 1

        accum_counter = flush_gradients(
            accum_counter)

    if args.save_model_path:
        model_path = os.path.join(
            args.save_model_path,
            "step_final",
            "transformer")
        paddle.save(
            model.state_dict(),
            model_path + ".pdparams")
        paddle.save(
            optimizer.state_dict(),
            model_path + ".pdopt")


if __name__ == "__main__":
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()
    check_gpu(args.use_cuda)
    check_version()
    do_train(args)

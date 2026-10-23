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

import reader
from transformer import TransformerModel
from utils.configure import PDConfig
from utils.check import check_gpu, check_version


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

    step_idx = 0
    total_batch_num = 0  # this is for benchmark
    for pass_id in range(args.epoch):
        batch_id = 0
        for data_inputs in batch_generator():
            if args.max_iter and total_batch_num == args.max_iter:  # this for benchmark
                return

            src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos, \
                trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight = [
                    paddle.to_tensor(x) for x in data_inputs
                ]

            sum_cost, avg_cost, token_num = model(
                src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
                trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight)

            avg_cost.backward()
            optimizer.step()
            optimizer.clear_grad()
            scheduler.step()

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

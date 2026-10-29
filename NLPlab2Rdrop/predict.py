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

import os

import paddle

import reader
from transformer import TransformerModel
from utils.configure import PDConfig
from utils.check import check_gpu, check_version


def post_process_seq(seq, bos_idx, eos_idx, output_bos=False, output_eos=False):
    """
    Post-process the beam-search decoded sequence. Truncate from the first
    <eos> and remove the <bos> and <eos> tokens currently.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def do_predict(args):
    if args.use_cuda:
        paddle.set_device('gpu')
    else:
        paddle.set_device('cpu')
    # define the data generator
    processor = reader.DataProcessor(
        fpattern=args.predict_file,
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        token_delimiter=args.token_delimiter,
        use_token_batch=False,
        batch_size=args.batch_size,
        device_count=1,
        pool_size=args.pool_size,
        sort_type=reader.SortType.NONE,
        shuffle=False,
        shuffle_batch=False,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        max_length=args.max_length,
        n_head=args.n_head)
    batch_generator = processor.data_generator(phase="predict")
    args.src_vocab_size, args.trg_vocab_size, args.bos_idx, args.eos_idx, \
        args.unk_idx = processor.get_vocab_summary()
    trg_idx2word = reader.DataProcessor.load_dict(
        dict_path=args.trg_vocab_fpath, reverse=True)

    # define the model

    model = TransformerModel.from_args(args)

    assert (
        args.init_from_params), "must set init_from_params to load parameters"
    model.set_state_dict(
        paddle.load(
            os.path.join(args.init_from_params, "transformer.pdparams")))
    print("finish initing model from params from %s" % (args.init_from_params))

    # This is used here to set dropout to the test mode.
    model.eval()

    f = open(args.output_file, "wb")
    # start predicting
    with paddle.no_grad():
        for data_inputs in batch_generator():
            src_word, src_pos, src_slf_attn_bias, trg_src_attn_bias = [
                paddle.to_tensor(x) for x in data_inputs
            ]
            batch_size = src_word.shape[0]

            # Run the encoder on the source side, then build the single-step
            # decode cell whose inputs are the selected token ids (the
            # `embedding_fn` of the beam search decoder is None, so the cell
            # performs the word embedding and the position encoding by
            # itself).
            enc_output = model.encode(src_word, src_pos, src_slf_attn_bias)
            cell = model.make_decode_cell(enc_output, trg_src_attn_bias,
                                          args.beam_size)
            decoder = paddle.nn.BeamSearchDecoder(
                cell,
                start_token=args.bos_idx,
                end_token=args.eos_idx,
                beam_size=args.beam_size,
                embedding_fn=None)

            # `seq_ids` is the backtraced (gather_tree) prediction with a
            # fixed shape [batch_size, time_step, beam_size], where the
            # hypotheses are ranked by their scores along the beam axis
            # (i.e. the best one is seq_ids[:, :, 0]).
            seq_ids, _ = paddle.nn.dynamic_decode(
                decoder,
                inits=model.initial_cell_states(batch_size),
                max_step_num=args.max_out_len)
            seq_ids = seq_ids.numpy()

            # How to parse the results is the same as the old
            # implementation: for each source sentence, output the first
            # n_best candidates (one line for each) ordered by their
            # scores.
            for i in range(seq_ids.shape[0]):  # for each source sentence
                for j in range(seq_ids.shape[2]):  # for each candidate
                    f.write(b" ".join([
                        trg_idx2word[idx]
                        for idx in post_process_seq(seq_ids[i, :, j],
                                                    args.bos_idx,
                                                    args.eos_idx)
                    ]) + b"\n")
                    if j + 1 >= args.n_best:
                        break

    f.close()


if __name__ == "__main__":
    args = PDConfig(yaml_file="./transformer.yaml")
    args.build()
    args.Print()
    check_gpu(args.use_cuda)
    check_version()

    do_predict(args)

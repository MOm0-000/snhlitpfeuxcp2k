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
The Transformer network rewritten with the paddle 3.x dygraph API
(paddle.nn.Layer), keeping the numerical behaviour of the original paddle
1.8 static graph implementation: pre-normalized (Pre-LN) sub-layers,
concatenated sin/cos position encodings, three-way weight sharing between
the source embedding, the target embedding and the output projection, and
the label-smoothed cross entropy loss masked by the label weights.
"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def position_encoding_init(n_position, d_pos_vec):
    """
    Generate the initial values for the sinusoid position encoding table.
    """
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (np.log(float(1e4) / float(1)) /
                               (num_timescales - 1))
    inv_timescales = np.exp(np.arange(
        num_timescales)) * -log_timescale_increment
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales,
                                                               0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], 'constant')
    position_enc = signal
    return position_enc.astype("float32")


class MultiHeadAttention(nn.Layer):
    """
    Multi-Head Attention. Note that attn_bias is added to the logit before
    computing softmax activation to mask certain selected positions so that
    they will not considered in attention weights.
    """

    def __init__(self, d_key, d_value, d_model, n_head=1, dropout_rate=0.):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        # The q/k/v/o projections are all bias-free as in the original
        # implementation.
        self.q_proj = nn.Linear(d_model, d_key * n_head, bias_attr=False)
        self.k_proj = nn.Linear(d_model, d_key * n_head, bias_attr=False)
        self.v_proj = nn.Linear(d_model, d_value * n_head, bias_attr=False)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(d_value * n_head, d_model, bias_attr=False)

    def _split_heads(self, x, d_head):
        """
        Reshape input tensors at the last dimension to split multi-heads
        and then transpose. Specifically, transform the input tensor with shape
        [bs, max_sequence_length, n_head * hidden_dim] to the output tensor
        with shape [bs, n_head, max_sequence_length, hidden_dim].
        """
        reshaped = paddle.reshape(
            x=x, shape=[x.shape[0], x.shape[1], self.n_head, d_head])
        return paddle.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def _combine_heads(self, x):
        """
        Transpose and then reshape the last two dimensions of inpunt tensor x
        so that it becomes one dimension, which is reverse to _split_heads.
        """
        trans_x = paddle.transpose(x, perm=[0, 2, 1, 3])
        return paddle.reshape(
            x=trans_x,
            shape=[trans_x.shape[0], trans_x.shape[1],
                   trans_x.shape[2] * trans_x.shape[3]])

    def forward(self, queries, keys, values, attn_bias, cache=None):
        """
        Compute the multi-head attention output. When `cache` is given, a
        tuple of the cached K/V of the decoder self-attention both with shape
        [batch_size, n_head, cached_time, d_key/d_value], the K/V of the
        current step are concatenated along the time axis and the updated
        cache is returned together with the attention output, which is only
        used for faster inference.
        """
        keys = queries if keys is None else keys
        values = keys if values is None else values

        # Add linear projection to queries, keys and values.
        q = self._split_heads(self.q_proj(queries), self.d_key)
        k = self._split_heads(self.k_proj(keys), self.d_key)
        v = self._split_heads(self.v_proj(values), self.d_value)

        if cache is not None:
            # Use cache and concat time steps.
            cache_k, cache_v = cache
            k = paddle.concat([cache_k, k], axis=2)
            v = paddle.concat([cache_v, v], axis=2)

        # Scaled Dot-Product Attention. Note that matmul has no alpha
        # argument in the dygraph API, so the scale has to be multiplied
        # explicitly. The scale uses d_model (not the per-head d_key) to
        # replicate the original static-graph implementation, whose call
        # site passed d_model into scaled_dot_product_attention.
        product = paddle.matmul(x=q, y=k,
                                transpose_y=True) * (self.d_model**-0.5)
        if attn_bias is not None:
            product = product + attn_bias
        weights = F.softmax(product)
        weights = self.attn_dropout(weights)
        ctx_multiheads = paddle.matmul(weights, v)

        # Project back to the model size.
        out = self.out_proj(self._combine_heads(ctx_multiheads))
        if cache is not None:
            return out, (k, v)
        return out


class PositionwiseFeedForward(nn.Layer):
    """
    Position-wise Feed-Forward Networks.
    This module consists of two linear transformations with a ReLU activation
    in between, which is applied to each position separately and identically.
    """

    def __init__(self, d_inner_hid, d_hid, dropout_rate=0.):
        super(PositionwiseFeedForward, self).__init__()
        # Both of the two linear transformations have a bias as in the
        # original implementation.
        self.fc1 = nn.Linear(d_hid, d_inner_hid)
        self.fc2 = nn.Linear(d_inner_hid, d_hid)
        self.relu_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        hidden = self.relu_dropout(hidden)
        return self.fc2(hidden)


class EncoderLayer(nn.Layer):
    """
    The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks. Every sub-layer takes the
    pre-normalized (Pre-LN) form `x + Dropout(Sublayer(LN(x)))`.
    """

    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
                                            attention_dropout)
        self.self_attn_ln = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_inner_hid, d_model, relu_dropout)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(prepostprocess_dropout)

    def forward(self, x, attn_bias):
        attn_output = self.self_attn(self.self_attn_ln(x), None, None,
                                     attn_bias)
        x = x + self.dropout(attn_output)
        ffd_output = self.ffn(self.ffn_ln(x))
        return x + self.dropout(ffd_output)


class Encoder(nn.Layer):
    """
    The encoder is composed of a stack of identical EncoderLayer layers,
    with one more layer normalization applied on the top of the stack.
    """

    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout):
        super(Encoder, self).__init__()
        self.layers = nn.LayerList([
            EncoderLayer(n_head, d_key, d_value, d_model, d_inner_hid,
                         prepostprocess_dropout, attention_dropout,
                         relu_dropout) for _ in range(n_layer)
        ])
        self.top_layer_norm = nn.LayerNorm(d_model)

    def forward(self, enc_input, attn_bias):
        for layer in self.layers:
            enc_input = layer(enc_input, attn_bias)
        return self.top_layer_norm(enc_input)


class DecoderLayer(nn.Layer):
    """
    The layer to be stacked in decoder part.
    The structure of this module is similar to that in the encoder part except
    a multi-head attention is added to implement encoder-decoder attention.
    Every sub-layer takes the pre-normalized (Pre-LN) form
    `x + Dropout(Sublayer(LN(x)))`.

    Besides the whole `forward` used in training, `step` provides one
    incremental decoding step with the cached self-attention K/V.
    """

    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
                                           attention_dropout)
        self.slf_attn_ln = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
                                             attention_dropout)
        self.cross_attn_ln = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_inner_hid, d_model, relu_dropout)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(prepostprocess_dropout)

    def forward(self, x, enc_output, slf_attn_bias, dec_enc_attn_bias):
        slf_attn_output = self.slf_attn(self.slf_attn_ln(x), None, None,
                                        slf_attn_bias)
        x = x + self.dropout(slf_attn_output)
        enc_attn_output = self.cross_attn(self.cross_attn_ln(x), enc_output,
                                          enc_output, dec_enc_attn_bias)
        x = x + self.dropout(enc_attn_output)
        ffd_output = self.ffn(self.ffn_ln(x))
        return x + self.dropout(ffd_output)

    def step(self, x, enc_output, dec_enc_attn_bias, cache):
        """
        One incremental decoding step, where `cache` is the cached
        self-attention K/V tuple of this layer and the updated one is
        returned. No self-attention bias is needed since only the history
        positions are attended. Only the self-attention K/V are cached: the
        old static-graph `fast_decode` additionally computed the
        encoder-decoder attention K/V once (via `static_kv=True` plus
        `wrap_layer_with_block`) and reused that projection across steps,
        whereas here they are re-projected from `enc_output` at every step.
        The two paths are numerically equivalent (identical projection
        weights applied to the same `enc_output`); the difference is only
        that this version recomputes, rather than caches, those K/V per
        step.
        """
        slf_attn_output, cache = self.slf_attn(
            self.slf_attn_ln(x), None, None, None, cache=cache)
        x = x + self.dropout(slf_attn_output)
        enc_attn_output = self.cross_attn(self.cross_attn_ln(x), enc_output,
                                          enc_output, dec_enc_attn_bias)
        x = x + self.dropout(enc_attn_output)
        ffd_output = self.ffn(self.ffn_ln(x))
        return x + self.dropout(ffd_output), cache


class Decoder(nn.Layer):
    """
    The decoder is composed of a stack of identical DecoderLayer layers,
    with one more layer normalization applied on the top of the stack.
    """

    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout):
        super(Decoder, self).__init__()
        self.layers = nn.LayerList([
            DecoderLayer(n_head, d_key, d_value, d_model, d_inner_hid,
                         prepostprocess_dropout, attention_dropout,
                         relu_dropout) for _ in range(n_layer)
        ])
        self.top_layer_norm = nn.LayerNorm(d_model)

    def forward(self, dec_input, enc_output, dec_slf_attn_bias,
                dec_enc_attn_bias):
        for layer in self.layers:
            dec_input = layer(dec_input, enc_output, dec_slf_attn_bias,
                              dec_enc_attn_bias)
        return self.top_layer_norm(dec_input)


class TransformerModel(nn.Layer):
    """
    The Transformer model assembled from the word embeddings, the sinusoid
    position encodings, the encoder and the decoder stacks, and the
    label-smoothed cross entropy loss for training.

    The `forward` method consumes the nine data fields produced by
    `reader.prepare_train_input` one by one and returns the summed cost, the
    averaged cost and the number of valid (non-padded) target tokens.

    For inference, `encode` runs the encoder on the source side, and
    `make_decode_cell` builds the single-step decode cell for
    `paddle.nn.BeamSearchDecoder` (whose `embedding_fn` is None, so the cell
    performs the word embedding and the position encoding by itself), with
    the initial cache states from `initial_cell_states`.

    Note that `max_length` follows the semantics of the configuration (the
    longest sentence kept after filtering), and the position encoding table
    is built with `max_length + 1` rows as the old `create_net` did.
    """

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 max_length,
                 n_layer=6,
                 n_head=8,
                 d_key=64,
                 d_value=64,
                 d_model=512,
                 d_inner_hid=2048,
                 prepostprocess_dropout=0.1,
                 attention_dropout=0.1,
                 relu_dropout=0.1,
                 preprocess_cmd="n",
                 postprocess_cmd="da",
                 weight_sharing=True,
                 label_smooth_eps=0.1,
                 bos_idx=0):
        super(TransformerModel, self).__init__()
        # Only the pre-normalized sub-layer structure of the course
        # configuration is supported by this rewrite.
        assert preprocess_cmd == "n" and postprocess_cmd == "da", (
            "Only the Pre-LN structure (preprocess_cmd='n' and "
            "postprocess_cmd='da') is supported.")
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight "
                "sharing.")

        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.max_length = max_length
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.weight_sharing = weight_sharing
        self.label_smooth_eps = label_smooth_eps
        self.bos_idx = bos_idx

        self.src_word_emb = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=bos_idx,  # set embedding of bos to 0
            weight_attr=nn.initializer.Normal(mean=0., std=d_model**-0.5))
        if weight_sharing:
            # Three-way weight sharing: the source embedding table also
            # serves as the target embedding and the output projection.
            self.trg_word_emb = self.src_word_emb
        else:
            self.trg_word_emb = nn.Embedding(
                trg_vocab_size,
                d_model,
                padding_idx=bos_idx,
                weight_attr=nn.initializer.Normal(
                    mean=0., std=d_model**-0.5))
            self.output_layer = nn.Linear(
                d_model, trg_vocab_size, bias_attr=False)

        self.encoder = Encoder(n_layer, n_head, d_key, d_value, d_model,
                               d_inner_hid, prepostprocess_dropout,
                               attention_dropout, relu_dropout)
        self.decoder = Decoder(n_layer, n_head, d_key, d_value, d_model,
                               d_inner_hid, prepostprocess_dropout,
                               attention_dropout, relu_dropout)
        self.prepostprocess_dropout = nn.Dropout(prepostprocess_dropout)
        # The concatenated sin/cos position encodings are constants rather
        # than the externally initialized "parameter tables" of the old
        # static graph, so a registered buffer is used instead.
        self.register_buffer(
            "pos_enc_table",
            paddle.to_tensor(
                position_encoding_init(max_length + 1, d_model),
                dtype="float32"))

    @classmethod
    def from_args(cls, args):
        """
        Build the model from a configuration object (the `args` produced by
        `utils.configure.PDConfig`), reading the hyper-parameters under the
        same names as the constructor arguments, so that the callers need
        not repeat the argument list.
        """
        return cls(
            src_vocab_size=args.src_vocab_size,
            trg_vocab_size=args.trg_vocab_size,
            max_length=args.max_length,
            n_layer=args.n_layer,
            n_head=args.n_head,
            d_key=args.d_key,
            d_value=args.d_value,
            d_model=args.d_model,
            d_inner_hid=args.d_inner_hid,
            prepostprocess_dropout=args.prepostprocess_dropout,
            attention_dropout=args.attention_dropout,
            relu_dropout=args.relu_dropout,
            preprocess_cmd=args.preprocess_cmd,
            postprocess_cmd=args.postprocess_cmd,
            weight_sharing=args.weight_sharing,
            label_smooth_eps=args.label_smooth_eps,
            bos_idx=args.bos_idx)

    def forward(self, src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
                trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight):
        """
        The training network, whose inputs are the nine data fields produced
        by `reader.prepare_train_input`. Returns the summed cost, the
        averaged cost and the number of valid target tokens.
        """
        enc_output = self.encode(src_word, src_pos, src_slf_attn_bias)
        dec_output = self.decode(trg_word, trg_pos, enc_output,
                                 trg_slf_attn_bias, trg_src_attn_bias)
        # Reshape to 2D tensor to use GEMM instead of BatchedGEMM.
        dec_output = paddle.reshape(
            dec_output, shape=[-1, dec_output.shape[-1]])
        predict = self.output_logits(dec_output)

        # Padding index do not contribute to the total loss. The weights is
        # used to cancel padding index in calculating the loss.
        if self.label_smooth_eps:
            label = F.one_hot(
                paddle.reshape(lbl_word, shape=[-1]),
                num_classes=self.trg_vocab_size).astype("float32")
            label = F.label_smooth(label, epsilon=self.label_smooth_eps)
            cost = F.softmax_with_cross_entropy(
                logits=predict, label=label, soft_label=True)
        else:
            cost = F.softmax_with_cross_entropy(
                logits=predict, label=lbl_word, soft_label=False)
        weighted_cost = paddle.multiply(cost, lbl_weight)
        sum_cost = paddle.sum(weighted_cost)
        token_num = paddle.sum(lbl_weight)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num
        return sum_cost, avg_cost, token_num

    def _embed_and_position(self, word_ids, pos_ids, trg=False):
        """
        Look up the word embedding from the target table when `trg` is True
        and from the source table otherwise, scale it by sqrt(d_model), add
        the position encoding rows selected by `pos_ids` and apply the
        pre/post-process dropout. Shared by `encode`, `decode` and the
        inference decode cell.
        """
        word_emb = self.trg_word_emb if trg else self.src_word_emb
        word_emb = word_emb(word_ids) * (self.d_model**0.5)
        pos_enc = F.embedding(pos_ids, self.pos_enc_table)
        return self.prepostprocess_dropout(word_emb + pos_enc)

    def encode(self, src_word, src_pos, src_slf_attn_bias):
        """
        Add word embeddings (with the source table) and position encodings,
        then pass the result through the encoder stack. This is used at the
        bottom of the encoder for both training and inference.
        """
        enc_input = self._embed_and_position(src_word, src_pos)
        return self.encoder(enc_input, src_slf_attn_bias)

    def decode(self, trg_word, trg_pos, enc_output, trg_slf_attn_bias,
               trg_src_attn_bias):
        """
        Add word embeddings (with the target table) and position encodings,
        then pass the result through the decoder stack.
        """
        dec_input = self._embed_and_position(trg_word, trg_pos, trg=True)
        return self.decoder(dec_input, enc_output, trg_slf_attn_bias,
                            trg_src_attn_bias)

    def output_logits(self, dec_output):
        """
        Project the decoder output to the vocabulary. With weight sharing,
        the transposed source embedding table is used as the output
        projection; otherwise an independent bias-free linear layer.
        """
        if self.weight_sharing:
            return paddle.matmul(
                x=dec_output, y=self.src_word_emb.weight, transpose_y=True)
        return self.output_layer(dec_output)

    def make_decode_cell(self, enc_output, trg_src_attn_bias, beam_size):
        """
        Tile the encoder output and the source attention bias from
        [batch_size, ...] to [batch_size * beam_size, ...] with
        `BeamSearchDecoder.tile_beam_merge_with_batch`, then build the
        single-step decode cell. The tiled tensors are batch-level constants
        captured by the cell and do not participate in the beam reordering.
        """
        enc_output = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            enc_output, beam_size)
        trg_src_attn_bias = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            trg_src_attn_bias, beam_size)
        return TransformerDecodeCell(self, enc_output, trg_src_attn_bias)

    def initial_cell_states(self, batch_size):
        """
        Create the initial (empty) self-attention K/V cache of all the
        decoder layers, shaped [batch_size, n_head, 0, d_key/d_value]. The
        beam search decoder expands them to the beam size itself, and the
        current position of a decoding step is derived from the time
        dimension of the cache.
        """
        dtype = self.pos_enc_table.dtype
        return [(paddle.zeros([batch_size, self.n_head, 0, self.d_key],
                              dtype=dtype),
                 paddle.zeros([batch_size, self.n_head, 0, self.d_value],
                              dtype=dtype)) for _ in range(self.n_layer)]


class TransformerDecodeCell(nn.Layer):
    """
    The single-step transformer decoder cell serving
    `paddle.nn.BeamSearchDecoder`, whose `embedding_fn` is None so that this
    cell performs the word embedding and the position encoding by itself.

    It implements `outputs, next_states = cell(inputs, cell_states)`, where
    `inputs` is a 1-D int64 tensor of the selected token ids with shape
    [batch_size * beam_size] and `cell_states` is a list of the per-layer
    self-attention K/V cache tuples, which is reordered along the parent
    beams by the beam search decoder automatically. The tiled encoder output
    and source attention bias are held by this cell as batch-level constants
    and are not reordered. The returned `outputs` are the logits over the
    vocabulary (the softmax is applied by the beam search decoder itself).
    """

    def __init__(self, model, enc_output, trg_src_attn_bias):
        super(TransformerDecodeCell, self).__init__()
        self.model = model
        self.enc_output = enc_output
        self.trg_src_attn_bias = trg_src_attn_bias

    def forward(self, inputs, cell_states):
        # The position of the current step is derived from the time dimension
        # of the cache, which equals the loop counter of the old fast_decode.
        time_step = cell_states[0][0].shape[2]
        # [batch_size * beam_size] ids -> [batch_size * beam_size, 1, d_model]
        # via the shared embedding helper, with every position id set to the
        # current `time_step` (equivalent to adding `pos_enc_table[time_step]`
        # directly). The ids are reshaped to [N, 1] first so that both
        # operands of the addition are [N, 1, d_model].
        pos_ids = paddle.full(
            shape=[inputs.shape[0], 1], fill_value=time_step, dtype="int64")
        dec_input = self.model._embed_and_position(
            paddle.reshape(inputs, [-1, 1]), pos_ids, trg=True)

        next_states = []
        for i, decoder_layer in enumerate(self.model.decoder.layers):
            dec_input, cache = decoder_layer.step(dec_input, self.enc_output,
                                                  self.trg_src_attn_bias,
                                                  cell_states[i])
            next_states.append(cache)
        dec_output = self.model.decoder.top_layer_norm(dec_input)
        # Reshape to 2D tensor to use GEMM instead of BatchedGEMM.
        dec_output = paddle.reshape(
            dec_output, shape=[-1, dec_output.shape[-1]])
        logits = self.model.output_logits(dec_output)
        return logits, next_states

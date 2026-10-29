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
Paddle 3.x dynamic-graph Transformer used by the course project.

The default configuration remains backward-compatible with the original
repository:

    attention_variant = "normal"
    ffn_variant       = "relu"

With those defaults, the baseline module/parameter names are unchanged, so
old baseline checkpoints can still be loaded.

Section 3.7 experimental variants are opt-in:

1) attention_variant="drophead"
   Scheduled DropHead. Entire attention heads are randomly removed during
   training. A V-shaped schedule is implemented: drop rate starts at p_max,
   decreases linearly to 0 at drophead_warmup_steps, then rises linearly back
   to p_max by drophead_total_steps. In eval()/prediction, DropHead is off.

2) attention_variant="relative_bias"
   Adds a learned per-head relative-position bias to self-attention logits.
   Relative distances are clipped to [-max_relative_position,
   +max_relative_position]. The encoder self-attention and decoder
   self-attention use the bias; encoder-decoder cross-attention stays normal.
   Incremental decoder inference derives the query's absolute position from
   the cached K/V length, so the same learned bias is valid during beam search.

3) ffn_variant="swiglu"
   Replaces the ReLU FFN by a SwiGLU FFN. swiglu_inner_hid controls its hidden
   width.

Important experimental rule:
Do not combine structural variants in the first ablation round. Train one
change at a time against the same tuned control (learning_rate=1.5).
"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def position_encoding_init(n_position, d_pos_vec):
    """Generate the sinusoidal position encoding table."""
    channels = d_pos_vec
    position = np.arange(n_position)
    num_timescales = channels // 2
    log_timescale_increment = (
        np.log(float(1e4) / float(1)) / (num_timescales - 1)
    )
    inv_timescales = np.exp(np.arange(num_timescales) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, np.mod(channels, 2)]], "constant")
    return signal.astype("float32")


class MultiHeadAttention(nn.Layer):
    """
    Multi-head attention with optional Section-3.7 variants.

    normal:
        Exact baseline path: concat heads -> W^O.

    drophead:
        Baseline path plus head-wise structured dropout before concatenation.

    relative_bias:
        Adds a learned per-head relative-position bias to self-attention
        logits before the ordinary attention mask and softmax. Cross-attention
        is deliberately constructed with the normal variant by DecoderLayer.
    """

    VALID_VARIANTS = ("normal", "drophead", "relative_bias")

    def __init__(self,
                 d_key,
                 d_value,
                 d_model,
                 n_head=1,
                 dropout_rate=0.,
                 attention_variant="normal",
                 drophead_rate=0.0,
                 max_relative_position=16):
        super(MultiHeadAttention, self).__init__()
        if attention_variant not in self.VALID_VARIANTS:
            raise ValueError(
                "attention_variant must be one of %s, got %r" %
                (self.VALID_VARIANTS, attention_variant))

        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.attention_variant = attention_variant
        self.drophead_rate = float(drophead_rate)
        self.max_relative_position = int(max_relative_position)

        if (attention_variant == "relative_bias" and
                self.max_relative_position <= 0):
            raise ValueError(
                "max_relative_position must be > 0 for relative_bias")

        # Keep the original baseline parameter names unchanged.
        self.q_proj = nn.Linear(d_model, d_key * n_head, bias_attr=False)
        self.k_proj = nn.Linear(d_model, d_key * n_head, bias_attr=False)
        self.v_proj = nn.Linear(d_model, d_value * n_head, bias_attr=False)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(d_value * n_head, d_model, bias_attr=False)

        # Zero initialization makes a newly constructed relative-bias model
        # begin exactly at the ordinary attention logits before learning.
        if attention_variant == "relative_bias":
            self.relative_position_bias = self.create_parameter(
                shape=[2 * self.max_relative_position + 1, n_head],
                dtype="float32",
                default_initializer=nn.initializer.Constant(0.0))

    def set_drophead_rate(self, rate):
        """Set the current DropHead rate. Called once per training batch."""
        self.drophead_rate = float(max(0.0, min(0.999, rate)))

    def _split_heads(self, x, d_head):
        reshaped = paddle.reshape(
            x=x, shape=[x.shape[0], x.shape[1], self.n_head, d_head])
        return paddle.transpose(x=reshaped, perm=[0, 2, 1, 3])

    def _combine_heads(self, x):
        trans_x = paddle.transpose(x, perm=[0, 2, 1, 3])
        return paddle.reshape(
            x=trans_x,
            shape=[trans_x.shape[0], trans_x.shape[1],
                   trans_x.shape[2] * trans_x.shape[3]])

    def _apply_drophead(self, ctx_multiheads):
        """
        Drop complete heads during training.

        Mask shape is [batch, n_head, 1, 1], therefore a selected head is
        removed for every token/value dimension of that sample.
        """
        if (not self.training or self.attention_variant != "drophead" or
                self.drophead_rate <= 0.0):
            return ctx_multiheads

        keep_prob = 1.0 - self.drophead_rate
        mask_shape = [ctx_multiheads.shape[0], self.n_head, 1, 1]
        mask = (paddle.rand(mask_shape, dtype=ctx_multiheads.dtype) < keep_prob)
        mask = mask.astype(ctx_multiheads.dtype) / keep_prob
        return ctx_multiheads * mask

    def _relative_bias(self, q_len, k_len, cache_is_used):
        """
        Return learned bias with shape [1, n_head, q_len, k_len].

        Full-sequence self-attention:
            query positions = 0 .. q_len-1
            key positions   = 0 .. k_len-1

        Incremental decoder self-attention:
            q_len is normally 1, while k_len includes cached history.
            Therefore the current query starts at position k_len-q_len.
        """
        if self.attention_variant != "relative_bias":
            return None

        query_start = (k_len - q_len) if cache_is_used else 0
        query_pos = paddle.arange(
            query_start, query_start + q_len, dtype="int64").reshape([-1, 1])
        key_pos = paddle.arange(0, k_len, dtype="int64").reshape([1, -1])

        # Positive distance means the key is to the right/future of the query.
        # The decoder causal mask still blocks future positions during training.
        relative = key_pos - query_pos
        m = self.max_relative_position
        relative = paddle.clip(relative, min=-m, max=m)
        bucket = (relative + m).astype("int64")

        # Parameter layout [num_relative_positions, n_head] allows embedding
        # lookup to produce [q_len, k_len, n_head].
        bias = F.embedding(bucket, self.relative_position_bias)
        bias = paddle.transpose(bias, perm=[2, 0, 1])
        return paddle.unsqueeze(bias, axis=0)

    def forward(self,
                queries,
                keys,
                values,
                attn_bias,
                cache=None):
        keys = queries if keys is None else keys
        values = keys if values is None else values

        q = self._split_heads(self.q_proj(queries), self.d_key)
        k = self._split_heads(self.k_proj(keys), self.d_key)
        v = self._split_heads(self.v_proj(values), self.d_value)

        cache_is_used = cache is not None
        if cache_is_used:
            cache_k, cache_v = cache
            k = paddle.concat([cache_k, k], axis=2)
            v = paddle.concat([cache_v, v], axis=2)

        # Preserve the original repository's scaling by d_model^-0.5.
        product = paddle.matmul(
            x=q, y=k, transpose_y=True) * (self.d_model ** -0.5)

        if self.attention_variant == "relative_bias":
            product = product + self._relative_bias(
                q.shape[2], k.shape[2], cache_is_used)

        if attn_bias is not None:
            product = product + attn_bias

        weights = F.softmax(product)
        weights = self.attn_dropout(weights)
        ctx_multiheads = paddle.matmul(weights, v)
        ctx_multiheads = self._apply_drophead(ctx_multiheads)
        out = self.out_proj(self._combine_heads(ctx_multiheads))

        if cache is not None:
            return out, (k, v)
        return out

class PositionwiseFeedForward(nn.Layer):
    """Position-wise FFN supporting the baseline ReLU and SwiGLU."""

    VALID_VARIANTS = ("relu", "swiglu")

    def __init__(self,
                 d_inner_hid,
                 d_hid,
                 dropout_rate=0.,
                 ffn_variant="relu",
                 swiglu_inner_hid=0):
        super(PositionwiseFeedForward, self).__init__()
        if ffn_variant not in self.VALID_VARIANTS:
            raise ValueError(
                "ffn_variant must be one of %s, got %r" %
                (self.VALID_VARIANTS, ffn_variant))
        self.ffn_variant = ffn_variant
        self.relu_dropout = nn.Dropout(dropout_rate)

        if ffn_variant == "relu":
            # Keep original names/shapes/biases for checkpoint compatibility.
            self.fc1 = nn.Linear(d_hid, d_inner_hid)
            self.fc2 = nn.Linear(d_inner_hid, d_hid)
        else:
            # A gated FFN has three matrices. ~2/3 width gives a parameter
            # count close to the baseline two-matrix FFN.
            inner = int(swiglu_inner_hid)
            if inner <= 0:
                inner = int(round(d_inner_hid * 2.0 / 3.0 / 16.0) * 16)
                inner = max(inner, 16)
            self.swiglu_inner_hid = inner
            # Shazeer's GLU-variant comparison uses bias-free projections.
            self.gate_proj = nn.Linear(d_hid, inner, bias_attr=False)
            self.value_proj = nn.Linear(d_hid, inner, bias_attr=False)
            self.out_proj = nn.Linear(inner, d_hid, bias_attr=False)

    def forward(self, x):
        if self.ffn_variant == "relu":
            hidden = F.relu(self.fc1(x))
            hidden = self.relu_dropout(hidden)
            return self.fc2(hidden)

        hidden = F.silu(self.gate_proj(x)) * self.value_proj(x)
        hidden = self.relu_dropout(hidden)
        return self.out_proj(hidden)


class EncoderLayer(nn.Layer):
    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 attention_variant="normal",
                 ffn_variant="relu",
                 drophead_rate=0.0,
                 swiglu_inner_hid=0,
                 max_relative_position=16):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            d_key, d_value, d_model, n_head, attention_dropout,
            attention_variant=attention_variant,
            drophead_rate=drophead_rate,
            max_relative_position=max_relative_position)
        self.self_attn_ln = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(
            d_inner_hid, d_model, relu_dropout,
            ffn_variant=ffn_variant,
            swiglu_inner_hid=swiglu_inner_hid)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(prepostprocess_dropout)

    def forward(self, x, attn_bias):
        attn_output = self.self_attn(
            self.self_attn_ln(x), None, None, attn_bias)
        x = x + self.dropout(attn_output)
        ffd_output = self.ffn(self.ffn_ln(x))
        return x + self.dropout(ffd_output)

class Encoder(nn.Layer):
    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 attention_variant="normal",
                 ffn_variant="relu",
                 drophead_rate=0.0,
                 swiglu_inner_hid=0,
                 max_relative_position=16):
        super(Encoder, self).__init__()
        self.layers = nn.LayerList([
            EncoderLayer(
                n_head, d_key, d_value, d_model, d_inner_hid,
                prepostprocess_dropout, attention_dropout, relu_dropout,
                attention_variant=attention_variant,
                ffn_variant=ffn_variant,
                drophead_rate=drophead_rate,
                swiglu_inner_hid=swiglu_inner_hid,
                max_relative_position=max_relative_position)
            for _ in range(n_layer)
        ])
        self.top_layer_norm = nn.LayerNorm(d_model)

    def forward(self, enc_input, attn_bias):
        for layer in self.layers:
            enc_input = layer(enc_input, attn_bias)
        return self.top_layer_norm(enc_input)

class DecoderLayer(nn.Layer):
    def __init__(self,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 attention_variant="normal",
                 ffn_variant="relu",
                 drophead_rate=0.0,
                 swiglu_inner_hid=0,
                 max_relative_position=16):
        super(DecoderLayer, self).__init__()

        # Relative position is meaningful inside one sequence. Therefore it is
        # used for decoder self-attention only; encoder-decoder cross-attention
        # remains ordinary attention. DropHead keeps its previous behavior and
        # is applied to both self- and cross-attention.
        cross_variant = (
            "normal" if attention_variant == "relative_bias"
            else attention_variant
        )

        self.slf_attn = MultiHeadAttention(
            d_key, d_value, d_model, n_head, attention_dropout,
            attention_variant=attention_variant,
            drophead_rate=drophead_rate,
            max_relative_position=max_relative_position)
        self.slf_attn_ln = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(
            d_key, d_value, d_model, n_head, attention_dropout,
            attention_variant=cross_variant,
            drophead_rate=drophead_rate,
            max_relative_position=max_relative_position)
        self.cross_attn_ln = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(
            d_inner_hid, d_model, relu_dropout,
            ffn_variant=ffn_variant,
            swiglu_inner_hid=swiglu_inner_hid)
        self.ffn_ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(prepostprocess_dropout)

    def forward(self, x, enc_output, slf_attn_bias, dec_enc_attn_bias):
        slf_attn_output = self.slf_attn(
            self.slf_attn_ln(x), None, None, slf_attn_bias)
        x = x + self.dropout(slf_attn_output)

        enc_attn_output = self.cross_attn(
            self.cross_attn_ln(x), enc_output, enc_output, dec_enc_attn_bias)
        x = x + self.dropout(enc_attn_output)

        ffd_output = self.ffn(self.ffn_ln(x))
        return x + self.dropout(ffd_output)

    def step(self, x, enc_output, dec_enc_attn_bias, cache):
        slf_attn_output, cache = self.slf_attn(
            self.slf_attn_ln(x), None, None, None, cache=cache)
        x = x + self.dropout(slf_attn_output)

        enc_attn_output = self.cross_attn(
            self.cross_attn_ln(x), enc_output, enc_output, dec_enc_attn_bias)
        x = x + self.dropout(enc_attn_output)

        ffd_output = self.ffn(self.ffn_ln(x))
        return x + self.dropout(ffd_output), cache

class Decoder(nn.Layer):
    def __init__(self,
                 n_layer,
                 n_head,
                 d_key,
                 d_value,
                 d_model,
                 d_inner_hid,
                 prepostprocess_dropout,
                 attention_dropout,
                 relu_dropout,
                 attention_variant="normal",
                 ffn_variant="relu",
                 drophead_rate=0.0,
                 swiglu_inner_hid=0,
                 max_relative_position=16):
        super(Decoder, self).__init__()
        self.layers = nn.LayerList([
            DecoderLayer(
                n_head, d_key, d_value, d_model, d_inner_hid,
                prepostprocess_dropout, attention_dropout, relu_dropout,
                attention_variant=attention_variant,
                ffn_variant=ffn_variant,
                drophead_rate=drophead_rate,
                swiglu_inner_hid=swiglu_inner_hid,
                max_relative_position=max_relative_position)
            for _ in range(n_layer)
        ])
        self.top_layer_norm = nn.LayerNorm(d_model)

    def forward(self, dec_input, enc_output, dec_slf_attn_bias,
                dec_enc_attn_bias):
        for layer in self.layers:
            dec_input = layer(
                dec_input, enc_output, dec_slf_attn_bias, dec_enc_attn_bias)
        return self.top_layer_norm(dec_input)

class TransformerModel(nn.Layer):
    """Encoder-decoder Transformer with opt-in Section-3.7 variants."""

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
                 bos_idx=0,
                 attention_variant="normal",
                 ffn_variant="relu",
                 drophead_rate=0.1,
                 drophead_schedule="v",
                 drophead_warmup_steps=8000,
                 drophead_total_steps=18400,
                 swiglu_inner_hid=1360,
                 max_relative_position=16):
        super(TransformerModel, self).__init__()
        assert preprocess_cmd == "n" and postprocess_cmd == "da", (
            "Only the Pre-LN structure (preprocess_cmd='n' and "
            "postprocess_cmd='da') is supported.")
        if weight_sharing:
            assert src_vocab_size == trg_vocab_size, (
                "Vocabularies in source and target should be same for weight "
                "sharing.")
        if attention_variant not in MultiHeadAttention.VALID_VARIANTS:
            raise ValueError("invalid attention_variant: %s" % attention_variant)
        if ffn_variant not in PositionwiseFeedForward.VALID_VARIANTS:
            raise ValueError("invalid ffn_variant: %s" % ffn_variant)
        if drophead_schedule not in ("fixed", "v"):
            raise ValueError("drophead_schedule must be 'fixed' or 'v'")

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

        self.attention_variant = attention_variant
        self.ffn_variant = ffn_variant
        self.drophead_rate = float(drophead_rate)
        self.drophead_schedule = drophead_schedule
        self.drophead_warmup_steps = int(drophead_warmup_steps)
        self.drophead_total_steps = int(drophead_total_steps)
        self.swiglu_inner_hid = int(swiglu_inner_hid)
        self.max_relative_position = int(max_relative_position)
        # Python-side batch counter; prediction never uses it because eval()
        # disables DropHead. If resuming a DropHead run, call
        # model.set_drophead_step(resumed_step) after loading the checkpoint.
        self._drophead_step = 0

        self.src_word_emb = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=bos_idx,
            weight_attr=nn.initializer.Normal(mean=0., std=d_model ** -0.5))
        if weight_sharing:
            self.trg_word_emb = self.src_word_emb
        else:
            self.trg_word_emb = nn.Embedding(
                trg_vocab_size,
                d_model,
                padding_idx=bos_idx,
                weight_attr=nn.initializer.Normal(
                    mean=0., std=d_model ** -0.5))
            self.output_layer = nn.Linear(
                d_model, trg_vocab_size, bias_attr=False)

        common = dict(
            attention_variant=attention_variant,
            ffn_variant=ffn_variant,
            drophead_rate=drophead_rate,
            swiglu_inner_hid=swiglu_inner_hid,
            max_relative_position=max_relative_position)
        self.encoder = Encoder(
            n_layer, n_head, d_key, d_value, d_model, d_inner_hid,
            prepostprocess_dropout, attention_dropout, relu_dropout, **common)
        self.decoder = Decoder(
            n_layer, n_head, d_key, d_value, d_model, d_inner_hid,
            prepostprocess_dropout, attention_dropout, relu_dropout, **common)

        self.prepostprocess_dropout = nn.Dropout(prepostprocess_dropout)
        self.register_buffer(
            "pos_enc_table",
            paddle.to_tensor(
                position_encoding_init(max_length + 1, d_model),
                dtype="float32"))

    @staticmethod
    def _arg(args, name, default):
        return getattr(args, name, default)

    @classmethod
    def from_args(cls, args):
        """
        Build from PDConfig args. getattr defaults keep old YAML files usable;
        add the new scalar keys to transformer.yaml to expose CLI switches.
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
            bos_idx=args.bos_idx,
            attention_variant=cls._arg(args, "attention_variant", "normal"),
            ffn_variant=cls._arg(args, "ffn_variant", "relu"),
            drophead_rate=cls._arg(args, "drophead_rate", 0.1),
            drophead_schedule=cls._arg(args, "drophead_schedule", "v"),
            drophead_warmup_steps=cls._arg(
                args, "drophead_warmup_steps", 8000),
            drophead_total_steps=cls._arg(
                args, "drophead_total_steps", 18400),
            swiglu_inner_hid=cls._arg(args, "swiglu_inner_hid", 1360),
            max_relative_position=cls._arg(
                args, "max_relative_position", 16))

    def set_drophead_step(self, step):
        """Useful only when resuming a Scheduled-DropHead training run."""
        self._drophead_step = max(0, int(step))

    def _scheduled_drophead_rate(self):
        p_max = max(0.0, min(0.999, self.drophead_rate))
        if self.drophead_schedule == "fixed" or self.drophead_total_steps <= 0:
            return p_max

        step = min(self._drophead_step, self.drophead_total_steps)
        warmup = max(0, min(self.drophead_warmup_steps,
                            self.drophead_total_steps))

        if warmup > 0 and step <= warmup:
            return p_max * (1.0 - float(step) / float(warmup))

        tail = self.drophead_total_steps - warmup
        if tail <= 0:
            return 0.0
        return p_max * float(step - warmup) / float(tail)

    def _set_all_drophead_rates(self, rate):
        for layer in self.encoder.layers:
            layer.self_attn.set_drophead_rate(rate)
        for layer in self.decoder.layers:
            layer.slf_attn.set_drophead_rate(rate)
            layer.cross_attn.set_drophead_rate(rate)

    def current_drophead_rate(self):
        """Expose the scheduled value for logging in experiments."""
        return self._scheduled_drophead_rate()

    def forward(self, src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
                trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight):
        # One global schedule value is shared by all heads/layers in a batch.
        if self.attention_variant == "drophead" and self.training:
            rate = self._scheduled_drophead_rate()
            self._set_all_drophead_rates(rate)
            self._drophead_step += 1

        enc_output = self.encode(src_word, src_pos, src_slf_attn_bias)
        dec_output = self.decode(trg_word, trg_pos, enc_output,
                                 trg_slf_attn_bias, trg_src_attn_bias)
        dec_output = paddle.reshape(
            dec_output, shape=[-1, dec_output.shape[-1]])
        predict = self.output_logits(dec_output)

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
        masked_cost = paddle.multiply(cost, lbl_weight)
        sum_cost = paddle.sum(masked_cost)
        token_num = paddle.sum(lbl_weight)
        token_num.stop_gradient = True
        avg_cost = sum_cost / token_num
        return sum_cost, avg_cost, token_num

    def _embed_and_position(self, word_ids, pos_ids, trg=False):
        word_emb_layer = self.trg_word_emb if trg else self.src_word_emb
        word_emb = word_emb_layer(word_ids) * (self.d_model ** 0.5)
        pos_enc = F.embedding(pos_ids, self.pos_enc_table)
        return self.prepostprocess_dropout(word_emb + pos_enc)

    def encode(self, src_word, src_pos, src_slf_attn_bias):
        enc_input = self._embed_and_position(src_word, src_pos)
        return self.encoder(enc_input, src_slf_attn_bias)

    def decode(self, trg_word, trg_pos, enc_output, trg_slf_attn_bias,
               trg_src_attn_bias):
        dec_input = self._embed_and_position(trg_word, trg_pos, trg=True)
        return self.decoder(dec_input, enc_output, trg_slf_attn_bias,
                            trg_src_attn_bias)

    def output_logits(self, dec_output):
        if self.weight_sharing:
            return paddle.matmul(
                x=dec_output, y=self.src_word_emb.weight, transpose_y=True)
        return self.output_layer(dec_output)

    def make_decode_cell(self, enc_output, trg_src_attn_bias, beam_size):
        enc_output = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            enc_output, beam_size)
        trg_src_attn_bias = nn.BeamSearchDecoder.tile_beam_merge_with_batch(
            trg_src_attn_bias, beam_size)
        return TransformerDecodeCell(self, enc_output, trg_src_attn_bias)

    def initial_cell_states(self, batch_size):
        dtype = self.pos_enc_table.dtype
        return [(paddle.zeros([batch_size, self.n_head, 0, self.d_key],
                              dtype=dtype),
                 paddle.zeros([batch_size, self.n_head, 0, self.d_value],
                              dtype=dtype)) for _ in range(self.n_layer)]

    def model_variant_summary(self):
        """Small serializable dict convenient for experiment metadata."""
        return {
            "attention_variant": self.attention_variant,
            "ffn_variant": self.ffn_variant,
            "drophead_rate": self.drophead_rate,
            "drophead_schedule": self.drophead_schedule,
            "drophead_warmup_steps": self.drophead_warmup_steps,
            "drophead_total_steps": self.drophead_total_steps,
            "swiglu_inner_hid": self.swiglu_inner_hid,
            "max_relative_position": self.max_relative_position,
        }


class TransformerDecodeCell(nn.Layer):
    """Single-step decoder cell used by paddle.nn.BeamSearchDecoder."""

    def __init__(self, model, enc_output, trg_src_attn_bias):
        super(TransformerDecodeCell, self).__init__()
        self.model = model
        self.enc_output = enc_output
        self.trg_src_attn_bias = trg_src_attn_bias

    def forward(self, inputs, cell_states):
        time_step = cell_states[0][0].shape[2]
        pos_ids = paddle.full(
            shape=[inputs.shape[0], 1], fill_value=time_step, dtype="int64")
        dec_input = self.model._embed_and_position(
            paddle.reshape(inputs, [-1, 1]), pos_ids, trg=True)

        next_states = []
        for i, decoder_layer in enumerate(self.model.decoder.layers):
            dec_input, cache = decoder_layer.step(
                dec_input, self.enc_output, self.trg_src_attn_bias,
                cell_states[i])
            next_states.append(cache)
        dec_output = self.model.decoder.top_layer_norm(dec_input)
        dec_output = paddle.reshape(
            dec_output, shape=[-1, dec_output.shape[-1]])
        logits = self.model.output_logits(dec_output)
        return logits, next_states

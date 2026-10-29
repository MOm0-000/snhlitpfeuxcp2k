"""Smoke tests for Section-3.7 Transformer variants.

Run in the nlpexp2 environment from the project root after replacing
transformer.py:

    python test_transformer_37.py

The test covers forward/backward and two incremental decoder steps so the
relative-position bias cache indexing is exercised as well.
"""
import paddle
from transformer import TransformerModel


def make_inputs(batch=2, src_len=5, trg_len=4, n_head=2, vocab=64):
    src_word = paddle.randint(2, vocab, [batch, src_len], dtype="int64")
    src_pos = paddle.arange(src_len, dtype="int64").reshape([1, -1]).tile([batch, 1])
    src_bias = paddle.zeros([batch, n_head, src_len, src_len], dtype="float32")

    trg_word = paddle.randint(2, vocab, [batch, trg_len], dtype="int64")
    trg_pos = paddle.arange(trg_len, dtype="int64").reshape([1, -1]).tile([batch, 1])
    causal = paddle.triu(paddle.ones([trg_len, trg_len]), diagonal=1) * -1e9
    trg_bias = causal.reshape([1, 1, trg_len, trg_len]).tile([batch, n_head, 1, 1])
    cross_bias = paddle.zeros([batch, n_head, trg_len, src_len], dtype="float32")

    lbl_word = paddle.randint(2, vocab, [batch * trg_len, 1], dtype="int64")
    lbl_weight = paddle.ones([batch * trg_len, 1], dtype="float32")
    return (src_word, src_pos, src_bias, trg_word, trg_pos,
            trg_bias, cross_bias, lbl_word, lbl_weight)


def build(attention_variant="normal", ffn_variant="relu"):
    return TransformerModel(
        src_vocab_size=64,
        trg_vocab_size=64,
        max_length=16,
        n_layer=2,
        n_head=2,
        d_key=8,
        d_value=8,
        d_model=16,
        d_inner_hid=32,
        prepostprocess_dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        weight_sharing=True,
        label_smooth_eps=0.1,
        bos_idx=0,
        attention_variant=attention_variant,
        ffn_variant=ffn_variant,
        drophead_rate=0.2,
        drophead_schedule="v",
        drophead_warmup_steps=2,
        drophead_total_steps=6,
        swiglu_inner_hid=24,
        max_relative_position=4)


def test_variant(name, attention_variant="normal", ffn_variant="relu"):
    paddle.seed(1234)
    model = build(attention_variant, ffn_variant)
    model.train()
    inputs = make_inputs()
    _, avg_cost, _ = model(*inputs)
    avg_cost.backward()

    keys = list(model.state_dict().keys())
    relative_keys = [k for k in keys if "relative_position_bias" in k]
    if attention_variant == "relative_bias":
        # 2 encoder self-attn + 2 decoder self-attn. Cross-attn stays normal.
        assert len(relative_keys) == 4, relative_keys
        for layer in model.decoder.layers:
            assert layer.cross_attn.attention_variant == "normal"
    else:
        assert not relative_keys, relative_keys

    # Exercise two incremental decoder steps. Step 2 is important because it
    # tests q-position recovery from non-empty K/V cache.
    model.eval()
    src_word, src_pos, src_bias = inputs[:3]
    enc = model.encode(src_word, src_pos, src_bias)
    states = model.initial_cell_states(src_word.shape[0])
    cross = paddle.zeros(
        [src_word.shape[0], model.n_head, 1, src_word.shape[1]],
        dtype="float32")

    for time_step in range(2):
        ids = paddle.full([src_word.shape[0], 1], 2 + time_step, dtype="int64")
        pos = paddle.full(
            [src_word.shape[0], 1], time_step, dtype="int64")
        dec = model._embed_and_position(ids, pos, trg=True)
        for i, layer in enumerate(model.decoder.layers):
            dec, states[i] = layer.step(dec, enc, cross, states[i])
        assert states[0][0].shape[2] == time_step + 1

    print("%-18s OK  avg_loss=%.6f  variant=%s/%s" % (
        name, float(avg_cost), attention_variant, ffn_variant))


if __name__ == "__main__":
    test_variant("baseline", "normal", "relu")
    test_variant("drophead", "drophead", "relu")
    test_variant("relative_bias", "relative_bias", "relu")
    test_variant("swiglu", "normal", "swiglu")
    print("all Section-3.7 smoke tests passed")

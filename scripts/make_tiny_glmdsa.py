#!/usr/bin/env python3
"""Generate a tiny random-weight GGUF with architecture "glm-dsa" for fast
plumbing tests (graph building, TP/KV-shard, sparse-attention paths) of
llama.cpp without loading the real 100+ GB GLM-5.2 model.

Layer layout mirrors the real model's role pattern, scaled down:
  real:  79 blocks = 3 dense FFN + 75 MoE + 1 nextn(MTP, MoE FFN)
  tiny:   5 blocks = 1 dense FFN +  3 MoE + 1 nextn(MTP, MoE FFN)

The tokenizer (tokens / token_type / merges / special ids / chat template) is
copied verbatim from the real GGUF so vocab behavior is identical; therefore
n_vocab stays 154880 and token_embd dominates the file size (Q8_0 to keep the
file ~45 MB). All other tensors are random normal * 0.02 in F16 (norm weights
are ones, norm biases zeros, F32 - mirroring the real model's per-tensor
dtypes: 1D norms/biases F32, ffn_gate_inp F32, exp_probs_b F32,
indexer.proj F32).

Usage:
  /home/frozenlock/sglang-env/bin/python scripts/make_tiny_glmdsa.py
"""

import sys

sys.path.insert(0, "/home/frozenlock/llama.cpp-prod/gguf-py")

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, quants

REAL_GGUF = "/home/frozenlock/models/GLM-5.2-GGUF/UD-IQ3_XXS/GLM-5.2-UD-IQ3_XXS-00001-of-00007.gguf"
OUT_PATH = "/home/frozenlock/models/tiny-glmdsa/tiny-glmdsa-f16.gguf"

ARCH = "glm-dsa"

# ---- tiny hparams (ratios mirror the real model) ---------------------------
# real: n_embd 6144, n_head 64, q_lora 2048, kv_lora 512, rope 64,
#       k_mla 256, v_mla 256, key_length 576 (=kv_lora+rope), value_length 512
#       (=kv_lora), n_ff 12288, n_ff_exp 2048, 256 experts / 8 used / 1 shared,
#       dense lead 3, blocks 79, nextn 1,
#       indexer: 32 heads x 128 head_size, top_k 2048
N_BLOCKS = 5           # total blocks incl. nextn (glm-dsa.block_count)
N_NEXTN = 1            # last layer is the MTP/nextn layer
N_DENSE_LEAD = 1       # layer 0 dense FFN, layers 1..4 MoE
N_EMBD = 256
N_HEAD = 8
N_HEAD_KV = 1
Q_LORA = 64
# NOTE: the MLA attention dims are kept at the REAL model's values. The MLA
# flash-attn kernels only support DKQ=576 / DV=512 (= kv_lora + rope /
# kv_lora); with scaled-down dims FA gets rejected and the graph falls into
# the non-FA MLA path, which the TP meta backend aborts on
# (TPHYBRID mul_mat unhandled, ggml-backend-meta.cpp:694).
KV_LORA = 512
N_ROT = 64             # rope.dimension_count
K_MLA = 256            # key_length_mla   -> qk_nope = K_MLA - N_ROT = 192
V_MLA = 256            # value_length_mla
KEY_LEN = KV_LORA + N_ROT     # 576
VAL_LEN = KV_LORA             # 512
N_FF = 512             # dense-layer FFN
N_FF_EXP = 128         # per-expert FFN
N_EXPERT = 8
N_EXPERT_USED = 2
N_EXPERT_SHARED = 1
IDX_N_HEAD = 4
IDX_HEAD_SIZE = 128    # kept at the real value (indexer kernel block size)
# top_k must stay a multiple of FATTN_KQ_STRIDE (256): the sparse decode path
# runs FA on the gathered top-k rows and the DKQ=576 kernel requires
# K->ne[1] % 256 == 0 (gqa_opt_applies), else FA is rejected and the non-FA
# path aborts (real model uses 2048)
IDX_TOP_K = 256
CTX_LEN = 131072
ROPE_FREQ_BASE = 8000000.0
RMS_EPS = 1e-5

rng = np.random.default_rng(1234)


def rand(shape_ne):
    """shape_ne is in GGUF ne order (like create_tensor {a,b,c}); numpy shape
    is the reverse."""
    np_shape = tuple(reversed(shape_ne))
    return (rng.standard_normal(np_shape) * 0.02).astype(np.float16)


def main():
    print(f"reading tokenizer from {REAL_GGUF} ...")
    r = GGUFReader(REAL_GGUF)

    def field(name):
        f = r.get_field(name)
        assert f is not None, f"missing field {name} in real GGUF"
        return f.contents()

    tokens = field("tokenizer.ggml.tokens")
    token_type = field("tokenizer.ggml.token_type")
    merges = field("tokenizer.ggml.merges")
    chat_template = field("tokenizer.chat_template")
    special = {
        k: field(f"tokenizer.ggml.{k}")
        for k in ("eos_token_id", "padding_token_id", "bos_token_id",
                  "eot_token_id", "unknown_token_id", "eom_token_id")
    }
    n_vocab = len(tokens)
    print(f"  vocab {n_vocab}, merges {len(merges)}")

    import os
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    w = GGUFWriter(OUT_PATH, ARCH)

    # ---- metadata (same keys as the real model, scaled values) ----
    w.add_string("general.name", "tiny-glmdsa")
    w.add_string("general.type", "model")
    w.add_uint32("general.quantization_version", 2)
    w.add_uint32("general.file_type", 1)  # mostly F16

    w.add_uint32(f"{ARCH}.block_count", N_BLOCKS)
    w.add_uint32(f"{ARCH}.context_length", CTX_LEN)
    w.add_uint32(f"{ARCH}.embedding_length", N_EMBD)
    w.add_uint32(f"{ARCH}.feed_forward_length", N_FF)
    w.add_uint32(f"{ARCH}.attention.head_count", N_HEAD)
    w.add_uint32(f"{ARCH}.attention.head_count_kv", N_HEAD_KV)
    w.add_float32(f"{ARCH}.rope.freq_base", ROPE_FREQ_BASE)
    w.add_float32(f"{ARCH}.attention.layer_norm_rms_epsilon", RMS_EPS)
    w.add_uint32(f"{ARCH}.expert_count", N_EXPERT)
    w.add_uint32(f"{ARCH}.expert_used_count", N_EXPERT_USED)
    w.add_uint32(f"{ARCH}.expert_group_count", 1)
    w.add_uint32(f"{ARCH}.expert_group_used_count", 1)
    w.add_uint32(f"{ARCH}.expert_gating_func", 2)  # sigmoid
    w.add_uint32(f"{ARCH}.attention.key_length", KEY_LEN)
    w.add_uint32(f"{ARCH}.attention.value_length", VAL_LEN)
    w.add_uint32(f"{ARCH}.leading_dense_block_count", N_DENSE_LEAD)
    w.add_uint32(f"{ARCH}.vocab_size", n_vocab)
    w.add_uint32(f"{ARCH}.attention.q_lora_rank", Q_LORA)
    w.add_uint32(f"{ARCH}.attention.kv_lora_rank", KV_LORA)
    w.add_uint32(f"{ARCH}.attention.key_length_mla", K_MLA)
    w.add_uint32(f"{ARCH}.attention.value_length_mla", V_MLA)
    w.add_uint32(f"{ARCH}.expert_feed_forward_length", N_FF_EXP)
    w.add_uint32(f"{ARCH}.expert_shared_count", N_EXPERT_SHARED)
    w.add_float32(f"{ARCH}.expert_weights_scale", 2.5)
    w.add_bool(f"{ARCH}.expert_weights_norm", True)
    w.add_uint32(f"{ARCH}.rope.dimension_count", N_ROT)
    w.add_uint32(f"{ARCH}.nextn_predict_layers", N_NEXTN)
    w.add_uint32(f"{ARCH}.attention.indexer.head_count", IDX_N_HEAD)
    w.add_uint32(f"{ARCH}.attention.indexer.key_length", IDX_HEAD_SIZE)
    w.add_uint32(f"{ARCH}.attention.indexer.top_k", IDX_TOP_K)

    # ---- tokenizer: verbatim copy ----
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "glm4")
    w.add_token_list(tokens)
    w.add_token_types(token_type)
    w.add_token_merges(merges)
    for k, v in special.items():
        w.add_uint32(f"tokenizer.ggml.{k}", int(v))
    w.add_string("tokenizer.chat_template", chat_template)

    # ---- tensors ----
    def add_f16(name, ne):
        w.add_tensor(name, rand(ne))

    def add_f32_ones(name, n):
        w.add_tensor(name, np.ones(n, dtype=np.float32))

    def add_f32_zeros(name, n):
        w.add_tensor(name, np.zeros(n, dtype=np.float32))

    def add_f32_rand(name, ne):
        w.add_tensor(name, rand(ne).astype(np.float32))

    # token_embd: Q8_0 to keep file size down (n_vocab is the real 154880);
    # output.weight omitted -> loader falls back to tied embeddings
    print("quantizing token_embd to Q8_0 ...")
    embd = (rng.standard_normal((n_vocab, N_EMBD)) * 0.02).astype(np.float32)
    q = quants.quantize(embd, GGMLQuantizationType.Q8_0)
    w.add_tensor("token_embd.weight", q, raw_dtype=GGMLQuantizationType.Q8_0)
    add_f32_ones("output_norm.weight", N_EMBD)

    qk_nope = K_MLA - N_ROT

    for i in range(N_BLOCKS):
        p = f"blk.{i}."
        add_f32_ones(p + "attn_norm.weight", N_EMBD)
        add_f32_ones(p + "attn_q_a_norm.weight", Q_LORA)
        add_f32_ones(p + "attn_kv_a_norm.weight", KV_LORA)

        add_f16(p + "attn_q_a.weight", (N_EMBD, Q_LORA))
        add_f16(p + "attn_q_b.weight", (Q_LORA, N_HEAD * K_MLA))
        add_f16(p + "attn_kv_a_mqa.weight", (N_EMBD, KV_LORA + N_ROT))
        add_f16(p + "attn_k_b.weight", (qk_nope, KV_LORA, N_HEAD))
        add_f16(p + "attn_v_b.weight", (KV_LORA, V_MLA, N_HEAD))
        add_f16(p + "attn_output.weight", (N_HEAD * V_MLA, N_EMBD))

        add_f32_ones(p + "ffn_norm.weight", N_EMBD)

        # DSA indexer (dtypes mirror the real model)
        add_f32_ones(p + "indexer.k_norm.weight", IDX_HEAD_SIZE)
        add_f32_zeros(p + "indexer.k_norm.bias", IDX_HEAD_SIZE)
        add_f32_rand(p + "indexer.proj.weight", (N_EMBD, IDX_N_HEAD))
        add_f16(p + "indexer.attn_k.weight", (N_EMBD, IDX_HEAD_SIZE))
        add_f16(p + "indexer.attn_q_b.weight", (Q_LORA, IDX_N_HEAD * IDX_HEAD_SIZE))

        if i < N_DENSE_LEAD:
            add_f16(p + "ffn_gate.weight", (N_EMBD, N_FF))
            add_f16(p + "ffn_down.weight", (N_FF, N_EMBD))
            add_f16(p + "ffn_up.weight", (N_EMBD, N_FF))
        else:
            add_f32_rand(p + "ffn_gate_inp.weight", (N_EMBD, N_EXPERT))
            add_f32_zeros(p + "exp_probs_b.bias", N_EXPERT)
            add_f16(p + "ffn_gate_exps.weight", (N_EMBD, N_FF_EXP, N_EXPERT))
            add_f16(p + "ffn_down_exps.weight", (N_FF_EXP, N_EMBD, N_EXPERT))
            add_f16(p + "ffn_up_exps.weight", (N_EMBD, N_FF_EXP, N_EXPERT))
            add_f16(p + "ffn_gate_shexp.weight", (N_EMBD, N_FF_EXP * N_EXPERT_SHARED))
            add_f16(p + "ffn_down_shexp.weight", (N_FF_EXP * N_EXPERT_SHARED, N_EMBD))
            add_f16(p + "ffn_up_shexp.weight", (N_EMBD, N_FF_EXP * N_EXPERT_SHARED))

        if i >= N_BLOCKS - N_NEXTN:
            # same tensor set as the real model's layer 78 (embed_tokens and
            # shared_head_head absent there too -> tied fallbacks)
            add_f16(p + "nextn.eh_proj.weight", (2 * N_EMBD, N_EMBD))
            add_f32_ones(p + "nextn.enorm.weight", N_EMBD)
            add_f32_ones(p + "nextn.hnorm.weight", N_EMBD)
            add_f32_ones(p + "nextn.shared_head_norm.weight", N_EMBD)

    print(f"writing {OUT_PATH} ...")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    import pathlib
    sz = pathlib.Path(OUT_PATH).stat().st_size
    print(f"done: {sz / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

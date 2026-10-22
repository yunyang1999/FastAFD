# DeepSeek-V4 (Flash/Pro) and Kimi-K3 in FastAFD — design note

Scope: colocated (`minisgl` engine) and AFD (attention workers + MegaMoE expert
workers) serving of `DeepseekV4ForCausalLM` and `KimiK3ForConditionalGeneration`.
References: official DeepSeek-V4 `inference/model.py` + `kernel.py` (HF repo),
Kimi-K3 `modeling_kimi_linear.py` (HF repo), sglang 0.5.18 `models/deepseek_v4.py`,
`models/kimi_k3.py`, `layers/attention/linear/kda_backend.py`.

## 1. What each model needs

| | DeepSeek-V4-Flash | DeepSeek-V4-Pro | Kimi-K3 |
|---|---|---|---|
| layers / hidden / vocab | 43 / 4096 / 129280 | 61 / 7168 / 129280 | 93 / 7168 / 163840 |
| attention | MQA, 64 heads x 512 (k = v = one 512-d latent, RoPE on the last 64 dims, interleaved) | 128 heads | 69 KDA + 24 MLA (NoPE, output gate), 96 heads |
| per-layer attention kind | 2 SWA-only, 21 C4 (compressed 1/4 + lightning indexer top-512), 20 C128 (compressed 1/128, dense over compressed) | 30 C4 (top-1024) + 31 C128 | KDA: conv(4)+gated delta rule, state 96x128x128; MLA: q_lora 1536, kv_lora 512 + 64 |
| residual | mHC: 4 residual copies, Sinkhorn-mixed (20 iters) pre/post per sub-block, `hc_head` at the end | same | attention residuals: block summaries every 12 layers, softmax-mixed before attention/MLP |
| MoE | 256 routed top-6 + 1 shared, I=2048, sqrt-softplus + bias (noaux), first 3 layers hash-routed (`tid2eid`), rsf 1.5, SwiGLU clamp 10 | 384 routed, I=3072, rsf 2.5 | layer 0 dense (I=33792, SiTU); 92 LatentMoE: router in 7168, latent 3584 (down-proj, experts, RMSNorm, up-proj), 896 routed top-16 I=3072, sigmoid + bias, renorm, rsf 1; 2 shared experts (7168, SiTU) |
| checkpoint precision | non-expert FP8 e4m3 (128x128, UE8M0 scales); experts FP4 e2m1 (UE8M0 per-32) | same | non-expert BF16; experts MXFP4 (compressed-tensors `weight_packed`/`weight_scale`) |
| MTP | 1 nextn layer (later) | 1 (later) | none (DSPARK draft optional, later) |

Sizing (computed from the HF configs; dummy and real weights use the same dtypes):

| | attention-side params | routed experts (MXFP4) | KV / state |
|---|---|---|---|
| DSV4-Flash | 6.2 B (FP8 ~5.8 GiB) + shared 1.1 B | 137 GiB | 7.6 KB/token (bf16 cache), 6 MiB SWA + 12 MiB compressor state per request |
| DSV4-Pro | 21.6 B (FP8 ~20 GiB) + shared 4.0 B | 766 GiB | 10.8 KB/token, 9 MiB + 18 MiB per request |
| Kimi-K3 | 56.7 B (BF16 ~106 GiB at attention TP1) | 1347 GiB | MLA 27 KB/token; KDA 414 MiB (fp32) + 15 MiB conv per request |

## 2. Kernels (all usable from the `minisgl-cuda130` env; no sglang import)

Decode on the attention side runs the kernels sglang 0.5.18 selects on GB200
wherever they exist outside compiled sglang binaries (details and timings in §9):

* Vendored sglang JIT CUDA kernels (`kernel/sgl_jit`, compiled with nvcc through
  tvm-ffi on first use, cached under `$MINISGL_JIT_CACHE_DIR`): KDA packed decode
  (B >= 8), radix router (K3), attention-residual TMA kernel (K3, SM100), DSV4
  main-q norm+RoPE, indexer-q RoPE+Hadamard+FP8 quant, compressed indexer-key
  norm+RoPE+Hadamard+FP8 store (fused_norm_rope_v2), (inverse) RoPE, top-k v2
  (fused top-512 + page-table transform) and the wo_a FP8 group quant.
* Vendored sglang Triton kernels (`kernel/sgl_triton`): `causal_conv1d_update`, the
  KDA packed-decode fallback (B < 8), the sigmoid-gated RMSNorm and the fused MoE
  router (sqrt-softplus for DSV4).
* DeepGEMM (minisgl's vendored build): FP8 GEMMs (`fp8_gemm_nt`, wq_a|wkv merged),
  the FP8 grouped einsum for DSV4 `wo_a`, the TF32 split-K mHC prenorm GEMM and the
  paged FP8 MQA logits of the DSV4 indexer (keys FP8 + FP32 scale in 64-entry pages).
* flashinfer 0.6.6: trtllm-gen MLA decode for K3 (paged BF16 latent, page 64; heads
  zero-padded to a power of two) and trtllm-gen sparse MLA for DSV4 (FP8 e4m3 query
  and KV rows with per-tensor scale 1 by default, BF16 with
  `MINISGL_DSV4_KV_DTYPE=bf16`; per-head sinks; heads padded to 128 at large
  batch x top-k for the 2-CTA kernel).
* Own Triton where sglang's kernel is not usable here: the fused mHC pre (split-K
  reduction + Sinkhorn + weighted sum + RMSNorm, the shape of sglang's TileLang
  `mhc_pre_big_fuse_with_norm`), mHC post, the DSV4 compressors and the sparse-index
  assembly. Prefill keeps the reference-style kernels (KDA recurrence, FA2 MLA,
  bf16 indexer scores over dequantised keys).
* Not available in this environment (compiled-only or a newer stack): FlashMLA's
  SM100 FP8 sparse decode (sgl_kernel `flashmla_ops`), TileLang mHC kernels
  (tilelang needs tvm-ffi >= 0.1.10), flashinfer >= 0.6.17 kernels (CuTe-DSL MLA,
  `trtllm_batch_decode_sparse_mla_dsv4`, mhc) and the CuTe-DSL TGV GEMM.
* MoE: split MegaMoE M2N (`moe/megamoe_m2n_afd.py`) for AFD; colocated MegaMoE
  (`kernel/megamoe_mega.py`) for the colocated server; a torch reference MoE for
  numerics.

## 3. AFD split (must equal the measured MegaMoE stage)

The F side runs exactly the `megamoe_model_profiles` stage; everything else is on
the attention side. The AFD MoE contract (hidden, intermediate, experts, top-k,
shared, rsf, activation, precision) is taken from the profile and cross-checked
against the HF config at start-up.

* DSV4 (`deepseek_v4_{flash,pro}_fp4`, `num_shared_experts=1`): attention worker runs
  embed, mHC, attention + compressor + indexer, router (incl. hash layers), final
  `hc_head`/norm/lm_head. Expert worker runs routed experts **and the shared expert**
  (replicated per expert rank, protocol top-k = 6+1, shared weight 1/rsf); the
  attention worker multiplies the combined output by rsf. SwiGLU clamp 10 is the
  kernel's `activation_clamp`.
* K3 (`kimi_k3_fp4`, hidden=3584): attention worker runs router (on 7168), latent
  down-proj, the 2 shared experts, latent RMSNorm + up-proj after combine, dense
  layer 0, KDA/MLA, attention residuals. Expert worker runs only the 896 routed
  latent experts (top-16, FP4). Dense layer 0 is skipped on the F side (MoE layer
  ids are passed explicitly; the persistent EG kernel serves 92 layers). The MegaMoE
  epilogue has no SiTU, so routed experts use SiLU as in the measured profile
  (documented deviation; colocated reference path keeps SiTU).

## 4. Cache / state design

Each model gets its own pool + attention backend (selected by architecture);
`page_table` keeps token slots, and per-request state is indexed by `table_idx`
(`max_running_req + 1` slots, last one = dummy request for graph padding).

* DSV4 (`page_size = 256`, as sglang: a page holds 64 C4 entries and 2 C128 entries):
  - per layer one KV buffer `[slots, 576]` (FP8 e4m3 by default, 576 B per row;
    BF16 with `MINISGL_DSV4_KV_DTYPE=bf16`) = SWA ring region `R x 128`
    (slot = `R0 + table_idx*128 + pos % 128`) + scratch + compressed region
    (`token_slot // ratio`, i.e. N/4 or N/128 entries);
  - per C4 layer a paged indexer-key cache: per page 64 x 128 FP8 keys followed by
    64 FP32 scales (DeepGEMM's fused layout, 132 B / entry);
  - per compressed layer compressor state (kv, score) `[R, coff*ratio, coff*512]` fp32
    (+ `[R, 8, 256]` for the indexer compressor).
* K3 (`page_size = 64`, trtllm-gen MLA pages): MLA latent cache `[24, N, 576]` bf16
  (ckv 512 + k_rot 64); KDA recurrent state `[69, R, 96, 128, 128]` fp32 and conv
  state `[69, R, 3, 36864]` bf16 (sglang's layout, feature dim contiguous).
* Sizing: the engine/AFD attention state asks the pool for
  `(bytes_per_token, bytes_per_request)` and allocates fixed per-request state first,
  then tokens from the remaining memory budget. Prefix caching is not supported for
  these models (use `--cache-type naive`).

## 5. Testing plan

1. Reduced depth (2-4 layers; K3 includes KDA, MLA, dense and MoE layers) colocated
   prefill+decode with dummy weights on one node.
2. Module numerics against torch transcriptions of the official references (and the
   sglang Triton KDA / flashinfer mHC kernels in the `sglang-cap` env where
   applicable): DSV4 attention (SWA/C4/C128), indexer, mHC, MoE; K3 KDA, MLA,
   LatentMoE; full reduced-depth forward (prefill and step-by-step decode).
3. AFD reduced depth on 2 nodes, then full depth dummy weights (e.g. 56A16F), step
   time vs batch from the coordinator CPU trace.
4. Real-weight name mapping / format conversion tested on single downloaded shards.

## 6. Implementation status (branch `afd/dsv4-k3-models`)

Code map:

| area | files |
|---|---|
| config / registry | `models/config.py` (`deepseek_v4`, `kimi_k3` parsing), `utils/hf.py` (raw config.json, no remote code), `models/register.py` |
| models (colocated) | `models/deepseek_v4.py`, `models/kimi_k3.py`, shared MoE helpers `models/moe_common.py` |
| models (AFD attention stage) | `models/deepseek_v4_afd.py`, `models/kimi_k3_afd.py`, contract `models/afd_hybrid.py` |
| kernels | `kernel/dsv4_triton.py` (mHC, q/kv norm+RoPE+quant-sim, inverse RoPE, compressor decode), `kernel/kda_triton.py` (short conv, KDA recurrence, gated RMSNorm), `kernel/kimi_triton.py` (attention residuals, SiTU) |
| caches / backends | `kvcache/dsv4_pool.py`, `kvcache/kimi_pool.py`, `attention/dsv4.py`, `attention/kimi.py` |
| weights | `models/dummy_init.py` (dummy), `models/ckpt_loaders.py` (real checkpoints) |
| AFD wiring | `afd_worker_base.py` (`_init_hybrid_runtime`), `afd_attention_worker.py` (latent dispatch / post-combine hook), `afd_expert_worker.py` (MoE layer ids), `moe/megamoe_m2n_afd.py` (packed MXFP4 registration), `afd_worker_launcher.py` |
| tests | `tests/dsv4_k3/` (kernels, numerics vs reference transcriptions, loader round trip / real shards) |
| AFD bench | `scripts/experiments/afd/dsv4_k3/` (Ray bootstrap sbatch, sweep runner, client, analyzer, config-only model dirs) |

Deviations / limits (current state):

* MTP layers are not implemented (DSV4 `mtp.*` weights are skipped).
* K3 routed experts use SiLU in the MegaMoE epilogue (both colocated MegaMoE and
  AFD) because the measured `kimi_k3_fp4` stage has no SiTU; the torch executor
  (`MINISGL_MOE_EXEC=torch`) keeps SiTU.
* Colocated MegaMoE (`fp8_fp4_mega_moe`) faults with a synthetic shared slot at
  EP=1, so colocated DSV4 runs the shared expert locally (FP8 GEMM); in AFD the
  shared expert is on the expert workers as in the profile.
* Prefill of the DSV4 compressor/indexer and the K3 KDA recurrence are
  per-request / token-sequential kernels (correct, not tuned); decode is the
  optimised path (CUDA graphs on both sides in AFD).
* Prefix caching is not supported (`--cache-type naive`).
* K3 tokenizer needs `tiktoken`, which the `minisgl-cuda130` env lacks; dummy
  runs set `MINISGL_TOKENIZER_PATH` to another tokenizer directory.
* Multi-node AFD runs can hit the intermittent MegaMoE M2N cross-node
  `DeepGEMM NVLink barrier timeout (300s)` (the "bug 5" seen with the other
  models' e2e runs): some ranks never observe the remote signals and every
  worker traps after 300 s. The bench runner detects it and the case can be
  retried; sweeps run one server per batch size to bound the loss.
* Dummy weights + greedy decoding collapse onto a single token (and thus onto
  one set of experts); the sweep client samples (temperature 1) instead.

## 7. How to run

Model directories: a real HF checkpoint directory, or (dummy weights) a
config/tokenizer-only copy made by
`scripts/experiments/afd/dsv4_k3/make_model_dir.py --src HF_DIR --dst DIR [--layers N]`
(`--layers` keeps every layer kind: DSV4 SWA/C4/C128 + hash routing, K3
KDA/MLA/dense/LatentMoE).

Colocated (one node, TP = EP = 4):

```bash
# DeepSeek-V4 (Flash or Pro)
python -m minisgl --model-path $DIR --dummy-weight --tensor-parallel-size 4 \
  --attention-backend dsv4 --page-size 256 --max-running-requests 64 \
  --cuda-graph-max-bs 64 --max-seq-len-override 4096 --memory-ratio 0.85
# Kimi-K3 (MINISGL_TOKENIZER_PATH=<other tokenizer dir> if tiktoken is missing)
python -m minisgl --model-path $DIR --dummy-weight --tensor-parallel-size 4 \
  --attention-backend kimi --page-size 64 --max-running-requests 64 \
  --cuda-graph-max-bs 64 --max-seq-len-override 4096 --memory-ratio 0.85
```

The engine forces the attention backend and page size of these models (DSV4:
`dsv4`, 256; K3: `kimi`, 64). Drop `--dummy-weight` to load a real checkpoint
(`models/ckpt_loaders.py`).
`MINISGL_MOE_EXEC=torch` switches the routed experts from colocated MegaMoE to
the torch reference executor.

AFD (attention workers + MegaMoE expert workers, `MINISGL_AFD_MOE_BACKEND=megamoe_m2n`
is set by the runner): `submit_afd_hybrid.sbatch` starts a Ray cluster (GPU-less
head + GPU workers; `HEAD_GPUS=1` to use the head's GPUs too) and runs
`afd_cases.sh`, which starts one server per case and sweeps the per-attention-GPU
batch size; `step_times.json` per case holds the median decode step time.

```bash
CONDA_ROOT=... FASTAFD_ROOT=$PWD ENV_SETUP=<optional env file> \
RUN_SCRIPT=$PWD/scripts/experiments/afd/dsv4_k3/afd_cases.sh \
DSV4K3_META_DIR=<dir with DeepSeek-V4-Flash, DeepSeek-V4-Pro, Kimi-K3> AFD_OUT_DIR=<out> \
CASES="k3:16:16:8/16/32/64 dsv4_pro:24:8:8/16/32/64/128 dsv4_flash:28:4:8/16/32/64/128" \
K3_TOKENIZER_PATH=<tokenizer dir> \
sbatch -N 9 --segment=9 --account=<acct> --partition=<gb200 partition> --time=03:00:00 \
  scripts/experiments/afd/dsv4_k3/submit_afd_hybrid.sbatch
```

`LAYERS=4` gives the reduced-depth variant; `MODEL_DIR_<model>=<checkpoint>` with
`DUMMY_WEIGHT=0` serves real weights.

## 8. Validation summary (dummy weights unless noted)

* Kernels (`tests/dsv4_k3/test_kernels.py`): 23/23 checks against torch
  references (mHC, q/kv norm+RoPE+quant simulation, inverse RoPE, compressor
  decode, short conv, KDA recurrence, gated RMSNorm, attention residuals, SiTU).
* Module / full-forward numerics (`tests/dsv4_k3/test_numerics.py`, 4 layers, BF16
  non-expert weights, torch MoE executor, prompts of 9/150/300 tokens + 4 decode
  steps, against torch transcriptions of the official references): attention
  modules rel-L2 4-6e-3 (DSV4 SWA/C4/C128, K3 MLA), KDA 3e-4-2e-3, MoE / LatentMoE
  2-5e-3; full-forward logits 0.5-2.2e-2 with the reference forced onto the
  engine's routed-expert ids. Without forcing, single prompts can differ by up to
  ~0.4 rel-L2 because random routers produce top-k near-ties (K3 routing-set
  agreement 0.54-0.97 over 300 tokens); CUDA-graph decode equals eager decode
  bit for bit.
* Real weights (truncated depth, `tests/dsv4_k3/test_ckpt_loader.py real`):
  DSV4-Flash layers {0, 2, 3} + embed/head: logits rel-L2 1.8-2.4e-2 vs the
  reference (FP8 W8A8 GEMMs in the engine); Kimi-K3 layers {0, 3} + embed/head:
  6.4-6.9e-3, argmax agreement 1.0. Loader round trip is bit-exact for the
  colocated models, the AFD attention stages and the expert-worker slices.
* sglang 0.5.18 kernels (`tests/dsv4_k3/test_vs_sglang.py`): KDA prefill
  (`causal_conv1d_fn` + `chunk_kda`, varlen, with/without initial state) and
  decode (`causal_conv1d_update` + `fused_sigmoid_gating_delta_rule_update`)
  agree to rel-L2 2.8e-3-5.7e-3 (outputs, final states; conv state exact). The
  mHC comparison needs sglang's tilelang JIT, which did not compile in the
  available sglang env (CUDA header/toolchain mismatch).
* AFD vs colocated on real weights (truncated checkpoints, greedy, 8 prompts x
  16 tokens, 2A2F): Kimi-K3 identical in 7/8 prompts (92 % of positions).
  DeepSeek-V4-Flash 26 %, rising to 69 % when the colocated shared expert also
  uses the MXFP4-requantised weights that the MegaMoE expert workers run
  (colocated FP8 vs MXFP4 shared alone: 30 %), i.e. the difference is the
  shared-expert precision imposed by the measured MegaMoE stage.
* Full DeepSeek-V4-Flash checkpoint (46 shards, read in place): colocated TP4 and
  AFD 4A4F (two nodes, `HEAD_GPUS=1`) both answer the fixed prompt set sensibly
  (Paris, Fibonacci code, special relativity, Jupiter, ...). Greedy agreement
  over 8 x 32 tokens: FP8 vs BF16 KV rows (colocated) 68 %, AFD vs colocated
  with BF16 KV 63 % and with FP8 KV 47 %; the runs agree on the answers and part
  in the open-ended continuations, where near-ties flip with the cache precision
  and the MXFP4 shared expert of the MegaMoE stage.
* Real routing skew (DeepSeek-V4-Flash, AFD 4A4F, 1152 real text prompts cut to
  2048 tokens, 16 sampled decode tokens, 4.46 M expert assignments): aggregated
  expert-load CV 23 %, per-layer CV 150 % (mean) / 146 % (median), max/mean load
  per layer 14.9 on average; the hash-routed layers 0-2 have CV 65-70 %. Saved as
  `real_routing_dsv4_flash_perlayer.json` (schema `fastafd.expert-load.v2`) next
  to the Qwen3 routing file.

### AFD step time vs batch (full depth, dummy weights, GB200 NVL72)

Median decode step from the coordinator trace; prompt 1024 tokens, 64 sampled
output tokens, attention TP1, one server per batch size, `--afd-num-mb 2`
unless noted, MegaMoE dispatch bucket 2048.

| model, shape | 8 | 16 | 32 | 64 | 128 (per attention GPU) |
|---|---:|---:|---:|---:|---:|
| Kimi-K3, 16A16F | 73.4 ms | 76.3 | 82.5 | 100.8 | - |
| DeepSeek-V4-Pro, 24A8F | 60.3 ms | 62.8 | 65.0 | 108.5 | 122.3 |
| DeepSeek-V4-Flash, 28A4F | 33.3 ms | 34.3 | 35.3 | 52.7 | 59.5 |
| DeepSeek-V4-Flash, 2A2F, mb=1 | 20.6 ms | 21.9 | 31.5 | 36.2 | 45.2 |

These were measured with the first, untuned attention-side kernels (Ray head
without GPUs, 9 nodes). With the decode kernels of section 9 and the head's GPUs
in use (`HEAD_GPUS=1`, 8 nodes = 32 GPUs, one NVL72 domain):

| model, shape | 8 | 16 | 32 | 64 | 128 (per attention GPU) |
|---|---:|---:|---:|---:|---:|
| Kimi-K3, 16A16F | 48.4 ms | 50.0 | 56.6 | 64.6 | - |
| DeepSeek-V4-Flash, 28A4F | 14.3 ms | 14.6 | 16.0 | 20.5 | 33.0 |

(1.5-1.6x faster for K3, 1.8-2.6x for DSV4-Flash; FP8 KV rows for DSV4-Flash, its
BF16-row variant measured 15.5 / 15.2 / 16.5 / 20.7 / 33.1 ms.) The one-node
DeepSeek-V4-Flash 2A2F mb=1 point runs at 11.8 / 12.9 / 14.0 / 16.3 / 20.6 ms
(BF16 KV rows).

## 9. Attention-side decode kernels vs sglang 0.5.18 (GB200)

Per-layer time on the attention GPU, CUDA-graph replay, dummy weights, decode
batch B per attention GPU at context C (`tests/dsv4_k3/bench_aside.py`). Model
averages weight the layer types by the full model (Flash: 2 SWA + 21 C4 + 20
C128; K3: 69 KDA + 24 MLA).

**Full A-side per layer** (attention + mHC / attention residuals + norms + router
+ the attention-side MoE parts: K3 latent projections and shared experts), us:

| model | TP | B | C=4K before / after | C=32K before / after |
|---|---:|---:|---:|---:|
| DSV4-Flash | 1 | 8 | 392 / 148 | 434 / 152 |
| DSV4-Flash | 1 | 32 | 580 / 150 | 681 / 154 |
| DSV4-Flash | 1 | 128 | 812 / 178 | 1184 / 190 |
| DSV4-Flash | 2 | 8 | 409 / 151 | 450 / 155 |
| DSV4-Flash | 2 | 32 | 596 / 161 | 703 / 169 |
| DSV4-Flash | 2 | 128 | 817 / 194 | 1201 / 207 |
| Kimi-K3 | 1 | 8 | 410 / 269 | 474 / 287 |
| Kimi-K3 | 1 | 32 | 504 / 329 | 789 / 377 |
| Kimi-K3 | 1 | 128 | 779 / 545 | 1762 / 730 |
| Kimi-K3 | 2 | 8 | 362 / 224 | 389 / 237 |
| Kimi-K3 | 2 | 32 | 427 / 274 | 617 / 325 |
| Kimi-K3 | 2 | 128 | 607 / 413 | 1177 / 601 |

**Attention core vs sglang** (`--core`; boundary of the sglang kernel benchmark:
DSV4 compressors + indexer + top-k + sparse MLA, K3 KDA conv/recurrence/gated
norm and the MLA kernel; projections excluded), model-weighted, us / layer,
FastAFD / sglang:

| model | TP | B=8 (4K, 32K) | B=32 (4K, 32K) | B=128 (4K, 32K) |
|---|---:|---|---|---|
| DSV4-Flash | 1 | 22.8/27.3, 24.8/30.6 | 28.4/31.1, 34.5/36.7 | 44.3/37.3, 56.6/52.3 |
| DSV4-Flash | 2 | 21.8/27.1, 23.2/30.5 | 25.7/31.2, 32.7/37.0 | 42.0/37.1, 55.4/52.1 |
| Kimi-K3 | 1 | 28.6/30.7, 40.4/43.1 | 85.0/79.4, 131.2/140.8 | 286.0/264.0, 470.3/490.0 |
| Kimi-K3 | 2 | 21.5/25.2, 33.7/37.0 | 53.0/53.3, 99.5/104.3 | 163.8/154.6, 349.3/348.3 |

K3 is within 0.85-1.08x of sglang everywhere (per layer type: KDA 0.80-1.08x, MLA
0.81-1.15x). DSV4 is 0.76-0.94x of sglang for B <= 32 and 1.06-1.19x at B=128.
sglang's sparse MLA is FlashMLA's SM100 FP8 sparse decode (sgl_kernel, BF16 query,
584 B/token with per-64 UE8M0 scales), a compiled op outside this environment;
FastAFD runs flashinfer 0.6.6's trtllm-gen e4m3 sparse MLA on FP8 query + FP8 KV
rows instead (per-tensor scales; C4 layer at B=128: 22-23 us vs FlashMLA 27 us,
BF16 trtllm-gen 37-43 us). What remains at B=128 is the compressor decode (Triton
here, sglang's JIT c4/c128 v2 kernels plan their state in pages) and the per-layer
query cast and index assembly (2-3 us each). The FP8 query costs accuracy against
BF16 attention (module rel-L2 3.5e-2 vs 5e-3 with dummy weights, full-forward
logits 1.5e-2 vs 1.0e-2, argmax agreement 1.0); `MINISGL_DSV4_KV_DTYPE=bf16`
restores the BF16 path. K3 MLA at TP1
compares against sglang's CuTe-DSL MLA (flashinfer 0.6.17; sglang's trtllm-gen
default rejects 96 heads); at TP2 the 48 heads are padded to 64 (flashinfer 0.6.17
has a 48-head kernel) and MLA stays within 0.96-1.15x. The padded [q_abs | q_rope]
query is built in one Triton pass.

**Kernel map (decode)**, sglang kernel used unless noted:

| op | DSV4 | K3 |
|---|---|---|
| attention core | trtllm-gen e4m3 sparse MLA, FP8 query + KV (sglang: FlashMLA FP8 KV, BF16 query) | KDA: causal_conv1d_update + KDA packed decode (CUDA, B>=8) + gated RMSNorm; MLA: trtllm-gen (page 64) |
| indexer | q RoPE+Hadamard+FP8 (JIT), DeepGEMM paged FP8 MQA logits, top-k v2 (JIT) | - |
| compressor | BF16->FP32 cuBLAS GEMM; Triton state update + pooling and KV-row store (sglang: JIT c4/c128 v2); indexer keys: JIT fused_norm_rope_v2 | - |
| projections | DeepGEMM FP8 (wq_a+wkv merged, wq_b, wo_b), FP8 grouped einsum wo_a | cuBLAS BF16, q/k/v/g and q_a/kv_a merged (sglang: TGV CuTe-DSL GEMM for m <= 8) |
| residual mixing | mHC: DeepGEMM TF32 prenorm + fused Triton pre/norm (sglang: TileLang) | attention residuals: SM100 TMA kernel with fused RMSNorm |
| router | BF16->FP32 GEMM + fused Triton router (sqrt-softplus) | merged front GEMM (router + latent down + shared gate_up, FP32 out) + radix router |
| other | JIT q norm+RoPE, inverse RoPE, FP8 wo_a quant | SiTU (Triton), latent RMSNorm + up projection |


## 10. End-to-end AFD vs non-AFD campaign (GB200 NVL72)

Mirrors the Qwen3-235B campaign: steady-state decode step time vs batch for the
AFD units and the non-AFD baselines, real DeepSeek-V4-Flash weights (FP8 KV rows),
dummy Kimi-K3 weights with sampled decoding, real text prompts, one server per
batch size for the AFD points.

**Workload.** ISL/OSL 4K/1K and 32K/1K. Each request is prompted with
`ISL + OSL/2 - 32` real-text tokens (distinct prompts, consecutive lines of the
prompt file concatenated and cut) and decodes 64 tokens, so the timed steps run at
the mid-generation context `ISL + OSL/2`; the first and last 3 decode steps are
dropped (57 timed steps per point). Batch sizes are per GPU (AFD: per attention
GPU; colocated: requests / GPUs of the unit).

**Measurement.** AFD: wall-clock delta between consecutive decode
`AFD_Coordinator_ScheduleDps` starts in the coordinator CPU trace; FastAFD
colocated: the scheduler step trace (`MINISGL_SCHED_STEP_TRACE`, one record per
forward on rank 0); sglang: client-side inter-token latency (median over the
batch of the arrival time of token k, deltas between consecutive k; the server's
"Decode batch ... gen throughput" log lines give an independent mean).
Coordinator host time per step = the coordinator's own ranges of that step minus
the worker-reply waits inside `CompleteCollect`; eta = that time divided by the
requests in the step. Profiler decomposition: `MINISGL_AFD_TORCH_PROFILE=decode:<first>:<count>`
runs torch.profiler on the dp0 attention and expert workers for `count` decode
steps, each wrapped in `record_function("afd_step[id]")`; `e2e_profile.py`
attributes kernels to steps through their launch correlation ids and groups them
(attention core, dense GEMMs, norm/RoPE/mHC, router, quant + MoE communication,
MegaMoE expert kernel, sampling, copies), reporting GPU span, busy (union) and
idle per step.

**Colocated per-step breakdown.** `MINISGL_SCHED_TORCH_PROFILE=decode:<first>:<count>[:<bs,...>]`
runs the same torch.profiler window in the colocated scheduler (rank 0; one window
per listed batch size of a sweep, `minisgl/utils/torch_profile.py`). Each profiled
step is wrapped in `record_function("sched_step[i]")` and ended with a device
synchronize, so its kernels -- CUDA-graph replays included -- are attributed by
timestamp. For the label breakdown the sweep is repeated eagerly (`GRAPH_BS=4`
keeps the measured batches out of the graphs) with `MINISGL_NVTX_TORCH_RECORD=1`,
which mirrors the model's NVTX ranges into the trace; every kernel is attributed
to its innermost label and the labels are grouped into attention core
(`DSV4AttnCore` sparse attention + `DSV4AttnIndex` compressor/indexer), attention
projections + TP all-reduce (`DSV4Attn` rest, `DSV4AttnOut`), router (`DSV4Gate`),
routed MegaMoE stage (`DSV4Routed`: FP8 cast, M2N kernel, all-gather), shared expert
(`DSV4Shared`), mHC / norms / residual (`DSV4Layer` rest), `DSV4Head` and everything
outside the layers (embedding, sampler, copies). The eager run gives kernel time
per bucket; the graph run gives the per-step total and the idle time that the
measured step contains. `MINISGL_DSV4_COLO_SHARED_SP` (default 1) runs the colocated
shared expert on this rank's token shard inside the MegaMoE executor (before its
all-gather; replicated weights, one bf16 add, the same rounding as before) instead
of recomputing it for all tokens on every rank (`=0`, the configuration of the first
colocated sweep): -0.7 ms per step at 256 tokens and -2.2 ms at 1024 tokens.

**sglang baseline.** `sgl_colocated_bench.sh` launches sglang 0.5.18 with
`--tp 4 --ep-size 4 --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85
--moe-runner-backend deep_gemm --max-running-requests <4 x b_max> --context-length
<prompt + 64 + 256> --cuda-graph-bs-decode <4 x b for every swept b>`; the
resolved server arguments are attention backend `dsv4` (page 256, FP8 KV with
scaling factors 1.0 -- sglang's warning), chunked prefill 16384, MoE a2a `none`,
FCFS scheduling, radix cache on (prompts are distinct), no DP attention. The
TP-sharded expert configuration (`--tp 4` without `--ep-size`) does not run in
0.5.18 with any MoE runner (auto/triton: hidden-size mismatch; deep_gemm: the DSV4
JIT activation requires `D // 8 >= num_experts`; flashinfer_trtllm: top-k output
format assertion; flashinfer_cutlass: `Fp8MoEMethod` has no `runner`), so the
colocated sglang numbers are EP4. A repeat of the 4K sweep in a second job agreed
within 1-3 % up to b = 192 and within 10 % at b = 256 (30.5 vs 33.7 ms). At 32K the
KV pool at `--mem-fraction-static 0.85` holds ~455 requests, so b >= 128 per GPU
queued and those points are not reported.

**Kimi-K3 non-AFD baseline (not obtained).** sglang 0.5.18 has a Kimi-K3 model, but
only as the multimodal architecture `KimiK3ForConditionalGeneration`: its tokenizer
manager instantiates the K3 multimodal processor, which needs the checkpoint's own
processor class (`media_processor`, remote code -- not executed here), and fails with
`AttributeError: TokenizersBackend has no attribute media_processor` before any
weight is loaded; `--skip-tokenizer-init` does not bypass it and
`--language-model-only` rejects the architecture. The attempted configuration was
4 nodes, `--tp 16 --dp 8 --enable-dp-attention --ep-size 16 --load-format dummy`
(attention TP2 per DP rank) with a2a `none` and `deepep`; the K3 colocated side
therefore stays AIC-based for sglang. FastAFD's colocated engine runs K3 across nodes
with `--launch-backend ray` (TP = EP = 16 or 32, one server per batch size); its MLA
latent KV is replicated across the TP ranks (27 KB/token/rank), which bounds the
resident batch to b <= 12 per GPU at 4.6K context (see the K3 table below).

**Launch failures in the MegaMoE M2N stage (K3, and rarely DSV4).** 16A16F runs at
4K for b <= 32 per attention GPU; b = 64 (TP1), b = 96/128 (TP2) and every 32K case
died with `CUDA error: unspecified launch failure`, as did DSV4 16A4F b = 96 (decode
graph replay) and 12A4F b = 256 (prefill). Attribution with `CUDA_LAUNCH_BLOCKING=1`
forwarded to the workers: the 32K TP2 b8 case then fails at launch on both sides of
the expert exchange -- attention workers in `mega_moe_m2n_ag`
(`sm100_mega_moe_m2n.hpp:107`, eager prefill dispatch) and expert workers in the
persistent `mega_moe_m2n_eg` kernel (`:348`), `CUDA_ERROR_LAUNCH_FAILED` -- while
the 4K TP1 b64 case runs to completion under serialised launches (73.8 ms, not a
measurement). The fault is therefore a device fault inside the vendored MegaMoE
M2N kernels (timing-dependent at 4K, reproducible at 32K TP2), surfaced earlier at
the next synchronisation point (which happened to be `attn_res_fused_tma`). The
standalone attention-side harness (`tests/dsv4_k3/bench_aside.py`, no M2N stage)
runs K3 at b 32-256 / 4K and b 8-64 / 32K-64K with compute-sanitizer memcheck
clean. Raising the KV budget (`MEMORY_RATIO=0.95`) only moved the capacity
failures (2423 / 15441 pages per rank at b = 64 TP1 / b = 128 TP2 under 0.85) into
the same kernel fault. The kernel fix is in the DeepGEMM-based M2N CUDA code
(kernel-owner territory). Setting experiments on 16A16F (FP8 projections): one
microbatch does not avoid it (b 64 TP1 with mb 1 and mb 2 at memory ratio 0.95 both
died; b 96 TP2 mb 1 passed, 77.5 ms); a 4x exchange bucket
(`MINISGL_AFD_MAX_COMM_TOKENS=8192`) is memory-infeasible (attention-side OOM at TP1
b 64 and TP2 b 128); 1024-token prefill chunks (`AFD_MAX_BATCHED_TOKENS=1024`,
exchange bucket 1024) passed b 64 TP1 and b 128 TP2 with two microbatches and b 8 / 32
of the re-measured series, but b 16 died the same way -- the fault is data- and
timing-dependent and smaller chunks only lower its rate (DSV4 16A4F b 192 also passed
with them after two failures). Prefill chunking does not touch the timed decode steps,
so chunk-1024 points keep their labels, with the chunk size in each point note; a
failed batch gets one retry. A 4096-token bucket with 1024-token chunks (`MINISGL_AFD_MAX_COMM_TOKENS=4096`,
+10 GiB per side) passed 16A16F b 16 / 32 / 64 (43.33 / 47.02 / 54.47 ms) where bucket 1024
had failed b 16 and b 128 TP2 once each; it is the setting for the remaining K3 jobs and
retries (recorded in the point notes).

**Colocated breakdown, measured** (TP4 = EP4, real weights, 4K/1K, GB200; ms per
decode step, medians over 5 profiled steps; `graph` = CUDA graphs as in the sweep,
`eager` = labelled eager trace, kernel time only):

| b / GPU (tokens / step) | step (sweep) | GPU span, graph | busy | idle in span | attention core | attention proj + all-reduce | router | routed MegaMoE | shared expert | mHC / norms | lm_head + outside layers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 (256), replicated shared expert | 21.23 | 21.67 | 19.90 | 1.76 | | | | | | | |
| 64 (256), sharded shared expert | - | 20.95 | 19.25 | 1.71 | 3.50 | 3.13 | 0.51 | 8.54 | 1.50 | 1.56 | 0.62 |
| 256 (1024), replicated shared expert | 34.98 | 35.52 | 33.71 | 1.77 | | | | | | | |
| 256 (1024), sharded shared expert | - | 33.31 | 31.59 | 1.72 | 9.29 | 5.00 | 0.59 | 11.43 | 1.75 | 2.94 | 1.81 |

Inside the buckets at b = 256: routed MegaMoE = M2N kernel 8.34 (weight-streaming
bound: +1.0 ms for 4x the tokens of b = 64; 35 GB of MXFP4 expert weights per GPU
and step) + NCCL all-gather of the token shards 2.20 (LL ring kernel, ~50 us per 8 MB)
+ copies 0.76; attention core = sparse FMHA 6.82 + compressor/indexer GEMMs 1.14 +
KV ring/index copies 1.19; projections = FP8 GEMMs 2.06 + symmetric all-reduce 1.31
+ per-token FP8 casts 0.67 + q/kv norm-RoPE 0.65; mHC pre/post + hc-prenorm GEMM
2.94. 2.5k kernels per step (60 per layer); the graph replay leaves 1.7 ms of
launch gaps. The sharded shared expert saves 0.7 ms (b 64) and 2.2 ms (b 256) per
step (dense GEMM and the fp32 cast/add kernels of the replicated path) and is the
default. The remaining gap to sglang EP4 (18.2 ms at b 64, 30.5 ms at b 256 with FP8
experts, i.e. twice the expert bytes) is spread over the glue of the colocated path:
the per-layer all-gather after the SP MegaMoE stage (sglang's EP path keeps the
replicated tokens and all-reduces once), ~200 copy/cast kernels per step, separate
FP8 cast kernels in front of every GEMM, and the many small mHC/norm kernels -- none
of them a single fix.

At 32K/1K the first colocated sweep was host-bound: the GPU span per step at b = 8
was 15.5 ms (busy 13.9) while the scheduler's step period was 37-43 ms, growing
linearly with the batch (1.03 s at b = 192); the AFD units and sglang do not show
this (28A4F 32K b 128 = 33.6 ms = its 4K value). py-spy on the scheduler ranks put
~30 % of the host samples in `Req.append_host`: a `torch.cat` of the whole sequence
per request and decode step, which beyond 32K elements ATen runs on the OpenMP
pool (~0.4 ms per call). The append now writes into a preallocated per-request host
buffer (commit "core: amortised Req.append_host"); the 32K colocated sweep was
repeated with that fix (table below), the first sweep's points are kept under
`results/.../superseded/`.

**Results** (median decode step ms over 57 timed steps; GB200 NVL72, campaign of
2026-10-20; `summary.csv` / `step_time_tables.md` carry p90 and throughput).

DeepSeek-V4-Flash, real weights, batch per GPU (AFD: per attention GPU):

| b | 12A4F m2 (4K) | 24A8F m2 (4K) | 28A4F m2 (4K) | 28A4F m2 (32K) | 24A8F m2 (32K) | sglang EP4 (4K) | sglang EP4 (32K) | FastAFD colo TP4=EP4 (4K) | FastAFD colo (32K) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 14.51 | 14.34 | 15.33 | 15.66 | 14.76 | 10.55 | 10.75 | 14.22 | 14.68 |
| 16 | 15.46 | 14.66 | 16.52 | 16.89 | 15.27 | 12.63 | 13.08 | 15.67 | 16.29 |
| 32 | 17.15 | 15.21 | 18.66 | 18.91 | 16.01 | 14.41 | 14.87 | 17.60 | 18.32 |
| 64 | 20.82 | 15.90 | 23.30 | 23.35 | 16.32 | 18.15 | 19.32 | 20.43 | 22.03 |
| 96 | 25.14 | 16.29 | 28.40 | - | - | - | - | - | - |
| 128 | 29.39 | 18.96 | 33.53 | 33.58 | 19.41 | 22.81 | KV-limited | 24.86 | 27.22 |
| 192 | 38.35 | 25.27 | 44.09 | prefill > 30 min | 25.79 | 25.86 | KV-limited | 28.24 | 31.63 |
| 256 | 47.30 (retry) | 31.66 | 54.80 | prefill > 30 min | - | 30.51 | - | 32.58 | - |

Additional DeepSeek-V4-Flash units (real weights, m = 2, 4K/1K; same protocol):

| b | 16A4F m2 (5 nodes) | 20A4F m2 (6 nodes) | 16A8F m2 (6 nodes) | 28A8F m2 (9 nodes, mlp_dp 4 x tp 2) | 56A16F m2 (18 nodes, mlp_dp 8 x tp 2) |
|---:|---:|---:|---:|---:|---:|
| 8 | 14.73 | 14.99 | 14.36 | 14.32 | 14.57 |
| 16 | 15.89 | 16.12 | 14.71 | 14.68 | 14.54 |
| 32 | 18.13 | 17.70 | 15.23 | 15.25 | 15.91 |
| 64 | 23.65 | 20.62 | 16.13 | 16.00 | 23.67 |
| 96 | 29.47 (retry) | 24.08 (retry) | 17.81 | 17.23 | - |
| 128 | 35.33 | 27.64 | 20.27 | 20.69 | 41.18 |
| 192 | 47.03 (retry, 1K chunks) | 35.10 (retry) | 26.20 | 28.27 | 60.38 |
| 256 | 59.16 | 42.77 | 32.37 | 36.04 | 79.26 |

Doubling F from 4 to 8 at A = 16 saves 7.5 ms at b 64 and
15 ms at b 128; 28A8F vs 28A4F saves 7.3 / 12.8 / 18.8 ms at b 64 / 128 / 256. 16A4F
b 96 died once in decode graph replay (the MegaMoE M2N-stage race described below)
and then ran three times in a row: with a single microbatch 29.31 ms, with the
original two microbatches 29.47 and 29.49 ms. 16A4F b 192 died the same way in two attempts
(the asynchronous launch failure surfaced in the prefill forward right after graph capture) and
passed on a third with 1024-token prefill chunks (`AFD_MAX_BATCHED_TOKENS=1024`, exchange bucket
1024; see the K3 fault notes below): 47.03 ms. Prefill chunking does not touch the timed decode
steps, so the point is comparable; the chunk size is recorded in its note. 20A4F b 96 and b 192 died in the
M2N stage on their first attempt and passed on the retry (24.08 and 35.10 ms).

Per-GPU decode throughput at b = 256 (4K): 24A8F 6064 tok/s/GPU, 28A4F 4087, sglang
EP4 8389, FastAFD colocated 7858; 12A4F at b = 192: 3755. The sglang EP4 4K sweep
repeated in a second job agreed within 1-3 % up to b = 192 and 10 % at b = 256
(33.7 ms); the MegaMoE-a2a variant (`--moe-a2a-backend megamoe`) is within the same
spread (results `variants/sglang_ep4mm`). sglang 32K b >= 128 queued requests (KV
pool at mem-fraction 0.85 holds ~455 x 33K-token requests) and is not reported.
12A4F b = 256 died twice in prefill (`afd_ag_materialize` step 70, `CUDA error:
unspecified launch failure`) and passed on the third, unchanged attempt (47.30 ms) --
the MegaMoE M2N-stage race described below. 28A4F 32K b >= 192: the prefill of
5376 / 7168 requests x 33K tokens exceeds the per-case client timeout (30 min).

Kimi-K3 (dummy weights, sampled decoding, 4K/1K), FP8 block-128 attention-side
projections (`MINISGL_K3_ATTN_FP8=1`, the AIC convention; the BF16-projection points
48.50 / 50.32 / 57.62 ms at b 8 / 16 / 32 are kept under `superseded/`). Colocated =
FastAFD's engine with TP = EP across nodes (Ray launch), one server per batch size;
`KV-limited` = the requests cannot all be resident (MLA KV replicated per TP rank).
16A8F uses attention TP2 (F = 8 holds 168 GiB of routed weights per expert GPU, 2 GiB
free after init). `not resident (KDA state)`: at 4.6K context the attention side cannot
hold b >= 192 requests per GPU -- the per-request KDA recurrent state (429 MiB at TP1,
215 MiB at TP2, allocated for max-running-requests) plus the MLA KV replicated across
the TP ranks exceed the 0.85 x 184 GiB budget (b 192 TP2: 81 GiB state + 50 GiB weights
leave a 15.8 GiB KV pool = 132 resident requests of 384 per DP group, so the client's
requests queue and the case runs past a 45-min limit); TP4 would halve the state but
double the replicated KV. b 128 at TP2 is the largest resident batch (54 GiB state,
47 GiB KV).

| b | 16A16F m2 (8 nodes) | 16A8F m2 attn-TP2 (6 nodes) | 16A32F m2 (12 nodes, mlp_dp 16 x tp 2) | 28A16F m2 (11 nodes, mlp_dp 4 x tp 4) | colocated TP16 = EP16 (4 nodes) | colocated TP32 = EP32 (8 nodes) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 40.87 | 57.22 | 39.77 | 42.03 | 43.82 | 49.12 |
| 12 | - | - | - | - | 50.49 | 57.59 |
| 16 | 43.53 (retry) | 66.33 | 41.55 | 44.13 | KV-limited (216 of 256) | KV-limited (417 of 512) |
| 32 | 47.30 | 72.38 | 45.60 | 47.61 | KV-limited (160 of 512) | KV-limited (361 of 1024) |
| 64 | 54.80 | 76.95 | 53.55 | 55.02 | - | - |
| 96 | 70.29 (TP2) | 81.24 | 71.28 (TP2) | TP1 retry pending | - | - |
| 128 | 79.81 (TP2, retry) | 87.91 | 78.86 (TP2) | TP1 retry pending | - | - |
| 192 | not resident (KDA state) | not resident (KDA state) | not resident (KDA state) | not resident (KDA state) | - | - |
| 256 | not resident (KDA state) | not resident (KDA state) | not resident (KDA state) | not resident (KDA state) | - | - |

16A16F at chunk 1024 (b 16 and b 128 TP2 died once each with the 1024 bucket and passed on
the retry with the 4096 bucket); the earlier 2048-chunk series failed at every b >= 64 and
at 32K. 32K/1K (TP2, 4096 bucket): b 8 = 42.51 ms; b 16 and b 32 died once each (retry pending);
b 64 is not resident. F = 32 vs F = 16 on the same 16-GPU attention side saves 1-2.7 ms per
step at b 8-64 (TP1) and nothing at TP2 b 96-128, where the attention side dominates. 28A16F at attention TP2 (b >= 96) has
no node-local MLP factorisation (attn_dp 14 -> mlp_dp 2 x tp 8), so those batches run at TP1;
20A32F has none at any attention TP (replaced by 16A32F, pending). Colocated per-GPU throughput at b 12: EP16
238 tok/s/GPU, EP32 208; AFD 16A16F at b 32: 338 tok/s/GPU. No K3 baseline runs in
sglang 0.5.18 (remote code).

Qwen3-235B-A22B-FP8, real weights, AFD 16A8F (6 GPU nodes + GPU-less head), 4K/1K,
regression protocol: m = 2: b32 15.38, b64 18.03, b96 22.83, b128 26.16 ms; m = 3:
b32 20.97, b64 23.20, b96 26.75, b128 33.10 ms. m = 3 costs 4-7 ms per step more
than m = 2 at every batch (three smaller microbatches per step on the same 16
attention GPUs). One m = 2 b96 attempt and the first m = 3 b64/b128 attempts died
with CUDA launch failures / barrier timeouts after an earlier crash in the same Ray
cluster; fresh jobs reproduced every point.

**Host overhead.** Coordinator busy time per decode step divided by the requests in
the step converges to 3.86-3.96 us/request for the DSV4 units at b >= 128 (12A4F
3.86 / 3.88 at b 128 / 192; 24A8F 3.87 / 3.89 / 3.87 at b 128 / 192 / 256; 28A4F
3.96 / 3.95 / 4.06; 28A4F 32K b 128 4.10), i.e. the Qwen3-235B anchor eta = 3.87 us
holds for DeepSeek-V4-Flash; small batches are intercept-dominated (8-9 us at b = 8;
linear fit over all AFD points: busy = 0.56 ms + 3.84 us x requests). K3 b 32 4.79,
Qwen3 m3 b 96 3.65, m2 b 32 4.5.

**Profiler decomposition, 28A4F 4K (dp0 workers, 7 steps).** The expert worker runs
one persistent MegaMoE EG kernel that is busy for the whole step (b 32: 18.72 ms of
an 18.66 ms step; b 128: 33.66 of 33.53), so the unit is expert-bound at both
batches. The attention worker's span equals the step (18.7 / 33.65 ms, idle 0.7); its
own kernels take ~13.5 / 14.7 ms (FP8 dense GEMMs 5.66 / 5.99, mHC + norm + RoPE
3.27 / 3.42, attention 2.29 / 2.88, copies 1.16 / 1.41, FP8 casts 0.66 / 0.77, router
0.22 / 0.25), the rest is inside the M2N AG kernels waiting for the experts (20.3 /
48.8 ms summed over the two microbatch streams).

**Profiler decomposition, K3 16A16F b 32, 4K (dp0 workers, 7 steps).** Expert worker:
one MegaMoE EG kernel busy for the whole 57.8 ms step (step 57.6). Attention worker:
span 57.8, idle 0.3; kernel time summed over its two microbatch streams 107 ms, of
which M2N AG (incl. waiting for the experts) 43.5, BF16 dense GEMMs 46.2 (nvjet
projection kernels; K3's attention and KDA projections are not quantised), KDA /
attention kernels 11.3 (`kda_packed_decode` 5.3), copies 4.4, norms 1.0, router 0.7
-- both sides of the K3 unit are saturated at b 32, so the next unit (56A16F)
rather than the batch is where K3 throughput would come from.

**Rack units (56A16F, 40A32F).** The AFD runtime requires `attn_dp` and `mlp_dp` (and
`attn_tp` and `mlp_tp`) to be mutually divisible, so these units cannot use the
`mlp_dp = F` convention of the smaller units (56 is not a multiple of 16). They run
with F = `mlp_dp x mlp_tp` and full-world EP (`--afd-mlp-ep-size F`): every expert GPU
keeps E / F experts and receives the tokens of all attention ranks, i.e. the MegaMoE
stage is the EP = F stage of the microbenchmark; `mlp_tp` only changes the expert-side
grouping (each MLP DP group owns the requests of `attn_dp / mlp_dp` attention ranks)
and the node placement (a DP group's TP ranks share a node). The "dense QKVO DP/TP =
MLP factorisation" of the topology contract has no effect for DSV4 and K3: their
expert workers run the MegaMoE kernel only (the 28A4F and K3 decompositions show no
dense kernel on the expert side). `afd_cases.sh` derives the smallest valid `mlp_tp`:
DSV4 56A16F attn TP1: mlp_dp 8 x tp 2; K3 56A16F TP1: 8 x 2, TP2 (attn_dp 28): 4 x 4;
K3 40A32F TP1: 8 x 4, TP2 (attn_dp 20): 4 x 8. The factorisation was smoke-tested on
12A4F with mlp_dp 2 x mlp_tp 2 before the rack jobs were queued.

**Scripts** (`scripts/experiments/afd/dsv4_k3/`): `afd_cases.sh` (+ `E2E_OUT_DIR`,
`ISL`, `OSL`, attention TP as a 6th case field), `colo_bench.sh` (FastAFD
colocated), `sgl_colocated_bench.sh` + `sgl_bench_client.py` (sglang 0.5.18),
`e2e_point.py` (point JSONs), `e2e_profile.py`, `e2e_summary.py` (CSV + tables).

**Result schema** `fastafd.e2e-step-time.v1`, one JSON per point
(`<model>_<software>_<deployment>_isl<I>_osl<O>_b<B>.json`):

| field | content |
|---|---|
| `model`, `weights` | `deepseek_v4_flash` / `kimi_k3` / `qwen3_235b`; `real` / `dummy` |
| `deployment` | `kind` (afd / colocated), `software` (fastafd / sglang), `label`, `a`, `f`, `microbatches`, `attn_tp` or `tp`, `ep`; `gpus`, `nodes` |
| `workload` | `isl`, `osl`, `prompt_tokens`, `generated_tokens`, `context_at_measurement`, `batch_per_gpu`, `concurrent_requests`, `sampling`, `prompts` |
| `step_ms` | `median`, `p50`, `p90`, `mean`, `min`, `max`, `n`, `skipped`, `source`, `all_ms` |
| `throughput` | `tokens_per_s_total`, `tokens_per_s_per_gpu` (all GPUs of the unit), `tokens_per_s_per_attention_gpu` (AFD) |
| `host` | AFD: `coordinator_busy_ms_median`, `coordinator_us_per_request` |
| `memory` | attention / expert worker allocated, reserved and free GiB (AFD, from the worker logs), KV cache size, `gpu_used_gib_max` (nvidia-smi, single node) |
| `software` | `stack`, `commit` / `version`, `kv_cache_dtype`, `moe_precision`, `attention_kernels`, server flags |
| `run` | `job_id`, `run_dir`, `case`, `note` |

Profiler files `profile_<model>_<unit>_isl<I>_b<B>_<role>.json` (`fastafd.e2e-profile.v1`)
hold the per-step category sums, span / busy / idle and the top kernels; the
colocated files (`..._colo_tp4_ep4_isl<I>_b<B>_<graph_rep|graph_sp|eager_sp>.json`)
add `median_by_label` and `median_by_bucket`.
`summary.csv` and `step_time_tables.md` are regenerated by `e2e_summary.py`.

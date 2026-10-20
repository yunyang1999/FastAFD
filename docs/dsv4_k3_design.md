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
  main-q norm+RoPE, indexer-q RoPE+Hadamard+FP8 quant, (inverse) RoPE, top-k v2
  (fused top-512 + page-table transform) and the wo_a FP8 group quant.
* Vendored sglang Triton kernels (`kernel/sgl_triton`): `causal_conv1d_update`, the
  KDA packed-decode fallback (B < 8), the sigmoid-gated RMSNorm and the fused MoE
  router (sqrt-softplus for DSV4).
* DeepGEMM (minisgl's vendored build): FP8 GEMMs (`fp8_gemm_nt`, wq_a|wkv merged),
  the FP8 grouped einsum for DSV4 `wo_a`, the TF32 split-K mHC prenorm GEMM and the
  paged FP8 MQA logits of the DSV4 indexer (keys FP8 + FP32 scale in 64-entry pages).
* flashinfer 0.6.6: trtllm-gen MLA decode for K3 (paged BF16 latent, page 64; heads
  zero-padded to a power of two) and trtllm-gen sparse MLA for DSV4 (BF16 rows,
  per-head sinks; heads padded to 128 at large batch x top-k for the 2-CTA kernel).
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
  - per layer one KV buffer `[slots, 576]` bf16 = SWA ring region `R x 128`
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
  --attention-backend dsv4 --page-size 128 --max-running-requests 64 \
  --cuda-graph-max-bs 64 --max-seq-len-override 4096 --memory-ratio 0.85
# Kimi-K3 (MINISGL_TOKENIZER_PATH=<other tokenizer dir> if tiktoken is missing)
python -m minisgl --model-path $DIR --dummy-weight --tensor-parallel-size 4 \
  --attention-backend kimi --page-size 1 --max-running-requests 64 \
  --cuda-graph-max-bs 64 --max-seq-len-override 4096 --memory-ratio 0.85
```

Drop `--dummy-weight` to load a real checkpoint (`models/ckpt_loaders.py`).
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

These are end-to-end FastAFD steps of this implementation: the attention-side
kernels (KDA, MLA, DSV4 compressor/indexer/sparse MLA, mHC) are functional,
not tuned, so the small-batch floor is attention-side bound (K3: ~0.4 ms per
layer and microbatch over 93 layers).

## 9. Attention-side decode kernels vs sglang 0.5.18 (GB200)

Per-layer time on the attention GPU, CUDA-graph replay, dummy weights, decode
batch B per attention GPU at context C (`tests/dsv4_k3/bench_aside.py`). Model
averages weight the layer types by the full model (Flash: 2 SWA + 21 C4 + 20
C128; K3: 69 KDA + 24 MLA).

**Full A-side per layer** (attention + mHC / attention residuals + norms + router
+ the attention-side MoE parts: K3 latent projections and shared experts), us:

| model | TP | B | C=4K before / after | C=32K before / after |
|---|---:|---:|---:|---:|
| DSV4-Flash | 1 | 8 | 392 / 156 | 434 / 160 |
| DSV4-Flash | 1 | 32 | 580 / 159 | 681 / 163 |
| DSV4-Flash | 1 | 128 | 812 / 202 | 1184 / 213 |
| DSV4-Flash | 2 | 8 | 409 / 156 | 450 / 161 |
| DSV4-Flash | 2 | 32 | 596 / 171 | 703 / 179 |
| DSV4-Flash | 2 | 128 | 817 / 214 | 1201 / 228 |
| Kimi-K3 | 1 | 8 | 410 / 272 | 474 / 283 |
| Kimi-K3 | 1 | 32 | 504 / 331 | 789 / 380 |
| Kimi-K3 | 1 | 128 | 779 / 556 | 1762 / 737 |
| Kimi-K3 | 2 | 8 | 362 / 229 | 389 / 240 |
| Kimi-K3 | 2 | 32 | 427 / 281 | 617 / 327 |
| Kimi-K3 | 2 | 128 | 607 / 421 | 1177 / 619 |

**Attention core vs sglang** (`--core`; boundary of the sglang kernel benchmark:
DSV4 compressors + indexer + top-k + sparse MLA, K3 KDA conv/recurrence/gated
norm and the MLA kernel; projections excluded), model-weighted, us / layer,
FastAFD / sglang:

| model | TP | B=8 (4K, 32K) | B=32 (4K, 32K) | B=128 (4K, 32K) |
|---|---:|---|---|---|
| DSV4-Flash | 1 | 27.1/27.3, 29.2/30.6 | 37.6/31.1, 45.3/36.7 | 70.7/37.3, 84.3/52.3 |
| DSV4-Flash | 2 | 25.5/27.1, 27.6/30.5 | 34.7/31.2, 42.0/37.0 | 64.8/37.1, 79.5/52.1 |
| Kimi-K3 | 1 | 30.9/30.7, 43.5/43.1 | 89.6/79.4, 135.7/140.8 | 296.0/264.0, 480.1/490.0 |
| Kimi-K3 | 2 | 23.3/25.2, 36.9/37.0 | 56.2/53.3, 102.3/104.3 | 169.7/154.6, 352.3/348.3 |

K3 is within 0.92-1.13x of sglang everywhere. DSV4 is at parity for B <= 8,
1.1-1.2x at B=32 and 1.5-1.9x at B=128; the gap is the sparse MLA kernel:
sglang runs FlashMLA's SM100 FP8 sparse decode (sgl_kernel, 584 B/token), which
is a compiled op outside this environment; flashinfer 0.6.6's trtllm-gen sparse
MLA on BF16 rows is used instead (with the query padded to 128 heads at large
batch x top-k: 43 us vs 27 us for the C4 layer at B=128, C=32K). K3 MLA at TP1
compares against sglang's CuTe-DSL MLA (flashinfer 0.6.17; sglang's trtllm-gen
default rejects 96 heads); at TP2, padding 48 -> 64 heads costs 20-40% at
B <= 32 against flashinfer 0.6.17's 48-head kernel.

**Kernel map (decode)**, sglang kernel used unless noted:

| op | DSV4 | K3 |
|---|---|---|
| attention core | trtllm-gen sparse MLA (BF16; sglang: FlashMLA FP8) | KDA: causal_conv1d_update + KDA packed decode (CUDA, B>=8) + gated RMSNorm; MLA: trtllm-gen (page 64) |
| indexer | q RoPE+Hadamard+FP8 (JIT), DeepGEMM paged FP8 MQA logits, top-k v2 (JIT) | - |
| compressor | BF16->FP32 cuBLAS GEMM; Triton state update/store (sglang: JIT c4/c128 v2) | - |
| projections | DeepGEMM FP8 (wq_a+wkv merged, wq_b, wo_b), FP8 grouped einsum wo_a | cuBLAS BF16, q/k/v/g and q_a/kv_a merged (sglang: TGV CuTe-DSL GEMM for m <= 8) |
| residual mixing | mHC: DeepGEMM TF32 prenorm + fused Triton pre/norm (sglang: TileLang) | attention residuals: SM100 TMA kernel with fused RMSNorm |
| router | BF16->FP32 GEMM + fused Triton router (sqrt-softplus) | merged front GEMM (router + latent down + shared gate_up, FP32 out) + radix router |
| other | JIT q norm+RoPE, inverse RoPE, FP8 wo_a quant | SiTU (Triton), latent RMSNorm + up projection |


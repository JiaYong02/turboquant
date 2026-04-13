"""Check actual compression ratios per bit width."""
import torch
from turboquant.integrations.hf_cache import TurboQuantCacheLayer
from turboquant.bit_packing import packed_dim

HEAD_DIM = 64
NUM_HEADS = 2
SEQ = 64

for b in [4, 3, 2]:
    layer = TurboQuantCacheLayer(
        layer_idx=0,
        num_kv_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        key_bits=b,
        value_bits=b,
        seed=42,
        device='cpu',
    )
    k = torch.randn(1, NUM_HEADS, SEQ, HEAD_DIM)
    v = torch.randn(1, NUM_HEADS, SEQ, HEAD_DIM)
    layer.update(k, v)

    ks = layer._key_store
    vs = layer._value_store

    key_bytes = 0
    val_bytes = 0

    print(f"\nb={b}:")
    if ks.qjl_packed is not None:
        nb = ks.qjl_packed.nbytes
        key_bytes += nb
        print(f"  qjl_packed shape={ks.qjl_packed.shape}, {nb} bytes")
    if ks.gamma is not None:
        nb = ks.gamma.nbytes
        key_bytes += nb
        print(f"  gamma shape={ks.gamma.shape}, {nb} bytes")
    if ks.norms is not None:
        nb = ks.norms.nbytes
        key_bytes += nb
        print(f"  norms shape={ks.norms.shape}, {nb} bytes")
    if ks.mse_packed is not None:
        nb = ks.mse_packed.nbytes
        key_bytes += nb
        print(f"  mse_packed shape={ks.mse_packed.shape}, {nb} bytes")
    if vs.idx_packed is not None:
        nb = vs.idx_packed.nbytes
        val_bytes += nb
        print(f"  val idx_packed shape={vs.idx_packed.shape}, {nb} bytes")
    if vs.norms is not None:
        nb = vs.norms.nbytes
        val_bytes += nb
        print(f"  val norms shape={vs.norms.shape}, {nb} bytes")

    total_comp = key_bytes + val_bytes
    # FP16 equivalent: K + V, each [1, H, S, D] in float16
    fp16 = 2 * 1 * NUM_HEADS * SEQ * HEAD_DIM * 2
    ratio = fp16 / total_comp
    print(f"  key_bytes={key_bytes}, val_bytes={val_bytes}, total={total_comp}")
    print(f"  fp16_bytes={fp16}, ratio={ratio:.2f}x")

    # Theoretical
    mse_bits = b - 1 if b > 1 else 0
    qjl_bytes = packed_dim(HEAD_DIM, 1)   # ceil(D/8)
    mse_bytes = packed_dim(HEAD_DIM, mse_bits) if mse_bits > 0 else 0
    gamma_bytes = 4
    norm_k_bytes = 4
    val_idx_bytes = packed_dim(HEAD_DIM, b)
    norm_v_bytes = 4
    key_per = qjl_bytes + gamma_bytes + norm_k_bytes + mse_bytes
    val_per = val_idx_bytes + norm_v_bytes
    total_per = key_per + val_per
    fp16_per = HEAD_DIM * 2 + HEAD_DIM * 2  # K + V in float16
    print(f"  Theoretical per head/token: key={key_per}B val={val_per}B total={total_per}B fp16={fp16_per}B ratio={fp16_per/total_per:.2f}x")

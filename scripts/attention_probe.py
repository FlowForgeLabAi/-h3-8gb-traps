"""Measure MiniMax H3 attention at H3 geometry: kitchen INT8 vs KJ sage path,
with and without the head_chunks=10 split MiniMaxLowVRAMAttention imposes.

H3 geometry: heads=56, head_dim=128. Read-only synthetic tensors.
"""
import time
import torch
import torch.nn.functional as F

H, D = 56, 128
S = 8192
NCHUNK = 10
torch.manual_seed(0)


def groups(heads, n):
    hs = 0
    for i in range(n):
        he = hs + heads // n + (1 if i < heads % n else 0)
        yield hs, he
        hs = he


def bench(fn, iters=5):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    dev = "cuda"
    q = torch.randn(1, H, S, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(1, H, S, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(1, H, S, D, dtype=torch.bfloat16, device=dev)

    # fp32 SDPA reference
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    ref = ref.to(torch.bfloat16)

    print(f"H3 geometry: heads={H} head_dim={D} S={S} dtype=bfloat16  (packed H3 layout [B,H,S,D])")
    print()

    import comfy_kitchen as ck

    def kitchen_all():
        return ck.int8_attention(q, k, v)

    def kitchen_chunked():
        out = torch.empty_like(q)
        for a, b in groups(H, NCHUNK):
            out[:, a:b] = ck.int8_attention(q[:, a:b], k[:, a:b], v[:, a:b])
        return out

    # SDPA bf16 (what the workflow currently falls back to without any patch)
    def sdpa_bf16():
        return F.scaled_dot_product_attention(q, k, v)

    # KJ sage path on sm120 (exactly as _sageattn_int8_fp8_nhd does)
    sage_ok = True
    try:
        from sageattention.core import per_warp_int8_cuda, per_channel_fp8
        import importlib
        core = importlib.import_module("sageattention.core")
        qattn = getattr(core, "_qattn_sm89", None) or importlib.import_module(
            "sageattention.sm89_compile")._qattn_sm89

        def sage_heads(qa, ka, va, out):
            qn = qa.transpose(1, 2)  # -> NHD
            kn = ka.transpose(1, 2)
            vn = va.transpose(1, 2)
            qi, qs, ki, ks = per_warp_int8_cuda(
                qn, kn, km=kn.mean(dim=1, keepdim=True), tensor_layout="NHD",
                BLKQ=128, WARPQ=32, BLKK=64)
            vf, vs, _ = per_channel_fp8(vn, tensor_layout="NHD", scale_max=448.0, smooth_v=False)
            tmp = torch.empty(qi.size(), dtype=torch.bfloat16, device=dev)
            qattn.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(
                qi, ki, vf, tmp, qs, ks, vs, 0, 0, 2, D ** -0.5, 0)
            out.copy_(tmp.transpose(1, 2))

        def sage_all():
            o = torch.empty_like(q)
            sage_heads(q, k, v, o)
            return o

        def sage_chunked():
            o = torch.empty_like(q)
            for a, b in groups(H, NCHUNK):
                sage_heads(q[:, a:b], k[:, a:b], v[:, a:b], o[:, a:b])
            return o
    except Exception as e:
        sage_ok = False
        print("sage path unavailable:", type(e).__name__, str(e)[:120])

    rows = []
    for name, fn in [("SDPA bf16 (current fallback)", sdpa_bf16),
                     ("kitchen INT8, 56 heads at once", kitchen_all),
                     (f"kitchen INT8, head_chunks={NCHUNK}", kitchen_chunked),
                     ("KJ sage sm89, 56 heads at once", sage_all if sage_ok else None),
                     (f"KJ sage sm89, head_chunks={NCHUNK}", sage_chunked if sage_ok else None)]:
        if fn is None:
            continue
        try:
            out = fn()
            err = (out.float() - ref.float()).abs().max().item()
            rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            t = bench(fn)
            rows.append((name, t, err, rel, None))
        except Exception as e:
            rows.append((name, None, None, None, f"{type(e).__name__}: {str(e)[:110]}"))

    print(f"{'backend':<38} {'ms':>9} {'maxabs':>9} {'rel L2':>9}")
    for name, t, err, rel, e in rows:
        if e:
            print(f"{name:<38} {'--':>9} {'--':>9} {'--':>9}   {e}")
        else:
            print(f"{name:<38} {t*1000:9.1f} {err:9.4f} {rel:9.4f}")

    print()
    for name, t, err, rel, e in rows:
        if t:
            print(f"  {name:<38} peak-side alloc sample ok")
            break
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"free VRAM after: {free/1024**3:.2f} GiB")


if __name__ == "__main__":
    main()

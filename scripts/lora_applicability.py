"""Count exactly which LoRA tensors can and cannot apply to the pruned base model."""

import collections
import json
import struct

LORA = r".\ComfyUI\models\loras\minimax_h3_turbo_v4_step600_comfyui_T8-convert.safetensors"
BASE = r".\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors"
PREFIX = "diffusion_model."


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8"))


def main():
    L = header(LORA)
    B = header(BASE)
    lk = [k for k in L if k != "__metadata__"]
    bk = set(k for k in B if k != "__metadata__")

    # group LoRA keys into linear-layer modules
    modules = {}
    for k in lk:
        base = k
        for suf in (".lora_A.weight", ".lora_B.weight"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        modules.setdefault(base, {})[k.split(".")[-2]] = L[k]["shape"]

    ok = bad = orphan = 0
    bad_list = []
    ok_list = []
    for mod, ab in sorted(modules.items()):
        base_key = mod[len(PREFIX):] + ".weight"
        A, B_ = ab.get("lora_A"), ab.get("lora_B")
        if base_key not in bk:
            orphan += 1
            bad_list.append((base_key, "base 里没有这个键", A, B_, "orphan"))
            continue
        bshape = B[base_key]["shape"]
        if A and B_:
            lora_out, lora_in = B_[0], A[1]
            if [lora_out, lora_in] == list(bshape):
                ok += 1
                ok_list.append((base_key, bshape))
            else:
                bad += 1
                bad_list.append((base_key, bshape, A, B_, "形状不匹配"))
        else:
            bad += 1
            bad_list.append((base_key, bshape, A, B_, "缺 A 或 B"))

    total = ok + bad + orphan
    print("=== LoRA 模块级可应用性 ===")
    print("  LoRA 模块总数        : %d" % total)
    print("  OK 能应用            : %d  (%.1f%%)" % (ok, 100 * ok / total if total else 0))
    print("  XX 不能应用          : %d  (%.1f%%)" % (bad, 100 * bad / total if total else 0))
    print("  ?  基座无对应键      : %d" % orphan)
    print()

    print("=== 不能应用的模块（按类型归类）===")
    cat = collections.Counter()
    for base_key, bs, A, B_, why in bad_list:
        parts = base_key.split(".")
        key = ".".join(parts[-3:]) if len(parts) >= 3 else base_key
        cat["%s  [%s]" % (key, why)] += 1
    for k, v in cat.most_common(20):
        print("  %-52s %d" % (k, v))

    print()
    print("=== 能应用的模块（按类型归类，前 12）===")
    cat2 = collections.Counter()
    for base_key, bs in ok_list:
        parts = base_key.split(".")
        key = ".".join(parts[-3:]) if len(parts) >= 3 else base_key
        cat2[key] += 1
    for k, v in cat2.most_common(12):
        print("  %-52s %d" % (k, v))

    print()
    print("=== 结论 ===")
    n_adaln = sum(1 for b, _, _, _, _ in bad_list if "adaln_proj" in b)
    print("  adaln_proj 层无法应用 : %d 个" % n_adaln)
    print("  attention/MLP 层       : %d 个全部可应用" % ok)
    print()
    print("  -> turbo LoRA 的注意力/MLP 部分生效，但 AdaLN 调制部分被跳过。")
    print("  -> AdaLN 是蒸馏 LoRA 影响时间步调制的关键通路，被跳过意味着蒸馏只被部分应用。")


if __name__ == "__main__":
    main()


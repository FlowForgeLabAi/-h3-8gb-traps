"""Generate the three MiniMax H3 profiles as ComfyUI workflow JSONs.

Design decisions and their evidence (see _dsh_research/*.md for the full chain):

  steps 20 / sampler res_multistep   - Comfy-Org official template + official docs
                                       ("uses 20 steps by default")
  steps 8 / 4, sampler euler         - official turbo calibration table (LightX2V/ModelTC);
                                       euler is the vendor default for turbo
  LoRA strength 1.0                  - original LoRA author's requirement (max 1.05-1.2)
  explicit MiniMaxH3SigmaShift       - 544p LoRA family trains at 12/3, 768p family at 6/3;
                                       ComfyUI hardcodes 12/3, so the node makes it auditable
  MiniMaxLowVRAMAttention kept       - node source: "Output is identical to the unpatched model"
  MiniMaxChunkFeedForward kept       - node source: "...the output matches the unchunked model"
  no TE-Speed                        - closed-source .pyd, author calls it a speed/quality
                                       tradeoff, zero published benchmark
  no RTX VSR                         - measured: close() hangs forever on this machine
  short edge <= 768, area <= 768x1344, multiples of 32
                                     - official template constraint
  frames = 17k+5                     - official node tooltip, trained range ~124-362
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_workflow import build, object_info  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(HERE), "新工作流")

BASE_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
TE_INT8 = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
TE_NVFP4 = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "minimax_h3_turbo_v4_step600_comfyui_T8-convert.safetensors"

CRF = 12.0          # H.264 质量参数：默认 23，12 = 接近视觉无损

FRAMES_EXPR = "max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17"


def nodes_common(prefix, *, steps, sampler, scheduler, lora, lora_strength,
                 shift_video, shift_audio, head_chunks, chunk_chunks,
                 seq_threshold, attention_backend, tespeed, model, te, mp, aspect,
                 seconds, prompt_file, ref_image, shift_note):
    """Return (nodes, links) for one profile."""
    n = []
    l = []

    def N(i, type_, x, y, widgets=None, widget_override=None, title=None, size=None, mode=0):
        d = {"id": i, "type": type_, "pos": [x, y], "mode": mode}
        if widgets:
            d["widgets"] = widgets
        if widget_override is not None:
            d["widgets_values"] = widget_override
        if title:
            d["title"] = title
        if size:
            d["size"] = size
        n.append(d)
        return i

    # ---- model chain
    N(1, "UNETLoader", 40, 40, {"unet_name": model, "weight_dtype": "default"},
      title="主模型 (int8_convrot)")
    N(2, "ModelPatchTorchSettings", 40, 180, {"enable_fp16_accumulation": True})
    N(3, "MiniMaxH3SigmaShift", 40, 300,
      {"shift_video": shift_video, "shift_audio": shift_audio},
      title="shift %s/%s (%s)" % (shift_video, shift_audio, shift_note))
    N(4, "MiniMaxLowVRAMAttention", 40, 440, {"head_chunks": head_chunks},
      title="LowVRAMAttention (源码级无损)")
    N(5, "MiniMaxChunkFeedForward", 40, 580,
      {"chunks": chunk_chunks, "seq_threshold": seq_threshold},
      title="ChunkFeedForward (源码级无损)")

    # 模型链：主模型 -> torch设置 -> shift -> 低显存注意力 -> 分块FF
    l += [
        [1, "MODEL", 2, "model"],
        [2, "MODEL", 3, "model"],
        [3, "MODEL", 4, "model"],
        [4, "model", 5, "model"],
    ]

    model_out = 5
    if attention_backend:
        N(6, "ModelAttentionBackend", 40, 720, {"attention": attention_backend},
          title="注意力后端 = %s" % attention_backend)
        l.append([model_out, "model", 6, "model"])
        model_out = 6

    if lora:
        N(7, "LoraLoaderModelOnly", 40, 860,
          {"lora_name": lora, "strength_model": lora_strength},
          title="Turbo LoRA @%.2f" % lora_strength)
        l.append([model_out, "model", 7, "model"])
        model_out = 7

    if tespeed:
        N(9, "TESpeedMiniMaxH3", 40, 950,
          {"processing_control_value": 0.15, "processing_percent_1": 0.0,
           "processing_percent_2": 0.9, "mcs": 2, "device": "auto", "mode": "standard"},
          title="TE-Speed (有损, 闭源)")
        l.append([model_out, "model", 9, "model"])
        model_out = 9

    N(8, "BasicGuider", 40, 1100)
    l.append([model_out, "model", 8, "model"])

    # ---- conditioning
    N(10, "CLIPLoader", 480, 40,
      {"clip_name": te, "type": "minimax", "device": "default"},
      title="文本编码器")
    N(11, "PrimitiveStringMultiline", 480, 190, {"value": prompt_file},
      title="提示词")
    N(12, "LoadImage", 480, 400, {"image": ref_image, "upload": "image"},
      title="参考图 (首帧)")
    N(13, "ResolutionSelector", 480, 590,
      {"aspect_ratio": aspect, "megapixels": mp, "multiple": 32, "preview": None},
      widget_override=[aspect, mp, 32],
      title="分辨率 (短边<=768, 32倍数)")
    N(14, "PrimitiveFloat", 480, 750, {"value": seconds}, title="时长(秒)")
    N(15, "ComfyMathExpression", 480, 860,
      {"expression": FRAMES_EXPR},
      widget_override=[FRAMES_EXPR],
      title="帧数 = 17k+5")
    N(16, "MiniMaxH3ReferenceToVideo", 480, 1030,
      {"prompt": "", "width": 1344, "height": 768, "length": 124,
       "ref_image_size": "match", "ref_images": None},
      widget_override=["", 1344, 768, 124, "match"],
      title="MiniMax H3 参考视频")

    l += [
        [10, "CLIP", 16, "clip"],
        [11, "STRING", 16, "prompt"],
        [13, "width", 16, "width"],
        [13, "height", 16, "height"],
        [15, "INT", 16, "length"],
        [14, "FLOAT", 15, "values.a"],
        [16, "positive", 8, "conditioning"],
    ]
    # VAEs
    N(17, "VAELoader", 940, 40, {"vae_name": VIDEO_VAE}, title="视频 VAE")
    N(18, "VAELoader", 940, 180, {"vae_name": AUDIO_VAE}, title="音频 VAE")
    l += [[17, "VAE", 16, "vae"], [18, "VAE", 16, "audio_vae"]]
    l += [[12, "IMAGE", 16, "ref_images.ref_image_0"]]

    # ---- sampling
    N(20, "RandomNoise", 940, 340)
    N(21, "KSamplerSelect", 940, 450, {"sampler_name": sampler})
    N(22, "BasicScheduler", 940, 560,
      {"scheduler": scheduler, "steps": steps, "denoise": 1.0},
      title="%d 步 / %s / %s" % (steps, sampler, scheduler))
    N(23, "SamplerCustomAdvanced", 940, 730)
    l += [
        [20, "NOISE", 23, "noise"],
        [8, "GUIDER", 23, "guider"],
        [21, "SAMPLER", 23, "sampler"],
        [22, "SIGMAS", 23, "sigmas"],
        [16, "LATENT", 23, "latent_image"],
        [model_out, "MODEL", 22, "model"],
    ]

    # ---- output
    #
    # VHS_VideoCombine replaces CreateVideo+SaveVideo. Evidence (bench/crf_probe.py):
    #   SaveVideo's h264 re-encode path is DEAD in this environment --
    #   avcodec_open2("libx264", {}) fails even with no options, because PyAV's
    #   bundled libx264 cannot open. Its only working mode is format=auto, which
    #   "preserves a compatible source stream" and therefore cannot change quality.
    #   VHS_VideoCombine shells out to the ffmpeg executable instead, and its crf
    #   measurably works: crf=12/19/40 produced 570,872 / 287,454 / 40,296 bytes.
    #
    # It also takes IMAGE + AUDIO directly, so the CreateVideo node is unnecessary.
    N(30, "VAEDecode", 1400, 40)
    N(31, "VAEDecodeAudio", 1400, 180)
    N(33, "VHS_VideoCombine", 1400, 400,
      {"frame_rate": 24.0, "loop_count": 0, "filename_prefix": "H3_%s" % prefix,
       "format": "video/h264-mp4", "pingpong": False, "save_output": True,
       "pix_fmt": "yuv420p", "crf": CRF, "save_metadata": True, "trim_to_audio": False},
      widget_override=[24.0, 0, "H3_%s" % prefix, "video/h264-mp4", False, True,
                       "yuv420p", CRF, True, False],
      title="VHS 输出 (h264, crf=%g)" % CRF)
    l += [
        [23, "output", 30, "samples"],
        [23, "output", 31, "samples"],
        [17, "VAE", 30, "vae"],
        [18, "VAE", 31, "vae"],
        [30, "IMAGE", 33, "images"],
        [31, "AUDIO", 33, "audio"],
    ]
    return n, l


def main():
    oi = object_info()
    os.makedirs(OUTDIR, exist_ok=True)
    written = []

    CK = "comfy kitchen attention"
    profiles = [
        # name, steps, sampler, lora, strength, shift, attn_backend, tespeed, note
        ("Quality", 20, "res_multistep", None, None, (12.0, 3.0), CK, False,
         "官方基线 20 步无 turbo；无任何有损加速；纯无损路径（最慢）"),
        ("Balanced", 8, "euler", TURBO_LORA, 1.0, (12.0, 3.0), CK, False,
         "8 步 turbo @1.0；无损注意力 + 分块；无 TE-Speed（默认推荐档）"),
        ("Fast", 4, "euler", TURBO_LORA, 1.0, (12.0, 3.0), CK, True,
         "4 步 turbo @1.0 + TE-Speed；最快但唯一含近似缓存（闭源有损）"),
    ]

    for name, steps, sampler, lora, strength, shift, attn, tespeed, note in profiles:
        nodes, links = nodes_common(
            name, steps=steps, sampler=sampler, scheduler="simple", lora=lora,
            lora_strength=strength or 1.0, shift_video=shift[0], shift_audio=shift[1],
            head_chunks=10, chunk_chunks=4, seq_threshold=4096,
            attention_backend=attn, tespeed=tespeed,
            model=BASE_MODEL, te=TE_NVFP4, mp=0.75,
            aspect="3:4 (Portrait Standard)", seconds=10,
            prompt_file="（在此粘贴提示词）",
            ref_image="微信图片_20260913134501_1457_5.jpg",
            shift_note=note)
        cfg = {"nodes": nodes,
               "links": [(a, ao, b, bi) for (a, ao, b, bi) in links]}
        try:
            wf = build(cfg, oi)
        except Exception as e:
            print("  [FAIL] %-10s %s: %s" % (name, type(e).__name__, e))
            continue
        path = os.path.join(OUTDIR, "H3_%s.json" % name)
        json.dump(wf, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("  [OK]   %-10s %2d 节点 %2d 连线  -> %s" % (
            name, len(wf["nodes"]), len(wf["links"]), os.path.basename(path)))
        written.append(path)

    print()
    print("输出目录:", OUTDIR)
    return written


if __name__ == "__main__":
    main()

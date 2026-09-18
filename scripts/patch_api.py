"""Patch generated API prompts: fixed prompt + a working, crf-controlled encoder.

Two hard-won facts drive this file.

1. SaveVideo's nested dynamic combos are spelled with DOTTED top-level keys
   (comfy_api/latest/_io.py: finalize_prefix joins with "."):
       format / format.codec / format.codec.encoding / format.codec.encoding.crf
   Flat keys are silently dropped; nesting everything inside `format` raises
   "missing required argument: 'format'".

2. Even spelled correctly, SaveVideo's h264 re-encode is DEAD here:
       av.error.ExternalError: avcodec_open2("libx264", {})   <- fails with NO options
   PyAV's bundled libx264 cannot open, so format=auto (which preserves the source
   stream) is the only working mode and it cannot change quality.

   VHS_VideoCombine shells out to the ffmpeg executable instead. Its crf works:
   crf=12/19/40 -> 570,872 / 287,454 / 40,296 bytes on the same 48-frame input.
   Its pix_fmt/crf/save_metadata/trim_to_audio inputs are NOT declared in
   /object_info (they hang off the format combo), so they must be written here
   rather than surviving the UI->API widget mapping.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = os.path.join(HERE, "prompt_fixed.txt")
CRF = 12
NAMES = ["H3_Quality", "H3_Balanced", "H3_Fast"]


def vhs_inputs(prefix):
    return {
        "frame_rate": 24.0,
        "loop_count": 0,
        "filename_prefix": prefix,
        "format": "video/h264-mp4",
        "pingpong": False,
        "save_output": True,
        # format-specific inputs (not listed in object_info)
        "pix_fmt": "yuv420p",
        "crf": CRF,
        "save_metadata": True,
        "trim_to_audio": False,
    }


def main():
    prompt = open(PROMPT, encoding="utf-8").read().strip()
    for name in NAMES:
        path = os.path.join(HERE, "%s.api.json" % name)
        a = json.load(open(path, encoding="utf-8"))
        found = False
        for n in a.values():
            if n["class_type"] == "VHS_VideoCombine":
                found = True
                keep = {k: v for k, v in n["inputs"].items() if isinstance(v, list)}
                n["inputs"] = dict(vhs_inputs(name), **keep)
            if n["class_type"] == "PrimitiveStringMultiline":
                n["inputs"]["value"] = prompt
        json.dump(a, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("  %-13s %s" % (name, "patched" if found else "!! no VHS_VideoCombine node"))

    print()
    print("=== verify ===")
    for name in NAMES:
        a = json.load(open(os.path.join(HERE, "%s.api.json" % name), encoding="utf-8"))
        info = {}
        for n in a.values():
            ct = n["class_type"]
            if ct == "VHS_VideoCombine":
                info["crf"] = n["inputs"].get("crf")
                info["format"] = n["inputs"].get("format")
                info["pix_fmt"] = n["inputs"].get("pix_fmt")
                info["links"] = {k: v for k, v in n["inputs"].items() if isinstance(v, list)}
            elif ct == "ResolutionSelector":
                info["ratio"] = n["inputs"].get("aspect_ratio")
                info["mp"] = n["inputs"].get("megapixels")
            elif ct == "BasicScheduler":
                info["steps"] = n["inputs"].get("steps")
            elif ct == "LoraLoaderModelOnly":
                info["lora"] = n["inputs"].get("strength_model")
            elif ct == "PrimitiveStringMultiline":
                info["prompt_chars"] = len(str(n["inputs"].get("value", "")))
            elif ct == "TESpeedMiniMaxH3":
                info["TESPEED"] = "PRESENT"
            elif ct == "MiniMaxH3MemoryEfficientSageAttentionPatch":
                info["SAGE"] = "PRESENT"
        print("  %-13s %s" % (name, json.dumps(info, ensure_ascii=False)))


if __name__ == "__main__":
    main()

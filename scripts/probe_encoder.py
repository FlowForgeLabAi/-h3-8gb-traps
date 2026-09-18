"""Seconds-fast encoder probe: no model loading, just images -> save.

Two encoders are compared:
  A. SaveVideo    -- uses PyAV. Its h264 re-encode path is DEAD on this machine:
                     avcodec_open2("libx264", {}) fails even with no options.
  B. VHS_VideoCombine -- shells out to the ffmpeg executable (imageio-ffmpeg),
                     so it does not depend on PyAV's libx264.

Nested dynamic combos on SaveVideo are spelled with dotted keys
(comfy_api/latest/_io.py: finalize_prefix joins with "."):

    format / format.codec / format.codec.encoding / format.codec.encoding.crf
"""

import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "http://127.0.0.1:8188"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF = "微信图片_20260913134501_1457_5.jpg"


def graph_vhs(crf, frames=48):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": REF, "upload": "image"}},
        "2": {"class_type": "RepeatImageBatch",
              "inputs": {"image": ["1", 0], "amount": frames}},
        "3": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["2", 0], "frame_rate": 24.0, "loop_count": 0,
            "filename_prefix": "VHSTEST", "format": "video/h264-mp4",
            "pingpong": False, "save_output": True,
            # dynamic inputs revealed by the format selection
            "pix_fmt": "yuv420p", "crf": int(crf),
            "save_metadata": True, "trim_to_audio": False,
        }},
    }


def graph_savevideo(crf):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": REF, "upload": "image"}},
        "2": {"class_type": "RepeatImageBatch", "inputs": {"image": ["1", 0], "amount": 48}},
        "3": {"class_type": "CreateVideo", "inputs": {"images": ["2", 0], "fps": 24,
                                                      "bit_depth": 8, "color_space": "sRGB"}},
        "4": {"class_type": "SaveVideo", "inputs": {
            "video": ["3", 0], "filename_prefix": "SVTEST",
            "format": "mp4", "format.codec": "h264",
            "format.codec.encoding": "re-encode", "format.codec.encoding.crf": float(crf),
        }},
    }


def publish(prompt, label, timeout=120):
    try:
        body = json.dumps({"prompt": prompt}).encode()
        req = urllib.request.Request(API + "/prompt", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            pid = json.load(r)["prompt_id"]
    except urllib.error.HTTPError as e:
        d = e.read().decode("utf-8", "replace")
        print("  %-30s REJECTED: %s" % (label, d[:300]))
        return None
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(1)
        try:
            h = json.load(urllib.request.urlopen(API + "/history/%s" % pid, timeout=10))
        except Exception:
            continue
        if pid in h:
            item = h[pid]
            msgs = json.dumps(item.get("status", {}), ensure_ascii=False)
            ok = item.get("outputs")
            if "error" in msgs.lower():
                # pull the exception line out of the status messages
                for m in item.get("status", {}).get("messages", []):
                    s = str(m)
                    if "Exception" in s or "Error" in s:
                        print("  %-30s FAIL: %s" % (label, s[-260:]))
                        return None
                print("  %-30s FAIL" % label)
                return None
            return ok
    print("  %-30s TIMEOUT" % label)
    return None


def newest(prefix):
    fs = sorted(glob.glob(os.path.join(ROOT, "ComfyUI", "output", "**", "%s*" % prefix),
                          recursive=True), key=os.path.getmtime)
    return fs[-1] if fs else None


def main():
    print("cases (48 identical frames @24fps = 2 s):")
    cases = [
        ("VHS h264 crf=12", "VHSTEST", graph_vhs(12)),
        ("VHS h264 crf=19(default)", "VHSTEST", graph_vhs(19)),
        ("VHS h264 crf=40", "VHSTEST", graph_vhs(40)),
        ("SaveVideo re-encode crf=12", "SVTEST", graph_savevideo(12)),
    ]
    rows = []
    for label, prefix, g in cases:
        before = set(glob.glob(os.path.join(ROOT, "ComfyUI", "output", "**", "%s*" % prefix),
                               recursive=True))
        out = publish(g, label)
        if out is None:
            rows.append((label, None, None))
            continue
        after = set(glob.glob(os.path.join(ROOT, "ComfyUI", "output", "**", "%s*" % prefix),
                              recursive=True))
        new = sorted(after - before, key=os.path.getmtime)
        f = new[-1] if new else newest(prefix)
        rows.append((label, os.path.basename(f) if f else "?", os.path.getsize(f) if f else 0))

    print()
    print("=== 结果 ===")
    for label, name, size in rows:
        if size is None:
            print("  %-30s 失败" % label)
        else:
            print("  %-30s %-26s %9d 字节" % (label, name, size))
    print()
    print("判读：")
    print("  VHS 三个 crf 值若产生不同大小 => VHS 的 crf 可用（走外部 ffmpeg）")
    print("  SaveVideo re-encode 若失败 => 证实 PyAV 的 libx264 不可用")


if __name__ == "__main__":
    main()

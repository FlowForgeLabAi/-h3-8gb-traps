"""Benchmark harness for MiniMax H3 profiles.

Records, for every run:
  hardware, workflow, resolution, duration, frames, steps, sampler, scheduler,
  optimization nodes, startup flags, peak VRAM, peak system RAM, peak pagefile,
  phase timings (load / denoise / post), wall time, crash status, output file.

Designed to be started as a long-lived background job: it restarts ComfyUI itself
before each run, because a fresh session is required for valid timings (measured
on this machine: 3-6 s/block fresh vs 212 s/block after ~1.7 h of session uptime).

Usage:
    python benchmark_loop.py --spec bench/spec_example.json
    python benchmark_loop.py --spec bench/spec_example.json --dry-run
"""

import argparse
import ctypes
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
API = "http://127.0.0.1:8188"
LOG = os.path.join(ROOT, "ComfyUI", "user", "comfyui.log")
RESULTS = os.path.join(HERE, "results.csv")
LAUNCHER = os.path.join(ROOT, "run_nvidia_gpu.bat")

MEM_FIELDS = [
    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = MEM_FIELDS


def mem():
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {
        "ram_total_gb": round(m.ullTotalPhys / 1024 ** 3, 2),
        "ram_avail_gb": round(m.ullAvailPhys / 1024 ** 3, 2),
        "pagefile_used_gb": round((m.ullTotalPageFile - m.ullAvailPageFile) / 1024 ** 3, 2),
        "ram_load_pct": m.dwMemoryLoad,
    }


def gpu():
    """nvidia-smi one-shot: util, sm clock, power, temp, vram used."""
    smi = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
    try:
        out = subprocess.run(
            [smi, "--query-gpu=utilization.gpu,clocks.current.sm,power.draw,temperature.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        p = [x.strip() for x in out.split(",")]
        return {"gpu_util": float(p[0]), "sm_mhz": float(p[1]), "power_w": float(p[2]),
                "temp_c": float(p[3]), "vram_used_mib": float(p[4])}
    except Exception:
        return {}


def api(path, data=None, timeout=30):
    url = API + path
    if data is None:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.load(r)
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def comfy_pids(verbose=False):
    """Find real ComfyUI server processes.

    Matching strategy, chosen from what is actually readable in this environment:
      - Path works, StartTime works, but CommandLine is access-denied under some
        shells, and Get-NetTCPConnection can be blocked. So do NOT rely on those.
      - This harness itself runs from python_embeded\\python.exe inside the ComfyUI
        tree, so a plain path match would make it kill itself.
    => match python/pythonw whose Path is under the portable root, then subtract the
       harness's own PID and its whole ancestor chain (pwsh/cmd/job host).
    """
    me = os.getpid()
    script = (
        "$me = %d; $keep = New-Object System.Collections.Generic.HashSet[int]; "
        "[void]$keep.Add($me); "
        "$p = Get-CimInstance Win32_Process -Filter (\"ProcessId=$me\") -ErrorAction SilentlyContinue; "
        "for ($i=0; $i -lt 8 -and $p; $i++) { "
        "  [void]$keep.Add([int]$p.ProcessId); "
        "  $p = Get-CimInstance Win32_Process -Filter (\"ProcessId=\" + $p.ParentProcessId) -ErrorAction SilentlyContinue } "
        "Get-Process -Name python,pythonw -ErrorAction SilentlyContinue | "
        "Where-Object { $keep -notcontains $_.Id -and $_.Path -like '*ComfyUI_windows_portable*' } | "
        "ForEach-Object { \"$($_.Id)`t$([math]::Round($_.WorkingSet64/1MB))`t$($_.StartTime.ToString('HH:mm:ss'))\" }"
        % me
    )
    pids = set()
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                             capture_output=True, text=True, timeout=60).stdout
        for line in out.splitlines():
            parts = line.split("\t")
            if parts and parts[0].strip().isdigit():
                pids.add(int(parts[0].strip()))
                if verbose and len(parts) >= 3:
                    print("      candidate pid %s  %s MB  started %s"
                          % (parts[0].strip(), parts[1].strip(), parts[2].strip()))
    except Exception as e:
        print("      comfy_pids failed:", type(e).__name__)
    return pids


def port_free(host="127.0.0.1", port=8188, timeout=2.0):
    import socket
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return False
    except Exception:
        return True
    finally:
        s.close()


def stop_pids(pids):
    """Kill processes via PowerShell Stop-Process.

    `taskkill /F /PID` silently fails in this environment (verified: a process kept
    answering on port 8188 after being "killed"), while Stop-Process works.
    """
    if not pids:
        return
    ids = ",".join(str(p) for p in sorted(pids))
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Stop-Process -Id %s -Force -ErrorAction SilentlyContinue" % ids],
                   capture_output=True, timeout=60)


def kill_comfy(settle=30, want_ram_gb=5.0):
    """Kill ComfyUI and wait until RAM actually recovers.

    Killing the process is not enough on its own: the pagefile keeps ~25 GB
    committed for a while, and starting the next instance too early produces a
    contaminated baseline (observed 25.5 GB vs the clean 18.8 GB). So keep killing
    whatever still matches until the port is free and RAM has recovered.
    """
    all_pids = set()
    t0 = time.time()
    while time.time() - t0 < settle:
        pids = comfy_pids(verbose=True)
        if pids:
            all_pids |= pids
            print("    killing:", sorted(pids))
            stop_pids(pids)
            time.sleep(5)
            continue
        if port_free() and mem()["ram_avail_gb"] >= want_ram_gb:
            break
        time.sleep(3)
    if not port_free():
        print("    !! port 8188 still answering after cleanup")
    return all_pids


def _kill_comfy_old(settle=25, want_ram_gb=5.0):
    pids = comfy_pids(verbose=True)
    for p in pids:
        subprocess.run(["taskkill", "/F", "/PID", str(p)], capture_output=True)
    if not pids:
        return []
    t0 = time.time()
    while time.time() - t0 < settle:
        time.sleep(2)
        if comfy_pids():
            continue
        m = mem()
        if m["ram_avail_gb"] >= want_ram_gb:
            break
    # one extra beat so the OS finishes tearing down the address space
    time.sleep(3)
    return pids


def start_comfy(wait=300):
    # never benchmark a leftover instance: if the port still answers, the "fresh"
    # baseline would be contaminated (observed pagefile 25 GB instead of 19 GB)
    t = time.time()
    while not port_free() and time.time() - t < 90:
        pids = comfy_pids(verbose=True)
        if pids:
            print("    start_comfy cleanup, killing:", sorted(pids))
            stop_pids(pids)
        time.sleep(4)
    if not port_free():
        print("    !! port 8188 still answering after cleanup -- a stale instance may win the race")
        return None

    subprocess.Popen(["cmd", "/c", "start", "", LAUNCHER], cwd=ROOT,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < wait:
        time.sleep(5)
        try:
            api("/system_stats", timeout=5)
            return time.time() - t0
        except Exception:
            continue
    return None


def log_size():
    try:
        return os.path.getsize(LOG)
    except OSError:
        return 0


TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]\s*(.*)$")


def parse_run(mark):
    """Parse the log from byte offset `mark` and return phase timings."""
    try:
        with open(LOG, "rb") as f:
            f.seek(mark)
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return {}
    ev = []
    for line in text.splitlines():
        m = TS.match(line.strip())
        if m:
            ev.append((datetime.datetime.strptime(m.group(1)[:19], "%Y-%m-%d %H:%M:%S"), m.group(2)))
    if not ev:
        return {}
    blk = [(t, m) for t, m in ev if re.search(r"blocks\.\d+\.adaln", m)]
    t0 = ev[0][0]
    t_model = next((t for t, m in ev if "Model MiniMaxH3 prepared" in m), None)
    t_end = next((t for t, m in reversed(ev) if "Prompt executed in" in m), None)

    # The per-block markers above come from LoRA application errors, which only
    # appear when a LoRA is attached. Without a LoRA there are none, so fall back
    # to the denoise bounds: model ready -> the VAE being pulled back in to decode.
    t_den_start = t_model
    t_den_end = None
    if t_model:
        for t, m in ev:
            if t > t_model and ("Requested to load MiniMaxH3VideoVAE" in m
                                or "Requested to load MiniMaxH3AudioVAE" in m):
                t_den_end = t
                break
    if blk:
        t_den_start = blk[0][0] if t_den_start is None else t_den_start
        t_den_end = blk[-1][0]
    out = {
        "blocks_logged": len(blk),
        "sec_total": (t_end - t0).total_seconds() if t_end else None,
        "sec_load": ((t_model or (blk[0][0] if blk else None)) - t0).total_seconds()
                    if (t_model or blk) else None,
        "sec_denoise": (t_den_end - t_den_start).total_seconds()
                       if (t_den_start and t_den_end) else None,
        "sec_post": (t_end - t_den_end).total_seconds() if (t_end and t_den_end) else None,
        "completed": bool(t_end),
        "crashed": ("Traceback" in text) or ("OutOfMemoryError" in text) or ("CUDA error" in text),
        "tespeed": (re.search(r"acceleration ([\d.]+)%", text).group(1) + "%")
                   if re.search(r"acceleration ([\d.]+)%", text) else None,
        "seconds_per_block": None,
    }
    nblk = out["blocks_logged"] if out["blocks_logged"] > 1 else None
    if out["sec_denoise"] and nblk:
        out["seconds_per_block"] = round(out["sec_denoise"] / (nblk - 1), 3)
    return out


def apply_overrides(api_prompt, overrides):
    for cls, field, value in overrides:
        n = 0
        for node in api_prompt.values():
            if node["class_type"] == cls:
                node["inputs"][field] = value
                n += 1
        if n == 0:
            print("      WARN override hit 0 nodes: %s.%s" % (cls, field))
    return api_prompt


def describe(api_prompt):
    import math
    d = {}
    counts = {}
    for node in api_prompt.values():
        ct = node["class_type"]
        counts[ct] = counts.get(ct, 0) + 1
        inp = node["inputs"]
        if ct == "BasicScheduler":
            d["steps"] = inp.get("steps"); d["scheduler"] = inp.get("scheduler")
        elif ct == "KSamplerSelect":
            d["sampler"] = inp.get("sampler_name")
        elif ct == "LoraLoaderModelOnly":
            d["lora"] = str(inp.get("lora_name")); d["lora_strength"] = inp.get("strength_model")
        elif ct == "ModelAttentionBackend":
            d["attn_backend"] = inp.get("attention")
        elif ct == "MiniMaxH3SigmaShift":
            d["shift"] = "%s/%s" % (inp.get("shift_video"), inp.get("shift_audio"))
        elif ct == "MiniMaxLowVRAMAttention":
            d["head_chunks"] = inp.get("head_chunks")
        elif ct == "MiniMaxChunkFeedForward":
            d["chunkff"] = "%s/%s" % (inp.get("chunks"), inp.get("seq_threshold"))
        elif ct == "CLIPLoader":
            d["text_encoder"] = str(inp.get("clip_name"))
        elif ct == "UNETLoader":
            d["model"] = str(inp.get("unet_name"))
        elif ct == "VAELoader":
            d.setdefault("vaes", []).append(str(inp.get("vae_name")))
        elif ct == "RandomNoise":
            d["seed"] = inp.get("noise_seed")
    d["opt_nodes"] = "|".join(sorted(k for k in counts if k in (
        "MiniMaxLowVRAMAttention", "MiniMaxChunkFeedForward", "ModelAttentionBackend",
        "MiniMaxH3MemoryEfficientSageAttentionPatch", "TESpeedMiniMaxH3",
        "ModelPatchTorchSettings", "EasyCache")))
    return d


def run_one(spec, idx, total):
    print()
    print("=" * 78)
    print("[%d/%d] %s" % (idx, total, spec.get("label", "run")))
    print("=" * 78)

    api_path = spec["api"]
    if not os.path.isabs(api_path):
        api_path = os.path.join(ROOT, api_path)
    prompt = json.load(open(api_path, encoding="utf-8"))
    prompt = apply_overrides(prompt, [tuple(x) for x in spec.get("overrides", [])])
    desc = describe(prompt)
    print("  config:", json.dumps(desc, ensure_ascii=False)[:300])

    if not spec.get("skip_restart"):
        print("  restarting ComfyUI (fresh session is required for valid timings)...")
        killed = kill_comfy()
        print("    killed pids:", sorted(killed) if killed else "none")
        took = start_comfy()
        if took is None:
            print("    !! ComfyUI did not come up in 300 s -- skipping this run")
            return None
        print("    ready after %.0f s" % took)

    base = mem()
    clean = base["ram_avail_gb"] >= 4.0 and base["pagefile_used_gb"] <= 22.0
    print("  baseline: %s   %s" % (json.dumps(base, ensure_ascii=False),
                                   "CLEAN" if clean else "!! NOT CLEAN (pagefile still high)"))
    if spec.get("require_clean") and not clean:
        print("  !! baseline not clean and require_clean=true -- skipping this run")
        return None

    mark = log_size()
    t0 = time.time()
    try:
        res = api("/prompt", {"prompt": prompt}, timeout=90)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print("  !! submission rejected (HTTP %s): %s" % (e.code, body[:600]))
        return {"label": spec.get("label"), "rejected": True, "error": body[:600]}
    pid = res.get("prompt_id")
    print("  submitted:", pid)

    peak = {"vram_used_mib": 0, "ram_used_gb": 0, "pagefile_used_gb": 0,
            "power_w": 0, "temp_c": 0, "gpu_util": 0}
    crashed = False
    while True:
        time.sleep(float(spec.get("poll", 8)))
        g = gpu()
        if g:
            peak["vram_used_mib"] = max(peak["vram_used_mib"], g.get("vram_used_mib", 0))
            peak["power_w"] = max(peak["power_w"], g.get("power_w", 0))
            peak["temp_c"] = max(peak["temp_c"], g.get("temp_c", 0))
            peak["gpu_util"] = max(peak["gpu_util"], g.get("gpu_util", 0))
        m = mem()
        peak["ram_used_gb"] = max(peak["ram_used_gb"], base["ram_total_gb"] - m["ram_avail_gb"])
        peak["pagefile_used_gb"] = max(peak["pagefile_used_gb"], m["pagefile_used_gb"])
        try:
            h = api("/history/%s" % pid, timeout=20)
            if pid in h:
                break
        except Exception:
            pass
        if time.time() - t0 > float(spec.get("timeout", 7200)):
            print("  !! timeout")
            break
        if not comfy_pids():
            crashed = True
            print("  !! ComfyUI process disappeared (crash)")
            break

    wall = time.time() - t0
    phases = parse_run(mark)
    row = {"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "label": spec.get("label"),
           "prompt_id": pid, "wall_sec": round(wall, 1),
           "peak_vram_mib": round(peak["vram_used_mib"]),
           "peak_ram_used_gb": round(peak["ram_used_gb"], 2),
           "peak_pagefile_gb": round(peak["pagefile_used_gb"], 2),
           "peak_power_w": round(peak["power_w"], 1), "peak_temp_c": round(peak["temp_c"]),
           "process_crash": crashed}
    row.update({k: desc.get(k) for k in
                ("steps", "sampler", "scheduler", "lora", "lora_strength", "attn_backend",
                 "shift", "head_chunks", "chunkff", "model", "text_encoder", "seed", "opt_nodes")})
    row.update({k: phases.get(k) for k in
                ("sec_load", "sec_denoise", "sec_post", "sec_total", "blocks_logged",
                 "seconds_per_block", "completed", "crashed", "tespeed")})
    row["quality_note"] = spec.get("quality_note", "")

    print("  wall %.1f s (%.1f min) | peak VRAM %d MiB | peak RAM %.2f GB | pagefile %.2f GB"
          % (wall, wall / 60, row["peak_vram_mib"], row["peak_ram_used_gb"], row["peak_pagefile_gb"]))
    if phases:
        print("  phases: load %s | denoise %s | post %s | %s s/block"
              % (phases.get("sec_load"), phases.get("sec_denoise"), phases.get("sec_post"),
                 phases.get("seconds_per_block")))
    return row


FIELDS = ["date", "label", "prompt_id", "model", "text_encoder", "lora", "lora_strength",
          "attn_backend", "opt_nodes", "shift", "head_chunks", "chunkff", "steps", "sampler",
          "scheduler", "seed", "peak_vram_mib", "peak_ram_used_gb", "peak_pagefile_gb",
          "peak_power_w", "peak_temp_c", "wall_sec", "sec_load", "sec_denoise", "sec_post",
          "sec_total", "blocks_logged", "seconds_per_block", "tespeed", "completed",
          "crashed", "process_crash", "quality_note"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    spec_path = a.spec if os.path.isabs(a.spec) else os.path.join(ROOT, a.spec)
    spec = json.load(open(spec_path, encoding="utf-8"))
    runs = spec["runs"]
    print("spec: %s  (%d runs)" % (os.path.basename(spec_path), len(runs)))
    print("env : %s" % json.dumps({**mem(), **gpu()}, ensure_ascii=False))

    if a.dry_run:
        for i, s in enumerate(runs, 1):
            print("  %2d. %-38s %s" % (i, s.get("label"), s.get("api")))
        return

    new = not os.path.exists(RESULTS)
    rows = []
    for i, s in enumerate(runs, 1):
        try:
            r = run_one(s, i, len(runs))
        except Exception as e:
            print("  !! run raised %s: %s" % (type(e).__name__, e))
            r = {"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                 "label": s.get("label"), "quality_note": "EXCEPTION %s" % e}
        if r:
            rows.append(r)
        with open(RESULTS, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
                new = False
            if r:
                w.writerow(r)
        print("  -> results.csv updated (%d rows this session)" % len(rows))

    print()
    print("done. results:", RESULTS)


if __name__ == "__main__":
    main()

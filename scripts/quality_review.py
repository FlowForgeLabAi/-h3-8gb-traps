"""Build side-by-side quality comparison sheets from benchmark output videos.

Every benchmark run shares a fixed seed and prompt, so differences between their
videos come from the parameter under test. This lays the same RELATIVE timestamps
of each run next to each other, so a 5 s and a 10 s clip still line up.

Usage:
    python quality_review.py --pairs "4steps=auto_00006_.mp4,8steps=auto_00008_.mp4"
    python quality_review.py --pairs "..." --out bench/quality/foo.png
    python quality_review.py --tag As --list
"""

import argparse
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "quality")
VIDDIR = os.path.join(ROOT, "ComfyUI", "output", "video")
RESULTS = os.path.join(HERE, "results.csv")

# Sample each clip at the same RELATIVE positions so different durations line up.
# Fixed absolute stamps break on shorter clips: asking for t=9.8 on a 5 s video
# yields no frame at all, which then desynchronises the whole filter graph.
REL_POS = [0.04, 0.25, 0.50, 0.75, 0.96]
NCOL = len(REL_POS)
THUMB_W = 340


def ffmpeg():
    sys.path.insert(0, os.path.join(ROOT, "python_embeded", "Lib", "site-packages"))
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def read_results():
    import csv
    if not os.path.exists(RESULTS):
        return []
    with open(RESULTS, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def duration(ff, video):
    r = subprocess.run([ff, "-hide_banner", "-i", video], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
    if not m:
        return 0.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def videos_for(tag, rows):
    vids = sorted(glob.glob(os.path.join(VIDDIR, "*.mp4")), key=os.path.getmtime)
    if not tag:
        return [(os.path.basename(v), v) for v in vids[-NCOL:]]
    labels = [r["label"] for r in rows if r.get("label", "").upper().startswith(tag.upper())]
    if not labels:
        return []
    return list(zip(labels, vids[-len(labels):]))


def resolve(fname):
    path = fname if os.path.isabs(fname) else os.path.join(VIDDIR, fname)
    if os.path.exists(path):
        return path
    cand = glob.glob(os.path.join(ROOT, "ComfyUI", "output", "**", fname), recursive=True)
    return cand[0] if cand else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--pairs", default="", help="comma-separated label=filename")
    ap.add_argument("--out", default="")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    ff = ffmpeg()
    if a.pairs:
        pairs = []
        for item in a.pairs.split(","):
            item = item.strip()
            if not item:
                continue
            label, _, fname = item.partition("=")
            p = resolve(fname.strip())
            if os.path.exists(p):
                pairs.append((label.strip(), p))
            else:
                print("  missing:", fname)
    else:
        pairs = videos_for(a.tag, read_results())

    if not pairs:
        print("no videos matched (looked in %s)" % VIDDIR)
        return

    print("comparing %d videos (columns at %.0f%%..%.0f%% of each clip):"
          % (len(pairs), REL_POS[0] * 100, REL_POS[-1] * 100))
    for label, path in pairs:
        print("  %-26s %-44s d=%5.2fs  %5.1f MB"
              % (label, os.path.basename(path), duration(ff, path),
                 os.path.getsize(path) / 1024 / 1024))
    if a.list:
        return

    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    rows = []
    for label, path in pairs:
        d = duration(ff, path)
        row = []
        for i, rel in enumerate(REL_POS):
            t = max(0.05, d * rel)
            f = os.path.join(tmp, re.sub(r"[^A-Za-z0-9_.-]", "_", label) + "_%d.png" % i)
            subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-ss", "%.3f" % t,
                            "-i", path, "-frames:v", "1", "-vf", "scale=%d:-1" % THUMB_W,
                            "-y", f], capture_output=True)
            row.append(f if os.path.exists(f) else None)
        rows.append((label, row, d))

    complete = [(l, r, d) for l, r, d in rows if all(r)]
    skipped = [l for l, r, d in rows if not all(r)]
    if skipped:
        print("  skipped (frame extraction failed):", skipped)
    if not complete:
        print("no complete rows -- nothing to tile")
        return

    args = [ff, "-hide_banner", "-loglevel", "error"]
    for _, row, _ in complete:
        for f in row:
            args += ["-i", f]
    parts = []
    for vi in range(len(complete)):
        base = vi * NCOL
        parts.append("".join("[%d:v]" % (base + c) for c in range(NCOL)) +
                     "hstack=inputs=%d[r%d]" % (NCOL, vi))
    parts.append("".join("[r%d]" % i for i in range(len(complete))) +
                 "vstack=inputs=%d[out]" % len(complete))
    sheet = a.out or os.path.join(OUT, "compare_%s.png" % (a.tag or "recent"))
    args += ["-filter_complex", ";".join(parts), "-map", "[out]", "-y", sheet]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(sheet):
        print("sheet build failed:", r.stderr[:600])
        return

    print()
    print("rows (top to bottom):  " + " | ".join("%s(%.2fs)" % (l, d) for l, _, d in complete))
    print("cols (left to right):  " + "%, ".join("%.0f%%" % (x * 100) for x in REL_POS) + "%")
    print()
    print("sheet:", sheet)


if __name__ == "__main__":
    main()

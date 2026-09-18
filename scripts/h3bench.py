"""Benchmark harness for the MiniMax H3 workflow.

Converts a UI-format workflow (nodes/links) into the API prompt format so runs can
be submitted programmatically with parameter overrides, then records a full
measurement row for every run.

Usage:
    python h3bench.py build   <workflow.json> [out_api.json]
    python h3bench.py run     <api.json> --label NAME [--set Class.input=value ...]
    python h3bench.py list    <api.json>
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from datetime import datetime

API = "http://127.0.0.1:8188"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.csv")

# widget-capable input types are the ones that appear in widgets_values;
# anything that is a connection type comes from a link instead.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_workflow import CONNECT_TYPES, widget_order as widget_names, is_connect  # noqa: E402


def fetch_object_info():
    """Reuse build_workflow's cached loader so conversion also works offline."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_workflow import object_info
    return object_info()




def build_api(ui_path, out_path):
    wf = json.load(open(ui_path, encoding="utf-8"))
    oi = fetch_object_info()
    nodes = {n["id"]: n for n in wf.get("nodes", [])}
    links = {l[0]: l for l in wf.get("links", [])}

    # ---- resolve bypassed / muted nodes by walking back through the graph
    def source_of(origin_id, origin_slot, depth=0):
        if depth > 12:
            return (origin_id, origin_slot)
        n = nodes.get(origin_id)
        if not n or n.get("mode", 0) == 0:
            return (origin_id, origin_slot)
        # node is bypassed (4) or muted (2): forward the matching input
        outs = n.get("outputs") or []
        want = outs[origin_slot].get("type") if origin_slot < len(outs) else None
        for i, inp in enumerate(n.get("inputs") or []):
            if inp.get("link") is None:
                continue
            if want and inp.get("type") != want:
                continue
            lk = links.get(inp["link"])
            if lk:
                return source_of(lk[1], lk[2], depth + 1)
        return (origin_id, origin_slot)

    api = {}
    skipped = []
    for nid, n in nodes.items():
        ct = n.get("type")
        if ct in ("Note", "MarkdownNote", "Reroute", "PrimitiveNode"):
            skipped.append((nid, ct))
            continue
        if ct not in oi:
            skipped.append((nid, ct + "  [NOT INSTALLED]"))
            continue
        if n.get("mode", 0) == 2:          # never/muted -> do not execute
            skipped.append((nid, ct + "  [muted]"))
            continue

        inputs = {}

        # widget values, mapped by name
        wnames = widget_names(oi, ct)
        wvals = n.get("widgets_values") or []
        if isinstance(wvals, dict):        # newer frontends may store named widgets
            inputs.update(wvals)
        else:
            for i, wname in enumerate(wnames):
                if i < len(wvals):
                    inputs[wname] = wvals[i]

        # link-driven inputs override/append
        for inp in n.get("inputs") or []:
            link_id = inp.get("link")
            if link_id is None:
                continue
            lk = links.get(link_id)
            if not lk:
                continue
            src_id, src_slot = source_of(lk[1], lk[2])
            inputs[inp["name"]] = [str(src_id), src_slot]

        api[str(nid)] = {"class_type": ct, "inputs": inputs}

    json.dump(api, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return api, skipped


def submit(api):
    body = json.dumps({"prompt": api}).encode("utf-8")
    req = urllib.request.Request(API + "/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError("HTTP %s: %s" % (e.code, detail[:1500]))


def wait(prompt_id, poll=5, timeout=7200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(poll)
        with urllib.request.urlopen("%s/history/%s" % (API, prompt_id), timeout=20) as r:
            h = json.load(r)
        if prompt_id in h:
            return h[prompt_id], time.time() - t0
    raise TimeoutError("prompt %s did not finish in %ss" % (prompt_id, timeout))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "list", "apinfo"])
    ap.add_argument("path")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--label", default="")
    ap.add_argument("--set", action="append", default=[],
                    help="Class.input=value  (value parsed as JSON)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    if a.cmd == "build":
        out = a.out or (os.path.splitext(a.path)[0] + ".api.json")
        api, skipped = ui_to_api_checked(a.path, out)
        print("built %s  (%d nodes)" % (out, len(api)))
        print("skipped:")
        for nid, why in skipped:
            print("   id=%-5s %s" % (nid, why))
        return

    if a.cmd == "apinfo":
        api = json.load(open(a.path, encoding="utf-8"))
        print("%d nodes" % len(api))
        for nid in sorted(api, key=lambda x: int(x)):
            n = api[nid]
            print("  id=%-5s %-42s %s" % (nid, n["class_type"],
                  {k: v for k, v in n["inputs"].items() if not isinstance(v, list)}))
        return

    if a.cmd == "list":
        api = json.load(open(a.path, encoding="utf-8"))
        for nid in sorted(api, key=lambda x: int(x)):
            print("  %-5s %s" % (nid, api[nid]["class_type"]))
        return

    # ---- run
    api = json.load(open(a.path, encoding="utf-8"))
    for s in a.set:
        key, _, raw = s.partition("=")
        cls, _, field = key.partition(".")
        val = json.loads(raw)
        hit = 0
        for nid, n in api.items():
            if n["class_type"] == cls:
                n["inputs"][field] = val
                hit += 1
        print("  set %-28s = %-22s (%d node(s))" % (key, raw, hit))

    t0 = time.time()
    res = submit(api)
    pid = res.get("prompt_id")
    print("submitted: %s" % pid)
    hist, dt = wait(pid)
    status = (hist.get("status") or {}).get("status_str")
    print("finished: %s  in %.1f s (%.1f min)" % (status, dt, dt / 60))
    return {"prompt_id": pid, "status": status, "seconds": dt}


def ui_to_api_checked(ui_path, out_path):
    api, skipped = build_api(ui_path, out_path)
    missing = [s for s in skipped if "NOT INSTALLED" in s[1]]
    if missing:
        print("WARNING: %d node(s) not installed:" % len(missing))
        for nid, why in missing:
            print("   ", why)
    return api, skipped


if __name__ == "__main__":
    main()

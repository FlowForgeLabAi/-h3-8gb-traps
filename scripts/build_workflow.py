"""Generate a MiniMax H3 ComfyUI workflow (UI format) from a declarative node list.

Design goals:
  - build from scratch, not by editing an existing workflow file
  - derive widget order and slot indices from ComfyUI's own /object_info, so the
    output stays valid across node-pack updates
  - one config dict per profile -> one workflow JSON

Usage:
    python build_workflow.py profiles/quality.json out.json
    python build_workflow.py --list-profiles
"""

import json
import os
import sys
import time
import urllib.request

API = "http://127.0.0.1:8188"
HERE = os.path.dirname(os.path.abspath(__file__))

CONNECT_TYPES = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK", "AUDIO",
    "SAMPLER", "SIGMAS", "GUIDER", "NOISE", "CLIP_VISION", "CONTROL_NET",
    # VIDEO was missing here, which made SaveVideo.video look like a widget and
    # shifted filename_prefix out of its slot (outputs came out as auto_0000N_.mp4)
    "VIDEO",
    # multi-type / wildcard sockets used by ComfyMathExpression and pass-through nodes
    "FLOAT,INT,BOOLEAN", "*", "RESOLUTION_PREVIEW",
}


def is_connect(type_str):
    """True when an object_info type means a link socket rather than a widget.

    Handles the multi-type form used by nodes such as ComfyMathExpression
    ("FLOAT,INT,BOOLEAN") and the wildcard "*" used by pass-through nodes.
    """
    if not isinstance(type_str, str):
        return False
    if type_str in CONNECT_TYPES or type_str == "*":
        return True
    if "," in type_str:
        return all(x.strip() in CONNECT_TYPES or x.strip() == "*" for x in type_str.split(","))
    return False

# ---------------------------------------------------------------- node schema


def object_info(use_cache=True):
    """Fetch /object_info, caching to disk so generation also works offline.

    Generating a workflow needs the live node schemas. Caching lets the 4060
    package regenerate its workflows without a running server, and stops a
    transient server outage from breaking a regeneration.
    """
    cache = os.path.join(HERE, "object_info_cache.json")
    if use_cache and os.path.exists(cache):
        try:
            if time.time() - os.path.getmtime(cache) < 86400:
                with open(cache, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
    try:
        with urllib.request.urlopen(API + "/object_info", timeout=180) as r:
            oi = json.load(r)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(oi, f)
        return oi
    except Exception as e:
        if os.path.exists(cache):
            print("  /object_info unreachable (%s) -- using cached copy" % type(e).__name__)
            with open(cache, encoding="utf-8") as f:
                return json.load(f)
        raise


def widget_order(oi, class_type):
    """Names of widget inputs, in the order ComfyUI stores them in widgets_values."""
    info = oi.get(class_type)
    if not info:
        raise KeyError("node class not installed: %s" % class_type)
    names = []
    spec = info.get("input", {})
    for section in ("required", "optional"):
        for name, d in (spec.get(section) or {}).items():
            if not isinstance(d, list) or not d:
                continue
            t = d[0]
            if isinstance(t, list) or not is_connect(t):
                names.append(name)
    return names


def input_index(oi, class_type, name):
    """Index among ALL inputs, in object_info declaration order.

    ComfyUI's UI format lists connection inputs and widget inputs in one array, in
    object_info order, so this counts both kinds.
    """
    info = oi[class_type]
    spec = info.get("input", {})
    base = name.split(".")[0]          # Autogrow sub-inputs are stored as "group.entry_0"
    i = 0
    for section in ("required", "optional"):
        for nm in (spec.get(section) or {}):
            if nm == name or nm == base:
                return i
            i += 1
    raise KeyError("%s has no input %r" % (class_type, name))


def slot_of(oi, class_type, direction, name):
    """Index of an input or output slot by name (outputs tolerate case differences)."""
    if direction == "in":
        return input_index(oi, class_type, name)
    info = oi[class_type]
    outs = info.get("output", [])
    names = info.get("output_name", outs)
    if name in names:
        return names.index(name)
    low = [str(x).lower() for x in names]
    if str(name).lower() in low:
        return low.index(str(name).lower())
    raise KeyError("%s has no output %r (has %s)" % (class_type, name, names))


def slot_type(oi, class_type, direction, name):
    info = oi[class_type]
    if direction == "in":
        spec = info.get("input", {})
        base = name.split(".")[0]
        for section in ("required", "optional"):
            for nm, d in (spec.get(section) or {}).items():
                if nm in (name, base) and isinstance(d, list) and d:
                    return d[0] if isinstance(d[0], str) else "COMBO"
        raise KeyError("%s has no input %r" % (class_type, name))
    outs = info.get("output", [])
    names = info.get("output_name", outs)
    if name in names:
        return outs[names.index(name)]
    low = [str(x).lower() for x in names]
    if str(name).lower() in low:
        return outs[low.index(str(name).lower())]
    raise KeyError("%s has no output %r" % (class_type, name))


# ---------------------------------------------------------------- generation


def build(cfg, oi):
    """cfg: {"title", "nodes": [{"id","type","pos","widgets":{},"title"?}], "links":[[from,from_out,to,to_in]]}"""
    nodes = cfg["nodes"]
    links_cfg = cfg["links"]
    by_id = {n["id"]: n for n in nodes}

    out_links = {}      # (node_id, out_name) -> [link ids]
    in_links = {}       # (node_id, in_name)  -> link id
    link_rows = []

    for lid, (a, aout, b, binp) in enumerate(links_cfg, start=1):
        ta = by_id[a]["type"]
        tb = by_id[b]["type"]
        t_out = slot_type(oi, ta, "out", aout)
        t_in = slot_type(oi, tb, "in", binp)
        # A widget input can legally receive a link (ComfyUI's "convert widget to
        # input"). Only flag a mismatch when both sides are real connection types.
        if t_out != t_in and is_connect(t_out) and is_connect(t_in):
            raise ValueError("link %d type mismatch: %s.%s(%s) -> %s.%s(%s)"
                             % (lid, ta, aout, t_out, tb, binp, t_in))
        link_rows.append([lid, a, slot_of(oi, ta, "out", aout), b, slot_of(oi, tb, "in", binp), t_out])
        out_links.setdefault((a, aout), []).append(lid)
        if (b, binp) in in_links:
            raise ValueError("input %s.%s already connected" % (tb, binp))
        in_links[(b, binp)] = lid

    json_nodes = []
    for order, n in enumerate(nodes):
        t = n["type"]
        info = oi[t]
        wnames = widget_order(oi, t)
        wvals = []
        for wname in wnames:
            if wname in n.get("widgets", {}):
                wvals.append(n["widgets"][wname])
            else:
                d = None
                spec = info.get("input", {})
                for section in ("required", "optional"):
                    if wname in (spec.get(section) or {}):
                        d = spec[section][wname]
                        break
                wvals.append(d[1].get("default") if isinstance(d, list) and len(d) > 1 and isinstance(d[1], dict) else None)
        if "widgets_values" in n:
            wvals = n["widgets_values"]

        # ComfyUI's UI format lists connection inputs AND widget inputs together.
        # Widget inputs carry a "widget" marker; a connected widget keeps the marker
        # and gains a link (that is how "convert widget to input" is stored).
        inputs = []
        spec = info.get("input", {})
        for section in ("required", "optional"):
            for nm, d in (spec.get(section) or {}).items():
                if not isinstance(d, list) or not d:
                    continue
                ty = d[0]
                if is_connect(ty):
                    inputs.append({"name": nm, "type": ty, "link": in_links.get((n["id"], nm))})
                elif nm in wnames:
                    inputs.append({"name": nm, "type": ty if isinstance(ty, str) else "COMBO",
                                   "widget": {"name": nm}, "link": in_links.get((n["id"], nm))})
        # Autogrow groups: the frontend stores each entry as "group.entry_N" and emits it
        # as a plain connection input (no widget marker), so append those explicitly.
        emitted = set()
        for it in inputs:
            if it.get("link"):
                emitted.add(it["name"])
        for (nid, nm), lid in sorted(in_links.items(), key=lambda kv: kv[0][1]):
            if nid != n["id"] or nm in emitted:
                continue
            src_id = next((r[1] for r in link_rows if r[0] == lid), None)
            src_type = next((r[5] for r in link_rows if r[0] == lid), "IMAGE")
            inputs.append({"name": nm, "type": src_type, "link": lid, "shape": 7})
            emitted.add(nm)

        outputs = []
        onames = info.get("output_name", info.get("output", []))
        for i, otype in enumerate(info.get("output", [])):
            nm = onames[i] if i < len(onames) else otype
            outputs.append({
                "name": nm,
                "type": otype,
                "links": out_links.get((n["id"], nm)) or out_links.get((n["id"], otype)),
                "slot_index": i,
            })

        node = {
            "id": n["id"],
            "type": t,
            "pos": n.get("pos", [0, order * 120]),
            "size": n.get("size", [300, 80]),
            "flags": {},
            "order": order,
            "mode": n.get("mode", 0),
            "inputs": inputs,
            "outputs": outputs,
            "properties": {"Node name for S&R": t},
            "widgets_values": wvals,
        }
        if n.get("title"):
            node["title"] = n["title"]
        json_nodes.append(node)

    return {
        "last_node_id": max(n["id"] for n in nodes),
        "last_link_id": len(link_rows),
        "nodes": json_nodes,
        "links": link_rows,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    if sys.argv[1] == "--list-profiles":
        d = os.path.join(HERE, "profiles")
        for f in sorted(os.listdir(d)):
            print("  ", f)
        return
    cfg = json.load(open(sys.argv[1], encoding="utf-8"))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(sys.argv[1])[0] + ".json"
    oi = object_info()
    wf = build(cfg, oi)
    json.dump(wf, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote %s (%d nodes, %d links)" % (out, len(wf["nodes"]), len(wf["links"])))


if __name__ == "__main__":
    main()

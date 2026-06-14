"""Subprocess harness: introspect generated CadQuery geometry WITHOUT exporting.

Invoked as `python introspect.py <code_file> <query_file>`. Like ``harness.py`` it
is deliberately self-contained (stdlib + cadquery only) — it never imports the
cad_gen package, so it can run in any interpreter that has cadquery installed.

It exists so the generator can *probe* its own model (which edges does a selector
match? what faces/edges did I actually build?) before committing a fragile
fillet/chamfer/shell — instead of guessing and crashing the real script. It does
NOT export STL/STEP and measures nothing, so it is much cheaper than a full run.

Contract:
- user code runs exactly as in harness.py (`cq`/`cadquery` pre-imported, show_object
  shim); the model is taken from `result`, else show_object() calls, else any
  Workplane/Shape left in the namespace, and is wrapped into a Workplane so string
  selectors can be evaluated against it.
- the query (JSON in <query_file>) is one of:
    {"mode": "describe"}
    {"mode": "selector", "target": "edges"|"faces", "selector": "<sel>"}
- on success: writes the JSON result to stdout, exit 0.
- if the USER CODE fails: traceback on stderr, exit 1 (the model must fix the code).
  A bad *selector* is NOT a code failure — it returns a normal result with
  count 0 and a `selector_error` string, because that is exactly the diagnostic the
  model asked for.
"""

import json
import math
import sys
import traceback
from pathlib import Path

_MAX_PER_GROUP = 12  # cap representative entities per geomType group to bound output


def _collect_shapes(namespace, shown):
    import cadquery as cq

    if isinstance(namespace.get("result"), (cq.Workplane, cq.Shape)):
        candidates = [namespace["result"]]
    elif shown:
        candidates = list(shown)
    else:
        candidates = [
            v
            for k, v in namespace.items()
            if not k.startswith("_") and isinstance(v, (cq.Workplane, cq.Shape))
        ]

    shapes = []
    for obj in candidates:
        if isinstance(obj, cq.Workplane):
            shapes.extend(s for s in obj.vals() if isinstance(s, cq.Shape))
        elif isinstance(obj, cq.Shape):
            shapes.append(obj)
    return shapes


def _load_workplane(namespace, shown):
    """Return a Workplane wrapping the produced geometry, so selectors can run on it."""
    import cadquery as cq

    shapes = _collect_shapes(namespace, shown)
    if not shapes:
        raise RuntimeError(
            "No geometry produced: assign the final solid to a variable named "
            "`result` (a cadquery Workplane or Shape)."
        )
    shape = shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)
    return cq.Workplane().add(shape)


def _round3(x):
    return round(float(x), 3)


def _xyz(p):
    return [_round3(p.x), _round3(p.y), _round3(p.z)]


def _edge_info(e):
    """Compact, selector-relevant description of one edge."""
    info = {"type": e.geomType(), "length": _round3(e.Length()), "center": _xyz(e.Center())}
    try:
        if e.geomType() == "LINE":
            sp, ep = e.startPoint(), e.endPoint()
            d = (ep.x - sp.x, ep.y - sp.y, ep.z - sp.z)
            n = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) or 1.0
            info["dir"] = [_round3(d[0] / n), _round3(d[1] / n), _round3(d[2] / n)]
        elif e.geomType() in ("CIRCLE", "ARC"):
            info["radius"] = _round3(e.radius())
    except Exception:  # noqa: BLE001 — geometry queries are best-effort enrichment
        pass
    return info


def _face_info(f):
    """Compact description of one face."""
    info = {"type": f.geomType(), "area": _round3(f.Area()), "center": _xyz(f.Center())}
    try:
        info["normal"] = _xyz(f.normalAt())
    except Exception:  # noqa: BLE001 — non-planar faces may not have a single normal
        pass
    return info


def _grouped(entities, info_fn):
    """Group entities by geomType: full counts + a capped sample of each group."""
    from collections import OrderedDict

    groups = OrderedDict()
    for ent in entities:
        groups.setdefault(ent.geomType(), []).append(ent)
    out = []
    for gtype, ents in groups.items():
        out.append(
            {
                "type": gtype,
                "count": len(ents),
                "sample": [info_fn(e) for e in ents[:_MAX_PER_GROUP]],
                "truncated": len(ents) > _MAX_PER_GROUP,
            }
        )
    return out


def _describe(wp):
    edges = wp.edges().vals()
    faces = wp.faces().vals()
    solids = wp.solids().vals()
    bb = wp.val().BoundingBox()
    return {
        "bbox_mm": [_round3(bb.xlen), _round3(bb.ylen), _round3(bb.zlen)],
        "n_solids": len(solids),
        "n_faces": len(faces),
        "n_edges": len(edges),
        "faces": _grouped(faces, _face_info),
        "edges": _grouped(edges, _edge_info),
    }


def _run_selector(wp, target, selector):
    if target not in ("edges", "faces"):
        return {"selector": selector, "target": target, "count": 0,
                "selector_error": f"target must be 'edges' or 'faces', got {target!r}"}
    info_fn = _edge_info if target == "edges" else _face_info
    try:
        matched = getattr(wp, target)(selector).vals()
    except Exception as exc:  # noqa: BLE001 — a bad selector is a result, not a crash
        return {
            "selector": selector,
            "target": target,
            "count": 0,
            "selector_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "selector": selector,
        "target": target,
        "count": len(matched),
        "matches": [info_fn(e) for e in matched[:_MAX_PER_GROUP]],
        "truncated": len(matched) > _MAX_PER_GROUP,
        "selector_error": None,
    }


def main() -> None:
    code_file, query_file = Path(sys.argv[1]), Path(sys.argv[2])
    code = code_file.read_text()
    query = json.loads(query_file.read_text())

    import cadquery as cq

    shown = []
    namespace = {
        "__name__": "__cadgen_model__",
        "cq": cq,
        "cadquery": cq,
        "show_object": lambda obj, *args, **kwargs: shown.append(obj),
    }
    exec(compile(code, str(code_file), "exec"), namespace)

    wp = _load_workplane(namespace, shown)

    mode = query.get("mode", "describe")
    if mode == "selector":
        result = _run_selector(wp, query.get("target", "edges"), query.get("selector", ""))
    else:
        result = _describe(wp)
    result["mode"] = mode

    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

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
    {"mode": "query", "target": "edges"|"faces", "where": {...}, "limit": <n>?}
  "query" is the imported-B-rep feature finder: `where` accepts type, area_min/max,
  length_min/max, radius_min/max, normal (unit vector) and center_box
  ([xmin,ymin,zmin,xmax,ymax,zmax]); results come back largest-first with the full match
  count and the model's absolute bounds. "describe" is unchanged and still caps each
  geomType group at 12 entities in traversal order — fine for a part you just built, useless
  on a 1000-face import, which is exactly the gap "query" exists to close.
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


def _face_info_rich(f):
    """`_face_info` plus the radius and axis of a cylindrical / conical face.

    Kept separate so `describe` output stays exactly what it has always been — this only
    feeds `query`, which is opt-in. CadQuery exposes radius on edges but not on faces, and
    face radius is precisely what an edit instruction names ("the largest-diameter bore",
    "the two 5 mm holes"): a filter on it finds a bore that no combination of area and centre
    could isolate.
    """
    info = _face_info(f)
    try:
        kind = f.geomType()
        if kind in ("CYLINDER", "CONE"):
            from OCP.BRepAdaptor import BRepAdaptor_Surface

            adaptor = BRepAdaptor_Surface(f.wrapped)
            surface = adaptor.Cylinder() if kind == "CYLINDER" else adaptor.Cone()
            axis = surface.Axis()
            direction, location = axis.Direction(), axis.Location()
            info["radius"] = _round3(
                surface.Radius() if kind == "CYLINDER" else surface.RefRadius()
            )
            info["axis"] = [_round3(direction.X()), _round3(direction.Y()), _round3(direction.Z())]
            info["axis_point"] = [
                _round3(location.X()), _round3(location.Y()), _round3(location.Z())
            ]
    except Exception:  # noqa: BLE001 — best-effort enrichment, never a probe failure
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


def _matches(info, where):
    """Does one entity's info dict satisfy the `where` filter?

    Unknown filter keys are ignored rather than rejected: a filter is a narrowing aid, and
    failing the whole query over a typo would cost the caller an entire probe round-trip for
    no diagnostic gain (`count` already tells them if they narrowed to nothing).
    """
    if not where:
        return True
    wanted_type = where.get("type")
    if wanted_type and info.get("type") != wanted_type:
        return False
    for key, field in (("area_min", "area"), ("length_min", "length"), ("radius_min", "radius")):
        if key in where and (info.get(field) is None or info[field] < where[key]):
            return False
    for key, field in (("area_max", "area"), ("length_max", "length"), ("radius_max", "radius")):
        if key in where and (info.get(field) is None or info[field] > where[key]):
            return False
    normal = where.get("normal")
    if normal is not None:
        actual = info.get("normal")
        if actual is None:
            return False
        # Dot product against a unit direction: 0.99 keeps faces within ~8 degrees, loose
        # enough to survive a slightly-off authored normal, tight enough to separate axes.
        if sum(a * b for a, b in zip(actual, normal)) < 0.99:
            return False
    box = where.get("center_box")
    if box is not None:
        center = info.get("center")
        if center is None:
            return False
        if any(not (box[i] <= center[i] <= box[i + 3]) for i in range(3)):
            return False
    return True


def _query(wp, target, where, limit):
    """Filtered, ranked listing of faces or edges — the imported-B-rep feature finder.

    Returns the full match count alongside the (capped) list, so a caller can tell "my filter
    matched 4 faces" from "my filter matched 400 and you are seeing 20 of them".
    """
    if target not in ("edges", "faces"):
        return {"target": target, "count": 0,
                "query_error": f"target must be 'edges' or 'faces', got {target!r}"}
    info_fn = _edge_info if target == "edges" else _face_info_rich
    infos = [info_fn(e) for e in getattr(wp, target)().vals()]
    matched = [info for info in infos if _matches(info, where)]
    # Largest-first, so a truncated result shows the features an instruction is likely to be
    # naming. Traversal order is an artifact of how the B-rep was authored and, on an
    # imported model, effectively arbitrary.
    matched.sort(key=lambda info: info.get("area", info.get("length", 0.0)), reverse=True)
    bb = wp.val().BoundingBox()
    return {
        "target": target,
        "where": where,
        "count": len(matched),
        "total": len(infos),
        "matches": matched[:limit],
        "truncated": len(matched) > limit,
        # Absolute bounds, unlike describe's extents: positioning a cutting primitive by
        # coordinate needs to know where the part actually sits, not just how big it is.
        "model_bbox_min_mm": [_round3(bb.xmin), _round3(bb.ymin), _round3(bb.zmin)],
        "model_bbox_max_mm": [_round3(bb.xmax), _round3(bb.ymax), _round3(bb.zmax)],
        "query_error": None,
    }


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
    elif mode == "query":
        result = _query(
            wp,
            query.get("target", "faces"),
            query.get("where") or {},
            int(query.get("limit") or _MAX_PER_GROUP),
        )
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

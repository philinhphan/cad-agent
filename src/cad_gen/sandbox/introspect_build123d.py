"""Subprocess harness: introspect generated build123d geometry WITHOUT exporting.

The build123d counterpart of ``introspect.py``. Invoked as
``python introspect_build123d.py <code_file> <query_file>``. Like its sibling it is
deliberately self-contained (stdlib + build123d only) — it never imports the cad_gen
package, so it can run in any interpreter that has build123d installed.

It exists so the generator can *probe* its own model (which edges does this ShapeList
expression match? what faces/edges did I actually build?) before committing a fragile
fillet/chamfer/offset — instead of guessing and crashing the real script. It does NOT
export STL/STEP and measures nothing, so it is much cheaper than a full run.

Contract:
- user code runs exactly as in harness_build123d.py (build123d API pre-imported,
  show_object shim); the model is taken from `result`, else show_object() calls, else any
  Shape/Builder left in the namespace.
- the query (JSON in <query_file>) is one of:
    {"mode": "describe"}
    {"mode": "selection", "expression": "<python expression>"}
- on success: writes the JSON result to stdout, exit 0.
- if the USER CODE fails: traceback on stderr, exit 1 (the model must fix the code).
  A bad *expression* is NOT a code failure — it returns a normal result with count 0 and a
  `selection_error` string, because that is exactly the diagnostic the model asked for.
  (This mirrors introspect.py, where a bad CadQuery selector is likewise a result.)
"""

import json
import sys
import traceback
from pathlib import Path

from harness_build123d import _collect_shapes, build123d_namespace

_MAX_PER_GROUP = 12  # cap representative entities per geom_type group to bound output


def _load_shape(namespace, shown):
    """Return the single Shape the script produced, so expressions can run against it."""
    from build123d import Compound

    shapes = _collect_shapes(namespace, shown)
    if not shapes:
        raise RuntimeError(
            "No geometry produced: assign the final solid to a variable named "
            "`result` (a build123d Shape, or the BuildPart builder that made it)."
        )
    return shapes[0] if len(shapes) == 1 else Compound(children=shapes)


def _round3(x):
    return round(float(x), 3)


def _xyz(p):
    return [_round3(p.X), _round3(p.Y), _round3(p.Z)]


def _geom_type(entity):
    """build123d exposes geom_type as a GeomType enum; report its name, as CadQuery does."""
    gt = entity.geom_type
    return getattr(gt, "name", str(gt))


def _edge_info(e):
    """Compact, selection-relevant description of one edge."""
    info = {"type": _geom_type(e), "length": _round3(e.length), "center": _xyz(e.center())}
    try:
        if _geom_type(e) == "LINE":
            sp, ep = e.start_point(), e.end_point()
            d = (ep.X - sp.X, ep.Y - sp.Y, ep.Z - sp.Z)
            n = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5 or 1.0
            info["dir"] = [_round3(d[0] / n), _round3(d[1] / n), _round3(d[2] / n)]
        elif _geom_type(e) in ("CIRCLE", "ELLIPSE"):
            info["radius"] = _round3(e.radius)
    except Exception:  # noqa: BLE001 — geometry queries are best-effort enrichment
        pass
    return info


def _face_info(f):
    """Compact description of one face."""
    info = {"type": _geom_type(f), "area": _round3(f.area), "center": _xyz(f.center())}
    try:
        info["normal"] = _xyz(f.normal_at())
    except Exception:  # noqa: BLE001 — non-planar faces may not have a single normal
        pass
    return info


def _entity_info(entity):
    """Describe an entity of unknown kind (a selection may mix faces and edges)."""
    from build123d import Face

    return _face_info(entity) if isinstance(entity, Face) else _edge_info(entity)


def _grouped(entities, info_fn):
    """Group entities by geom_type: full counts + a capped sample of each group."""
    from collections import OrderedDict

    groups = OrderedDict()
    for ent in entities:
        groups.setdefault(_geom_type(ent), []).append(ent)
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


def _describe(shape):
    edges = shape.edges()
    faces = shape.faces()
    solids = shape.solids()
    bb = shape.bounding_box()
    return {
        "bbox_mm": [_round3(bb.size.X), _round3(bb.size.Y), _round3(bb.size.Z)],
        "n_solids": len(solids),
        "n_faces": len(faces),
        "n_edges": len(edges),
        "faces": _grouped(faces, _face_info),
        "edges": _grouped(edges, _edge_info),
    }


def _run_selection(shape, namespace, expression):
    """Evaluate a ShapeList expression against the built model.

    The expression is evaluated in the script's own namespace with `result` rebound to the
    resolved Shape, so `result.edges().filter_by(Axis.Z)` works whether the script assigned
    a builder or a Shape. Evaluating model-authored code here is no wider a door than the
    `exec` above: this is a crash-isolation boundary, not a security boundary.
    """
    if not expression or not expression.strip():
        return {
            "expression": expression,
            "count": 0,
            "selection_error": "empty expression",
        }
    scope = dict(namespace)
    scope["result"] = shape
    try:
        # eval is deliberate and adds no capability the caller does not already have: the
        # line in main() above exec()s the model's entire script from the same source, and
        # this harness is a crash/timeout isolation boundary rather than a security one
        # (see the module docstring of sandbox/executor.py). The alternative — a bespoke
        # parser for build123d's ShapeList grammar — would reject valid selections and give
        # the model worse diagnostics, which is the whole point of the probe.
        matched = eval(expression, scope)  # noqa: S307
    except Exception as exc:  # noqa: BLE001 — a bad expression is a result, not a crash
        return {
            "expression": expression,
            "count": 0,
            "selection_error": f"{type(exc).__name__}: {exc}",
        }

    from build123d import Shape

    if isinstance(matched, Shape):
        entities = [matched]
    else:
        try:
            entities = list(matched)
        except TypeError:
            return {
                "expression": expression,
                "count": 0,
                "selection_error": (
                    f"expression produced {type(matched).__name__}, which is neither a Shape "
                    "nor a list of shapes — end it with .edges()/.faces() or an index"
                ),
            }
    non_shapes = [e for e in entities if not isinstance(e, Shape)]
    if non_shapes:
        return {
            "expression": expression,
            "count": 0,
            "selection_error": (
                f"expression produced {len(non_shapes)} non-Shape item(s) such as "
                f"{type(non_shapes[0]).__name__}"
            ),
        }
    return {
        "expression": expression,
        "count": len(entities),
        "matches": [_entity_info(e) for e in entities[:_MAX_PER_GROUP]],
        "truncated": len(entities) > _MAX_PER_GROUP,
        "selection_error": None,
    }


def main() -> None:
    code_file, query_file = Path(sys.argv[1]), Path(sys.argv[2])
    code = code_file.read_text()
    query = json.loads(query_file.read_text())

    shown = []
    namespace = build123d_namespace(shown)
    exec(compile(code, str(code_file), "exec"), namespace)

    shape = _load_shape(namespace, shown)

    mode = query.get("mode", "describe")
    if mode == "selection":
        result = _run_selection(shape, namespace, query.get("expression", ""))
    else:
        result = _describe(shape)
    result["mode"] = mode

    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

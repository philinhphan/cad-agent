"""Subprocess harness: execute generated build123d code and export artifacts.

The build123d counterpart of ``harness.py``. Invoked as
``python harness_build123d.py <code_file> <out_dir>``. Deliberately self-contained
(stdlib + build123d only) — it never imports the cad_gen package, so it can run in
any interpreter that has build123d installed.

Contract (identical to harness.py, so the executor and every downstream consumer
cannot tell the two apart):
- user code runs with the build123d API pre-imported and a `show_object()` shim
- the model is taken from the `result` variable, else show_object() calls,
  else any Shape/Builder left in the namespace
- on success: writes model.stl, model.step, metrics.json into out_dir, exit 0
- on any failure: traceback on stderr, exit 1
"""

import json
import sys
import traceback
from pathlib import Path


def build123d_namespace(shown):
    """Namespace mirroring `from build123d import *`, plus the show_object shim.

    build123d exposes its public API through `__all__`; replicating it here means the
    generated script does not have to write the import itself (though writing it is
    harmless and idempotent).
    """
    import build123d

    names = getattr(build123d, "__all__", None) or [
        n for n in dir(build123d) if not n.startswith("_")
    ]
    namespace = {name: getattr(build123d, name) for name in names if hasattr(build123d, name)}
    namespace["__name__"] = "__cadgen_model__"
    namespace["build123d"] = build123d
    namespace["show_object"] = lambda obj, *args, **kwargs: shown.append(obj)
    return namespace


def _unwrap(obj):
    """Return the Shape a candidate carries, or None.

    A BuildPart/BuildSketch/BuildLine builder is not itself a Shape — the geometry hangs
    off `.part` / `.sketch` / `.line` — so assigning the builder to `result` (the natural
    thing to do, and what the prompt permits) has to work too.
    """
    import build123d

    if isinstance(obj, build123d.Shape):
        return obj
    if isinstance(obj, build123d.Builder):
        for attr in ("part", "sketch", "line"):
            candidate = getattr(obj, attr, None)
            if isinstance(candidate, build123d.Shape):
                return candidate
    return None


def _collect_shapes(namespace, shown):
    import build123d

    if _unwrap(namespace.get("result")) is not None:
        candidates = [namespace["result"]]
    elif shown:
        candidates = list(shown)
    else:
        candidates = [
            v
            for k, v in namespace.items()
            if not k.startswith("_")
            and isinstance(v, (build123d.Shape, build123d.Builder))
            # Skip the API itself: `from build123d import *` binds classes such as Box and
            # Plane.XY into the namespace, and those are types/instances we must not mistake
            # for the model the script built.
            and not isinstance(v, type)
        ]

    shapes = []
    for obj in candidates:
        shape = _unwrap(obj)
        if shape is not None:
            shapes.append(shape)
    return shapes


def main() -> None:
    code_file, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    code = code_file.read_text()

    from build123d import CenterOf, Compound, export_step, export_stl

    shown = []
    namespace = build123d_namespace(shown)
    exec(compile(code, str(code_file), "exec"), namespace)

    shapes = _collect_shapes(namespace, shown)
    if not shapes:
        raise RuntimeError(
            "No geometry produced: assign the final solid to a variable named "
            "`result` (a build123d Shape, or the BuildPart builder that made it)."
        )

    shape = shapes[0] if len(shapes) == 1 else Compound(children=shapes)

    # Measure BEFORE exporting: STL export tessellates the shape, and OCCT
    # bounding boxes computed afterwards include the mesh sag of curved faces
    # (microns to millimetres), which corrupts exact-dimension checks.
    bb = shape.bounding_box()
    com = shape.center(CenterOf.MASS)
    metrics = {
        "volume_mm3": shape.volume,
        "bbox_mm": [bb.size.X, bb.size.Y, bb.size.Z],
        "center_of_mass": [com.X, com.Y, com.Z],
        "n_solids": len(shape.solids()),
        "n_faces": len(shape.faces()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics))

    export_stl(shape, str(out_dir / "model.stl"))
    export_step(shape, str(out_dir / "model.step"))

    # build123d's exporters report failure by return value in some paths rather than
    # raising; the executor only checks that the files exist, so make a silent
    # non-export an explicit failure here.
    for name in ("model.stl", "model.step"):
        if not (out_dir / name).exists():
            raise RuntimeError(f"build123d failed to write {name}.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

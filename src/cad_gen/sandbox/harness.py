"""Subprocess harness: execute generated CadQuery code and export artifacts.

Invoked as `python harness.py <code_file> <out_dir>`. Deliberately
self-contained (stdlib + cadquery only) — it never imports the cad_gen
package, so it can run in any interpreter that has cadquery installed.

Contract:
- user code runs with `cq`/`cadquery` pre-imported and a `show_object()` shim
- the model is taken from the `result` variable, else show_object() calls,
  else any Workplane/Shape left in the namespace
- on success: writes model.stl, model.step, metrics.json into out_dir, exit 0
- on any failure: traceback on stderr, exit 1
"""

import json
import sys
import traceback
from pathlib import Path


def _cylinders(shape):
    """Cylindrical B-rep faces as ``{radius_mm, axis}`` — measured off the exact solid for
    the deterministic hole check (holes, counterbores and round slots are cylinder faces).

    Best-effort and defensive: this runs in the sandbox subprocess, so any OCCT/import
    hiccup returns an empty list rather than failing an otherwise-good build.
    """
    out: list[dict] = []
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Cylinder
    except Exception:
        return out
    for face in shape.Faces():
        try:
            surf = BRepAdaptor_Surface(face.wrapped)
            if surf.GetType() != GeomAbs_Cylinder:
                continue
            cyl = surf.Cylinder()
            ax1 = cyl.Axis()
            axis = ax1.Direction()
            loc = ax1.Location()
            out.append(
                {
                    "radius_mm": float(cyl.Radius()),
                    "axis": [float(axis.X()), float(axis.Y()), float(axis.Z())],
                    "location": [float(loc.X()), float(loc.Y()), float(loc.Z())],
                }
            )
        except Exception:
            continue
    return out


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


def main() -> None:
    code_file, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    code = code_file.read_text()

    import cadquery as cq

    shown = []
    namespace = {
        "__name__": "__cadgen_model__",
        "cq": cq,
        "cadquery": cq,
        "show_object": lambda obj, *args, **kwargs: shown.append(obj),
    }
    exec(compile(code, str(code_file), "exec"), namespace)

    shapes = _collect_shapes(namespace, shown)
    if not shapes:
        raise RuntimeError(
            "No geometry produced: assign the final solid to a variable named "
            "`result` (a cadquery Workplane or Shape)."
        )

    shape = shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)

    # Measure BEFORE exporting: STL export tessellates the shape, and OCCT
    # bounding boxes computed afterwards include the mesh sag of curved faces
    # (microns to millimetres), which corrupts exact-dimension checks.
    bb = shape.BoundingBox()
    com = shape.Center()
    metrics = {
        "volume_mm3": shape.Volume(),
        "bbox_mm": [bb.xlen, bb.ylen, bb.zlen],
        "center_of_mass": [com.x, com.y, com.z],
        "n_solids": len(shape.Solids()),
        "n_faces": len(shape.Faces()),
        "cylinders": _cylinders(shape),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics))

    cq.exporters.export(shape, str(out_dir / "model.stl"))
    cq.exporters.export(shape, str(out_dir / "model.step"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

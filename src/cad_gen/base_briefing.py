"""A feature inventory of the base model an editing task is asked to modify.

Editing instructions name features in engineering language — "the four non-circular pockets
on the +X side of the central bore", "the wall further in the -X direction with one large
hole", "the groove cut out of the inside of the largest-diameter bore". To act on one, the
generator has to map that phrase onto specific faces of an imported B-rep, and real benchmark
bases carry 334 to 2157 faces.

What it had to work with was ``inspect_geometry``, which groups faces by geometry type and
returns the first ``_MAX_PER_GROUP = 12`` of each group in traversal order. On a 1000-face
part that is a near-random 3% sample. The v3 candidates show the consequence directly: the
generated scripts give up on selectors and write ad-hoc ``for f in obj.Faces()`` scans with
hand-tuned area thresholds, and the resulting edits land in the wrong place — sample 224 cut
a full-length cylinder to remove one internal groove, sample 201 rebuilt the part outright.

So this module reads the base model once per run and writes down what is actually in it:

- **Bores** — cylindrical faces clustered by (axis direction, axis position, radius), which
  is what makes a hole one feature instead of four quarter-cylinder faces. Each entry carries
  its radius, axis, extent along that axis, and whether it is a full 2pi hole or a partial
  sweep (a fillet).
- **Planar walls** — planar faces clustered by (normal, offset from origin), so "the two
  primary walls parallel to the YZ plane" resolves to two entries rather than 40 faces.
- **Everything else** — counts by surface type, so a part dominated by B-splines says so
  instead of silently omitting half its geometry.

Read through raw OCP, like `step_metrics.py` and `step_validity.py`, so the briefing is the
same whichever CAD library the generator writes in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

# Clustering tolerances. Deliberately loose: the goal is to name features the way the
# instruction does, so two coaxial cylinder faces split by a seam must land in one bore, and
# two faces of one wall milled to the same depth must land in one plane. Being slightly too
# coarse merges two genuinely distinct features into one line of text; being too fine floods
# the briefing with duplicates, which is the failure mode that made the old output useless.
_AXIS_ANGLE_TOL = math.radians(1.0)
_POSITION_TOL_MM = 1e-3
_RADIUS_TOL_MM = 1e-3

_FULL_SWEEP_TOL = math.radians(5.0)  # within this of 2pi counts as a complete hole

# The briefing is prompt text, so it has a budget. Features are emitted largest-first, and
# the tail of a real part is fillets and blend faces the instruction never refers to.
_MAX_BORES = 30
_MAX_PLANES = 30


@dataclass(frozen=True)
class Bore:
    """A cylindrical feature: a hole, a bore, or (when partial) a fillet."""

    radius_mm: float
    axis: tuple[float, float, float]
    axis_point_mm: tuple[float, float, float]  # a point on the axis, at the feature's start
    extent_mm: float  # length along the axis
    full_circle: bool  # 2pi sweep => hole/bore; partial => fillet or a rounded corner
    face_count: int
    area_mm2: float


@dataclass(frozen=True)
class PlanarGroup:
    """Coplanar faces sharing one normal and offset — an engineering "wall" or "face"."""

    normal: tuple[float, float, float]
    offset_mm: float  # signed distance from the origin along the normal
    area_mm2: float
    face_count: int
    bbox_min_mm: tuple[float, float, float]
    bbox_max_mm: tuple[float, float, float]


@dataclass(frozen=True)
class BaseBriefing:
    bbox_min_mm: tuple[float, float, float]
    bbox_max_mm: tuple[float, float, float]
    volume_mm3: float
    n_solids: int
    n_faces: int
    n_edges: int
    bores: tuple[Bore, ...]
    planes: tuple[PlanarGroup, ...]
    other_surfaces: tuple[tuple[str, int], ...]  # (surface type, face count), commonest first

    @property
    def extent_mm(self) -> tuple[float, float, float]:
        return tuple(hi - lo for lo, hi in zip(self.bbox_min_mm, self.bbox_max_mm, strict=True))


def build_base_briefing(step_path: str | Path) -> BaseBriefing | None:
    """Read `step_path` and inventory its features; None if it cannot be read.

    Returns None rather than raising: the briefing is a prompt enrichment, and a run whose
    base model defeats the inventory should still attempt the edit.
    """
    try:
        return _inventory(Path(step_path))
    except Exception:  # noqa: BLE001 — advisory enrichment, never a hard failure
        return None


def format_briefing(briefing: BaseBriefing) -> str:
    """Render the inventory as the prompt block the generator sees.

    Coordinates are absolute and in millimetres, matching what `inspect_geometry` reports, so
    a feature located here can be cut with a coordinate-positioned primitive without any
    further conversion — which is the construction the editing rules ask for.
    """
    lo, hi = briefing.bbox_min_mm, briefing.bbox_max_mm
    extent = briefing.extent_mm
    lines = [
        "## BASE MODEL BRIEFING (`input.step`)",
        "",
        "A deterministic inventory of the model you are editing, measured from the B-rep. Use "
        "it to locate the feature the instruction names, then confirm with `inspect_geometry` "
        "before you cut.",
        "",
        f"- bounding box: x {lo[0]:.3f}..{hi[0]:.3f}, y {lo[1]:.3f}..{hi[1]:.3f}, "
        f"z {lo[2]:.3f}..{hi[2]:.3f} mm  (extent {extent[0]:.3f} x {extent[1]:.3f} x "
        f"{extent[2]:.3f})",
        f"- volume: {briefing.volume_mm3:.1f} mm3",
        f"- {briefing.n_solids} solid(s), {briefing.n_faces} faces, {briefing.n_edges} edges",
    ]

    if briefing.bores:
        lines += ["", "### Cylindrical features (holes, bores, fillets), largest radius first", ""]
        for bore in briefing.bores:
            kind = "HOLE/BORE" if bore.full_circle else "fillet/partial"
            axis = ", ".join(f"{v:+.3f}" for v in bore.axis)
            point = ", ".join(f"{v:.3f}" for v in bore.axis_point_mm)
            lines.append(
                f"- r={bore.radius_mm:.3f} mm  {kind}  axis ({axis})  from ({point})  "
                f"length {bore.extent_mm:.3f} mm  [{bore.face_count} face(s)]"
            )

    if briefing.planes:
        lines += ["", "### Planar walls, grouped by normal and offset, largest area first", ""]
        for plane in briefing.planes:
            normal = ", ".join(f"{v:+.3f}" for v in plane.normal)
            plo, phi = plane.bbox_min_mm, plane.bbox_max_mm
            lines.append(
                f"- normal ({normal})  offset {plane.offset_mm:+.3f} mm  "
                f"area {plane.area_mm2:.1f} mm2  [{plane.face_count} face(s)]  "
                f"spans x {plo[0]:.1f}..{phi[0]:.1f}, y {plo[1]:.1f}..{phi[1]:.1f}, "
                f"z {plo[2]:.1f}..{phi[2]:.1f}"
            )

    if briefing.other_surfaces:
        described = ", ".join(f"{count} x {name}" for name, count in briefing.other_surfaces)
        lines += ["", f"### Other surfaces: {described}"]

    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------------------


def _inventory(step_path: Path) -> BaseBriefing:
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape

    shape = _read_shape(step_path)
    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
    edge_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, edge_map)
    solid_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_SOLID, solid_map)

    cylinders: list[dict] = []
    planes: list[dict] = []
    others: dict[str, int] = {}

    for i in range(1, face_map.Size() + 1):
        face = TopoDS.Face_s(face_map.FindKey(i))
        adaptor = BRepAdaptor_Surface(face)
        kind = adaptor.GetType()
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, props)
        area = float(props.Mass())
        if kind == GeomAbs_SurfaceType.GeomAbs_Cylinder:
            cylinders.append(_cylinder_info(face, adaptor, area))
        elif kind == GeomAbs_SurfaceType.GeomAbs_Plane:
            planes.append(_plane_info(face, adaptor, area, BRep_Tool))
        else:
            name = str(kind).rsplit("_", 1)[-1]
            others[name] = others.get(name, 0) + 1

    volume_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume_props)
    lo, hi = _bounds(shape)

    return BaseBriefing(
        bbox_min_mm=lo,
        bbox_max_mm=hi,
        volume_mm3=float(volume_props.Mass()),
        n_solids=solid_map.Size(),
        n_faces=face_map.Size(),
        n_edges=edge_map.Size(),
        bores=_cluster_cylinders(cylinders),
        planes=_cluster_planes(planes),
        other_surfaces=tuple(sorted(others.items(), key=lambda kv: kv[1], reverse=True)),
    )


def _read_shape(path: Path):
    """Load a STEP file, refusing to transfer a model the parser rejected.

    The ReadStatus check is load-bearing, not defensive tidiness: calling TransferRoots()
    after a failed parse SEGFAULTS, which no `except` can catch and which would take the
    whole run down with it. This module runs in-process (unlike `step_validity` and
    `edit_diff`, which are subprocess-isolated), so it has to not crash.
    """
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise ValueError(f"{path} could not be parsed as STEP")
    reader.TransferRoots()
    return reader.OneShape()


def _bounds(shape) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box)  # see step_metrics.measure_step for why not Add_s
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (xmin, ymin, zmin), (xmax, ymax, zmax)


def _canonical_direction(direction) -> tuple[float, float, float]:
    """Normalise an axis to one hemisphere so +Z and -Z cluster as the same axis.

    A through-hole's faces can be authored with either sense; treating them as two features
    would list every hole twice.
    """
    vec = (direction.X(), direction.Y(), direction.Z())
    for component in vec:
        if abs(component) > 1e-9:
            return vec if component > 0 else tuple(-v for v in vec)
    return vec


def _cylinder_info(face, adaptor, area: float) -> dict:
    from OCP.BRepTools import BRepTools

    cylinder = adaptor.Cylinder()
    axis = cylinder.Axis()
    umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
    location = axis.Location()
    direction = _canonical_direction(axis.Direction())
    # v parametrises distance along the axis for a cylinder, u the angle around it.
    return {
        "radius": float(cylinder.Radius()),
        "axis": direction,
        "point": (location.X(), location.Y(), location.Z()),
        "v_range": (float(vmin), float(vmax)),
        "sweep": abs(float(umax) - float(umin)),
        "area": area,
    }


def _plane_info(face, adaptor, area: float, brep_tool) -> dict:
    plane = adaptor.Plane()
    axis = plane.Axis()
    location = axis.Location()
    normal = _canonical_direction(axis.Direction())
    offset = sum(n * p for n, p in zip(normal, (location.X(), location.Y(), location.Z())))
    lo, hi = _bounds(face)
    return {"normal": normal, "offset": offset, "area": area, "bbox_min": lo, "bbox_max": hi}


def _same_axis(a: dict, b: dict) -> bool:
    """True when two cylinders are coaxial and the same size.

    Coaxial means parallel directions AND the perpendicular distance between the two axis
    lines is negligible — parallel alone would merge every hole in a bolt circle into one.
    """
    dot = abs(sum(x * y for x, y in zip(a["axis"], b["axis"], strict=True)))
    if dot < math.cos(_AXIS_ANGLE_TOL):
        return False
    if abs(a["radius"] - b["radius"]) > _RADIUS_TOL_MM:
        return False
    delta = tuple(x - y for x, y in zip(a["point"], b["point"], strict=True))
    along = sum(d * n for d, n in zip(delta, a["axis"], strict=True))
    perpendicular = math.sqrt(max(sum(d * d for d in delta) - along * along, 0.0))
    return perpendicular <= _POSITION_TOL_MM


def _cluster_cylinders(cylinders: list[dict]) -> tuple[Bore, ...]:
    clusters: list[list[dict]] = []
    for cylinder in cylinders:
        for cluster in clusters:
            if _same_axis(cluster[0], cylinder):
                cluster.append(cylinder)
                break
        else:
            clusters.append([cylinder])

    bores: list[Bore] = []
    for cluster in clusters:
        head = cluster[0]
        v_lo = min(c["v_range"][0] for c in cluster)
        v_hi = max(c["v_range"][1] for c in cluster)
        sweep = sum(c["sweep"] for c in cluster)
        start = tuple(p + v_lo * a for p, a in zip(head["point"], head["axis"], strict=True))
        bores.append(
            Bore(
                radius_mm=head["radius"],
                axis=head["axis"],
                axis_point_mm=start,
                extent_mm=v_hi - v_lo,
                full_circle=sweep >= 2 * math.pi - _FULL_SWEEP_TOL,
                face_count=len(cluster),
                area_mm2=sum(c["area"] for c in cluster),
            )
        )
    bores.sort(key=lambda b: (b.full_circle, b.radius_mm), reverse=True)
    return tuple(bores[:_MAX_BORES])


def _cluster_planes(planes: list[dict]) -> tuple[PlanarGroup, ...]:
    clusters: list[list[dict]] = []
    for plane in planes:
        for cluster in clusters:
            head = cluster[0]
            dot = abs(sum(x * y for x, y in zip(head["normal"], plane["normal"], strict=True)))
            if dot >= math.cos(_AXIS_ANGLE_TOL) and abs(head["offset"] - plane["offset"]) <= _POSITION_TOL_MM:
                cluster.append(plane)
                break
        else:
            clusters.append([plane])

    groups = [
        PlanarGroup(
            normal=cluster[0]["normal"],
            offset_mm=cluster[0]["offset"],
            area_mm2=sum(c["area"] for c in cluster),
            face_count=len(cluster),
            bbox_min_mm=tuple(min(c["bbox_min"][i] for c in cluster) for i in range(3)),
            bbox_max_mm=tuple(max(c["bbox_max"][i] for c in cluster) for i in range(3)),
        )
        for cluster in clusters
    ]
    groups.sort(key=lambda g: g.area_mm2, reverse=True)
    return tuple(groups[:_MAX_PLANES])

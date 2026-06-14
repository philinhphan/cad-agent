from cad_gen.drawing_constraints import (
    derive_drawing_constraints,
    validate_drawing_constraints,
)
from cad_gen.models import GeometryMetrics


def test_derives_structured_constraints_from_interpretation_markdown():
    text = """
    ## Overall envelope
    - **135 x 85 x 65 mm** overall envelope.

    ## Features
    - Central hole: **Ø15 THRU**
    - Counterbore: **⌴ Ø30 depth 12**
    - Fillets called out: **R20, 2 places** and **R77**
    - Inclined rear face: **15°**
    """

    constraints = derive_drawing_constraints(text)

    assert constraints.envelope_mm == (135.0, 85.0, 65.0)
    assert constraints.holes[0].diameter_mm == 15.0
    assert constraints.holes[0].count == 1
    assert constraints.holes[0].through is True
    assert constraints.counterbores[0].diameter_mm == 30.0
    assert constraints.counterbores[0].depth_mm == 12.0
    assert {r.value_mm for r in constraints.radii} >= {20.0, 77.0}
    assert constraints.angles[0].degrees == 15.0


def test_validate_constraints_checks_bbox_and_marks_unverified_features():
    constraints = derive_drawing_constraints("Envelope 135 x 85 x 65 mm. 2x Ø5 THRU. R20.")
    metrics = GeometryMetrics(
        volume_mm3=1.0,
        bbox_mm=(135.0, 80.0, 65.0),
        center_of_mass=(0.0, 0.0, 0.0),
        n_solids=1,
        n_faces=12,
    )

    report = validate_drawing_constraints(metrics, constraints)

    assert report.passed is False
    bbox_check = next(c for c in report.checks if c.name == "overall envelope")
    assert bbox_check.status == "fail"
    assert "expected 135 x 85 x 65 mm" in bbox_check.message
    assert any(c.status == "unverified" and "hole" in c.name for c in report.checks)
    assert "Constraint validation" in report.digest



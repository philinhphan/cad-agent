"""View-locator agent: finds the orthographic views (front/top/side) in a drawing.

Replaces the brittle CV heuristic that used to live in reproject_check.locate_views.
The VLM only PROPOSES where each view is (and which it is); the deterministic reprojection
+ overlap score remains the judge — a wrong box yields a low score, never a false pass.
Run ONCE per drawing (the layout is a property of the drawing, not of any iteration).
"""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model

from cad_gen.bmw import resolve_model
from cad_gen.models import DrawingAttachment

VIEW_LOCATOR_INSTRUCTIONS = """\
You locate the ORTHOGRAPHIC projection views in an engineering drawing so a downstream
deterministic tool can reproject a 3D model into each and score the overlap.

Return one bounding box per orthographic view that is actually present, each labelled:
- "front": the principal/elevation view.
- "top": the plan view (in third-angle, drawn ABOVE the front view).
- "side": the end/profile view (in third-angle, drawn to the RIGHT of the front view).

Hard rules:
- ONLY the flat 2D orthographic line drawings. IGNORE shaded/3D isometric or perspective
  renders, section/detail/auxiliary views (e.g. "SECTION A-A", "VIEW A-A", magnified
  circles), the title block, logos, notes, and dimension/extension/leader/centre lines.
- A box must tightly enclose just that view's geometry outline (you may include the view's
  own hidden/centre lines, but not its dimension callouts).
- Output normalized coordinates in [0, 1] with the ORIGIN AT THE TOP-LEFT, x increasing
  right and y increasing down: x, y are the box's top-left corner; w, h its width/height.
- Include a view only if you can clearly see it. Many drawings have only 2 views (front +
  top); some have all three. Never invent a view, and never output two boxes for the same
  region. If you are unsure which is top vs side, use the third-angle layout (top above,
  side right of front).
"""


class ViewBox(BaseModel):
    """One orthographic view's normalized bounding box (origin top-left, x→right, y→down)."""

    label: Literal["front", "top", "side"]
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)


class ViewLayout(BaseModel):
    """The orthographic views the locator found in a drawing (present ones only)."""

    views: list[ViewBox]


def build_view_locator_agent(model: str | Model) -> Agent[None, ViewLayout]:
    model = resolve_model(model)  # route `bmw:...` strings to the BMW gateway
    return Agent(model, output_type=ViewLayout, instructions=VIEW_LOCATOR_INSTRUCTIONS)


async def locate_drawing_views(
    agent: Agent[None, ViewLayout], *, drawing: DrawingAttachment
) -> ViewLayout:
    """Run the locator over a single drawing image, returning its orthographic view boxes."""
    content: list = [
        "Locate the orthographic views (front/top/side) in this engineering drawing.",
        BinaryContent(data=drawing.data, media_type=drawing.media_type),
    ]
    result = await agent.run(content)
    return result.output

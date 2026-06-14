"""View-locator agent test: a scripted VLM returns a ViewLayout structured output."""

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from cad_gen.reproject.locator import (
    ViewLayout,
    build_view_locator_agent,
    locate_drawing_views,
)
from cad_gen.models import DrawingAttachment

PNG = b"\x89PNG\r\n\x1a\nfakepng"


def _layout_model(views: list[dict]) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"views": views})])

    return FunctionModel(fn)


async def test_locate_drawing_views_parses_boxes():
    model = _layout_model(
        [
            {"label": "front", "x": 0.13, "y": 0.66, "w": 0.23, "h": 0.39},
            {"label": "top", "x": 0.13, "y": 0.18, "w": 0.13, "h": 0.30},
        ]
    )
    agent = build_view_locator_agent(model)
    drawing = DrawingAttachment(filename="d.png", media_type="image/png", data=PNG)

    layout = await locate_drawing_views(agent, drawing=drawing)

    assert isinstance(layout, ViewLayout)
    assert {v.label for v in layout.views} == {"front", "top"}
    front = next(v for v in layout.views if v.label == "front")
    assert front.x == 0.13 and front.h == 0.39


async def test_empty_layout_is_allowed():
    agent = build_view_locator_agent(_layout_model([]))
    drawing = DrawingAttachment(filename="d.png", media_type="image/png", data=PNG)

    layout = await locate_drawing_views(agent, drawing=drawing)

    assert layout.views == []

"""Tests for per-library prompt assembly.

The CadQuery prompts are pinned to snapshots captured BEFORE `prompts.py` was refactored
from flat constants into composed per-library instructions. They are heavily tuned from
real failures (which selectors match nothing, which ops crash on empty selections), so a
silent reword during refactoring would degrade generation quality with nothing else
failing. If a CadQuery prompt is ever changed deliberately, update the snapshot in the
same commit — that makes the change reviewable instead of invisible.
"""

from pathlib import Path

import pytest

from cad_gen.agents.prompts import (
    DRAWING_PARSER_INSTRUCTIONS,
    critic_instructions,
    generator_instructions,
)
from cad_gen.models import CAD_LIBRARIES

SNAPSHOTS = Path(__file__).parent / "snapshots"


@pytest.mark.parametrize(
    ("name", "kwargs", "builder"),
    [
        ("generator_cadquery.txt", {}, generator_instructions),
        ("generator_cadquery_editing.txt", {"editing": True}, generator_instructions),
        ("critic_cadquery.txt", {}, critic_instructions),
        ("critic_cadquery_editing.txt", {"editing": True}, critic_instructions),
    ],
)
def test_cadquery_prompts_match_snapshot(name, kwargs, builder):
    assert builder("cadquery", **kwargs) == (SNAPSHOTS / name).read_text()


@pytest.mark.parametrize("editing", [False, True])
def test_build123d_generator_prompt_teaches_build123d(editing):
    text = generator_instructions("build123d", editing=editing)

    assert "build123d" in text
    assert "BuildPart" in text
    # No CadQuery leakage: a prompt mixing both APIs produces code that imports neither
    # cleanly, and the model has no way to tell which dialect is wanted.
    assert "CadQuery" not in text
    assert "cq.Workplane" not in text


@pytest.mark.parametrize("editing", [False, True])
def test_build123d_critic_prompt_names_build123d(editing):
    text = critic_instructions("build123d", editing=editing)

    assert "build123d" in text
    assert "CadQuery" not in text


def test_editing_prompts_load_the_seeded_base_model():
    """Each library must be told the call that reads the sandbox-seeded input.step."""
    assert 'cq.importers.importStep("input.step")' in generator_instructions(
        "cadquery", editing=True
    )
    assert 'import_step("input.step")' in generator_instructions("build123d", editing=True)


@pytest.mark.parametrize("library", CAD_LIBRARIES)
@pytest.mark.parametrize("editing", [False, True])
def test_every_prompt_is_non_empty_and_mentions_its_probe_tools(library, editing):
    text = generator_instructions(library, editing=editing)

    assert text.strip()
    assert "execute_cad_code" in text
    assert "inspect_geometry" in text
    # The selection probe is named differently per library because the APIs differ.
    expected_probe = "check_selector" if library == "cadquery" else "check_selection"
    assert expected_probe in text


@pytest.mark.parametrize("library", CAD_LIBRARIES)
def test_shared_task_guidance_is_identical_across_libraries(library):
    """Drawing/overlay guidance is about the task, not the language — it must not drift."""
    text = generator_instructions(library)

    assert "THE DRAWING IMAGE IS AUTHORITATIVE" in text
    assert "reprojection OVERLAY image" in text
    assert "never estimate a dimension from the overlay" in text


def test_unknown_library_rejected():
    with pytest.raises(ValueError, match="Unknown CAD library"):
        generator_instructions("openscad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown CAD library"):
        critic_instructions("openscad")  # type: ignore[arg-type]


def test_drawing_parser_prompt_is_library_agnostic():
    """It transcribes drawings into prose, so it must not name a CAD library at all."""
    assert "CadQuery" not in DRAWING_PARSER_INSTRUCTIONS
    assert "build123d" not in DRAWING_PARSER_INSTRUCTIONS

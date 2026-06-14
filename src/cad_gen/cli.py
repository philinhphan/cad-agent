"""Command-line interface: `cad-gen "<spec>" [options]`."""

import asyncio
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from cad_gen.agents.drawing_parser import build_drawing_parser_agent, interpret_drawing
from cad_gen.imaging import media_type_for
from cad_gen.models import DrawingAttachment, IterationRecord, RunConfig
from cad_gen.orchestrator import generate_cad

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def generate(
    spec: str = typer.Argument(
        None, help="Natural-language part specification (optional if --drawing is given)."
    ),
    drawing: list[Path] = typer.Option(
        None,
        "--drawing",
        "-d",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Engineering drawing image (JPEG/PNG) to build from. Repeatable.",
    ),
    no_review: bool = typer.Option(
        False,
        "--no-review",
        help="Skip the editable review of the drawing's extracted dimensions.",
    ),
    max_iterations: int = typer.Option(
        5, "--max-iterations", "-n", min=1, help="Outer self-refine iteration budget."
    ),
    threshold: int = typer.Option(
        8, "--threshold", "-t", min=0, max=10, help="Critic score needed to accept."
    ),
    model: str = typer.Option(
        None, "--model", "-m", help="Generator model (default google:gemini-3.5-flash)."
    ),
    critic_model: str = typer.Option(
        None, "--critic-model", help="Vision critic model (default google:gemini-3.5-flash)."
    ),
    timeout: float = typer.Option(
        60.0, "--timeout", help="Sandbox execution timeout per attempt (seconds)."
    ),
    out: Path = typer.Option(
        Path("runs"), "--out", "-o", help="Directory for run artifacts."
    ),
) -> None:
    """Generate CAD geometry from a text spec and/or technical drawing via a self-refine loop."""
    # load_dotenv must run before RunConfig is built below — RunConfig resolves
    # CAD_GEN_MODEL / CAD_GEN_CRITIC_MODEL from the environment; --model /
    # --critic-model flags (set here) override that.
    load_dotenv(Path.cwd() / ".env")

    spec = spec or ""
    if not spec.strip() and not drawing:
        console.print("[red bold]Provide a spec, a --drawing, or both.[/red bold]")
        raise typer.Exit(2)
    drawings = [_load_drawing(p) for p in (drawing or [])]

    config_kwargs = dict(
        max_iterations=max_iterations,
        score_threshold=threshold,
        exec_timeout_s=timeout,
        out_dir=out,
    )
    if model:
        config_kwargs["model"] = model
    if critic_model:
        config_kwargs["critic_model"] = critic_model
    config = RunConfig(**config_kwargs)

    _require_api_key(config)

    console.print(f"[bold]cad-gen[/bold]  model={config.model}  critic={config.critic_model}")
    console.print(f"spec: [italic]{spec or '(from drawing)'}[/italic]")
    if drawings:
        console.print(f"drawings: [italic]{', '.join(d.filename for d in drawings)}[/italic]")
    console.print()

    interpretation = _review_drawings(spec, drawings, config, no_review=no_review)

    result = asyncio.run(
        generate_cad(
            spec,
            config,
            drawings=drawings,
            interpretation=interpretation,
            on_iteration=_print_iteration,
        )
    )

    console.print()
    if result.accepted:
        console.print(
            f"[green bold]ACCEPTED[/green bold] — score "
            f"{result.best.effective_score}/10 (iteration {result.best.index})"
        )
    else:
        console.print(
            f"[yellow bold]BUDGET EXHAUSTED[/yellow bold] — best score "
            f"{result.best.effective_score}/10 (iteration {result.best.index}); "
            "returning best effort"
        )
    console.print(f"artifacts: {result.run_dir}")
    console.print("  final/model.step · final/model.stl · final/views.png · report.md")
    raise typer.Exit(0 if result.accepted else 1)


def _load_drawing(path: Path) -> DrawingAttachment:
    data = path.read_bytes()
    media = media_type_for(path.name, data)
    if media is None:
        console.print(
            f"[red bold]{path.name} is not a PNG or JPEG image.[/red bold] "
            "Only --drawing files of type image/png or image/jpeg are supported."
        )
        raise typer.Exit(2)
    return DrawingAttachment(filename=path.name, media_type=media, data=data)


def _review_drawings(
    spec: str, drawings: list[DrawingAttachment], config: RunConfig, *, no_review: bool
) -> str | None:
    """Extract the drawing's dimensions and (unless skipped) let the user edit them."""
    if not drawings:
        return None
    console.print("[bold]reading drawing…[/bold] extracting dimensions")
    interpreter = build_drawing_parser_agent(config.model)
    interpretation = asyncio.run(
        interpret_drawing(interpreter, spec=spec, drawings=drawings)
    )

    if not no_review and sys.stdin.isatty():
        console.print(
            "opening the extracted dimensions in your editor — correct anything wrong, "
            "save and close to continue."
        )
        edited = _edit_text(interpretation)
        if edited.strip():
            interpretation = edited

    console.rule("[bold]extracted dimensions[/bold]")
    console.print(interpretation)
    console.rule()
    return interpretation


def _edit_text(text: str) -> str:
    """Open `text` in $VISUAL/$EDITOR (fallback: vi) and return the edited content."""
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        subprocess.run([*shlex.split(editor), tmp], check=True)
        return Path(tmp).read_text()
    except (subprocess.SubprocessError, OSError):
        return text  # editor unavailable/failed → keep the extraction unchanged
    finally:
        Path(tmp).unlink(missing_ok=True)


# provider prefix -> env var(s) that satisfy it (any one suffices).
_PROVIDER_KEYS = {
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google-gla": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google-vertex": ("GOOGLE_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}


def _require_api_key(config: RunConfig) -> None:
    providers = {
        m.split(":", 1)[0] for m in (config.model, config.critic_model, config.view_model)
    }
    missing = [
        (p, keys)
        for p in providers
        if (keys := _PROVIDER_KEYS.get(p)) and not any(os.environ.get(k) for k in keys)
    ]
    if missing:
        for provider, keys in missing:
            console.print(
                f"[red bold]{' or '.join(keys)} is not set[/red bold] — required by "
                f"the '{provider}' model provider."
            )
        console.print("Add it to a .env file (see .env.example) or export it.")
        raise typer.Exit(2)


def _print_iteration(record: IterationRecord) -> None:
    executed = record.execution is not None and record.execution.success
    if record.critique is not None:
        mark = "[green]✓[/green]" if record.critique.matches_spec else "[red]✗[/red]"
        detail = record.critique.issues[0] if record.critique.issues else record.critique.summary
    else:
        mark = "[red]✗[/red]"
        detail = "execution failed" if not executed else "no critique"
        if record.execution is not None and record.execution.error:
            detail = f"execution failed: {record.execution.error.splitlines()[-1]}"
    console.print(
        f"  iter {record.index}  score {record.effective_score}/10  {mark} {detail}"
    )


if __name__ == "__main__":
    app()

"""Command-line interface: `cad-gen "<spec>" [options]`."""

import asyncio
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from cad_gen.models import IterationRecord, RunConfig
from cad_gen.orchestrator import generate_cad

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def generate(
    spec: str = typer.Argument(..., help="Natural-language part specification."),
    max_iterations: int = typer.Option(
        5, "--max-iterations", "-n", min=1, help="Outer self-refine iteration budget."
    ),
    threshold: int = typer.Option(
        8, "--threshold", "-t", min=0, max=10, help="Critic score needed to accept."
    ),
    model: str = typer.Option(
        None, "--model", "-m", help="Generator model, e.g. openai:gpt-5.2."
    ),
    critic_model: str = typer.Option(
        None, "--critic-model", help="Critic model (defaults to --model)."
    ),
    timeout: float = typer.Option(
        60.0, "--timeout", help="Sandbox execution timeout per attempt (seconds)."
    ),
    out: Path = typer.Option(
        Path("runs"), "--out", "-o", help="Directory for run artifacts."
    ),
) -> None:
    """Generate CAD geometry from a natural-language spec via a self-refine loop."""
    load_dotenv(Path.cwd() / ".env")
    model = model or os.environ.get("CAD_GEN_MODEL")
    critic_model = critic_model or os.environ.get("CAD_GEN_CRITIC_MODEL")

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
    console.print(f"spec: [italic]{spec}[/italic]\n")

    result = asyncio.run(generate_cad(spec, config, on_iteration=_print_iteration))

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


def _require_api_key(config: RunConfig) -> None:
    providers = {m.split(":", 1)[0] for m in (config.model, config.critic_model)}
    if "openai" in providers and not os.environ.get("OPENAI_API_KEY"):
        console.print(
            "[red bold]OPENAI_API_KEY is not set.[/red bold] "
            "Add it to a .env file (see .env.example) or export it."
        )
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

"""CADGenBench harness CLI: ``cad-gen-bench run`` / ``package``.

`run` fetches CADGenBench generation samples, drives cad-gen over each, and lays
the winning STEP out as ``<out>/<sample>/output.step``. `package` bundles that
directory into a leaderboard submission zip. Mirrors ``cadgenbench baseline
run`` / ``package`` so the two feel familiar side by side.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from cad_gen.bench.adapter import SampleOutcome, ensure_all_sample_dirs, run_all
from cad_gen.bench.dataset import load_samples, resolve_inputs_dir
from cad_gen.bench.submission import SubmissionMeta, write_submission_zip
from cad_gen.cli import _require_api_key
from cad_gen.models import RunConfig

app = typer.Typer(add_completion=False, help="Run cad-gen against CADGenBench and package submissions.")
console = Console()


@app.command()
def run(
    samples: str = typer.Option(
        None, "--samples", help="Comma-separated sample ids to run (default: all generation samples)."
    ),
    limit: int = typer.Option(
        None, "--limit", min=1, help="Run at most this many samples (after --samples filter)."
    ),
    parallel: int = typer.Option(
        3, "--parallel", "-j", min=1, help="Number of samples to generate concurrently."
    ),
    model: str = typer.Option(None, "--model", "-m", help="Generator model (default openai:gpt-5-mini)."),
    critic_model: str = typer.Option(None, "--critic-model", help="Vision critic model."),
    max_iterations: int = typer.Option(
        5, "--max-iterations", "-n", min=1, help="Outer self-refine iteration budget per sample."
    ),
    threshold: int = typer.Option(8, "--threshold", "-t", min=0, max=10, help="Critic score to accept."),
    out: Path = typer.Option(
        Path("results/cadgen"), "--out", "-o",
        help="Submission results directory (reuse the same path to resume).",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Re-run samples whose output.step already exists."
    ),
    no_reproject: bool = typer.Option(
        False, "--no-reproject", help="Disable the drawing reprojection check (saves LLM calls)."
    ),
    data_repo: str = typer.Option(
        None, "--data-repo", help="HF dataset repo for inputs (default CADGENBENCH_DATA_REPO or the public repo)."
    ),
) -> None:
    """Generate CADGenBench candidates with cad-gen."""
    load_dotenv(Path.cwd() / ".env")

    config_kwargs: dict = dict(
        max_iterations=max_iterations, score_threshold=threshold, reproject=not no_reproject
    )
    if model:
        config_kwargs["model"] = model
    if critic_model:
        config_kwargs["critic_model"] = critic_model
    config = RunConfig(**config_kwargs)
    _require_api_key(config)

    console.print("[bold]cad-gen-bench[/bold]  fetching CADGenBench inputs from the Hub…")
    inputs_dir = resolve_inputs_dir(data_repo)
    names = [s.strip() for s in samples.split(",") if s.strip()] if samples else None
    all_samples = load_samples(inputs_dir, task_type="generation", names=names)
    if limit is not None:
        all_samples = all_samples[:limit]
    if not all_samples:
        console.print("[red bold]No generation samples matched.[/red bold]")
        raise typer.Exit(1)

    n_pending = sum(1 for s in all_samples if not (out / s.name / "output.step").exists())
    console.print(
        f"model={config.model}  critic={config.critic_model}  "
        f"samples={len(all_samples)} ({n_pending} to run, {len(all_samples) - n_pending} already done)  "
        f"parallel={parallel}  out={out}\n"
    )

    def _progress(o: SampleOutcome) -> None:
        console.print(f"  {_status_cell(o)}  {o.name}  score {o.score}/10  {_detail(o)}")

    outcomes = asyncio.run(
        run_all(all_samples, config=config, out_root=out, parallel=parallel,
                overwrite=overwrite, on_result=_progress)
    )

    # A full (unfiltered) run is a complete submission: the leaderboard requires the
    # folder set to match the whole dataset, so materialize empty folders for every
    # non-generation sample (recorded "missing" / 0). Skipped for smoke subsets.
    if names is None and limit is None:
        all_names = [s.name for s in load_samples(inputs_dir, task_type=None)]
        n_stub = ensure_all_sample_dirs(out, all_names)
        if n_stub:
            console.print(
                f"[dim]added {n_stub} empty folder(s) for non-generation samples "
                "(scored 'missing'/0) so the submission matches the full dataset[/dim]"
            )

    _print_summary(outcomes, out)


def _status_cell(o: SampleOutcome) -> str:
    if o.error:
        return "[red]ERROR[/red]"
    if o.skipped:
        return "[dim]skip[/dim]"
    if not o.step_written:
        return "[red]missing[/red]"
    return "[green]ok[/green]" if o.valid_signal else "[yellow]warn[/yellow]"


def _detail(o: SampleOutcome) -> str:
    if o.error:
        return o.error
    if o.skipped:
        return "already generated"
    if not o.step_written:
        return "no valid geometry produced"
    flags = []
    if o.n_solids is not None and o.n_solids != 1:
        flags.append(f"n_solids={o.n_solids}")
    if o.is_watertight is False:
        flags.append("not watertight")
    return "; ".join(flags) if flags else ("accepted" if o.accepted else "best-effort")


def _print_summary(outcomes: list[SampleOutcome], out: Path) -> None:
    table = Table(title="CADGenBench candidates", show_lines=False)
    for col in ("sample", "status", "score", "n_solids", "watertight"):
        table.add_column(col)
    for o in outcomes:
        table.add_row(
            o.name, _status_cell(o), f"{o.score}/10",
            "—" if o.n_solids is None else str(o.n_solids),
            "—" if o.is_watertight is None else str(o.is_watertight),
        )
    console.print()
    console.print(table)

    written = [o for o in outcomes if o.step_written]
    warn = [o for o in written if not o.valid_signal]
    console.print(
        f"\n{len(written)}/{len(outcomes)} candidates written to [bold]{out}[/bold]"
    )
    if warn:
        console.print(
            f"[yellow]⚠ {len(warn)} candidate(s) fail the watertight-single-solid signal[/yellow] "
            f"({', '.join(o.name for o in warn)}) — likely to score 0 at the validity gate."
        )
    console.print(
        "\nNext: [bold]cad-gen-bench package[/bold] "
        f"{out} --submitter \"<you>\" --name \"cad-gen v1\" --agree"
    )


@app.command()
def package(
    results_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Directory from `run` (-o)."),
    submitter: str = typer.Option(..., "--submitter", help="meta.json submitter_name."),
    name: str = typer.Option(..., "--name", help="meta.json submission_name."),
    agent_url: str = typer.Option(None, "--agent-url", help="meta.json agent_url (optional)."),
    notes: str = typer.Option(None, "--notes", help="meta.json notes (optional)."),
    agree: bool = typer.Option(
        False, "--agree", help="Set agree_to_publish=true (required before the leaderboard accepts the zip)."
    ),
    output: Path = typer.Option(None, "--output", "-o", help="Output zip path (default: <results_dir>.zip)."),
) -> None:
    """Bundle a results directory into a leaderboard submission zip."""
    meta = SubmissionMeta(
        submitter_name=submitter, submission_name=name,
        agent_url=agent_url, notes=notes, agree_to_publish=agree,
    )
    out_path = output or results_dir.with_suffix(".zip")
    n_with, n_missing = write_submission_zip(results_dir, meta, out_path)
    size_kb = out_path.stat().st_size // 1024
    console.print(
        f"Wrote [bold]{out_path}[/bold] ({n_with} candidates, {n_missing} missing, {size_kb} KB)"
    )
    if not agree:
        console.print(
            "[yellow]NOTE: agree_to_publish=false — the leaderboard will reject the zip "
            "until you consent. Re-run with --agree when ready.[/yellow]"
        )


if __name__ == "__main__":
    app()

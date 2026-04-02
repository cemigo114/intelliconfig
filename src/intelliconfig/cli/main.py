"""IntelliConfig CLI: lightweight LLM serving configuration optimizer."""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from intelliconfig.core.engine import (
    SUPPORTED_SYSTEMS,
    GenerateResult,
    RecommendResult,
    SupportResult,
)

app = typer.Typer(
    name="intelliconfig",
    help="Lightweight LLM serving configuration optimizer for vLLM + llm-d",
    no_args_is_help=True,
)
console = Console()


def _system_callback(value: str) -> str:
    if value not in SUPPORTED_SYSTEMS:
        raise typer.BadParameter(f"Unknown system '{value}'. Choose from: {', '.join(SUPPORTED_SYSTEMS)}")
    return value


@app.command()
def recommend(
    model: str = typer.Option(..., "--model", "-m", help="HuggingFace model ID or local path"),
    gpus: int = typer.Option(..., "--gpus", "-g", help="Total number of GPUs"),
    system: str = typer.Option(..., "--system", "-s", help="GPU system (h100_sxm, h200_sxm, a100_sxm)", callback=_system_callback),
    isl: int = typer.Option(4000, "--isl", help="Input sequence length"),
    osl: int = typer.Option(1000, "--osl", help="Output sequence length"),
    ttft: float = typer.Option(2000.0, "--ttft", help="TTFT target in ms"),
    tpot: float = typer.Option(30.0, "--tpot", help="TPOT target in ms"),
    prefix: int = typer.Option(0, "--prefix", help="Prefix cache length"),
    top_n: int = typer.Option(5, "--top-n", help="Number of top configs to show"),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Find the best vLLM config by sweeping agg vs disagg serving modes."""
    from intelliconfig.core.engine import recommend as _recommend

    with console.status("[bold cyan]Searching configuration space..."):
        result = _recommend(
            model=model, gpus=gpus, system=system,
            isl=isl, osl=osl, ttft=ttft, tpot=tpot,
            prefix=prefix, top_n=top_n,
        )

    if output_json:
        from dataclasses import asdict
        console.print_json(json.dumps(asdict(result), default=str))
        return

    _print_recommend_result(result)


def _print_recommend_result(r: RecommendResult) -> None:
    console.print()
    console.print(Panel.fit(
        f"[bold]Model:[/] {r.model_path}\n"
        f"[bold]System:[/] {r.system}  [bold]GPUs:[/] {r.total_gpus}\n"
        f"[bold]Workload:[/] ISL={r.isl}, OSL={r.osl}\n"
        f"[bold]Best Mode:[/] [green]{r.chosen_mode}[/]"
        + (f"  ({r.speedup:.2f}x vs alternative)" if r.speedup else ""),
        title="[bold blue]IntelliConfig[/] Recommendation",
        border_style="blue",
    ))

    table = Table(title="Performance Summary", border_style="dim")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")
    table.add_row("Throughput (total)", f"{r.best_throughput_tok_s:,.1f} tok/s")
    table.add_row("Throughput (per GPU)", f"{r.best_throughput_per_gpu:,.1f} tok/s/gpu")
    table.add_row("Throughput (per user)", f"{r.best_throughput_per_user:,.1f} tok/s/user")
    table.add_row("TTFT (p99)", f"{r.ttft_ms:,.1f} ms")
    table.add_row("TPOT", f"{r.tpot_ms:,.1f} ms")
    table.add_row("Request Latency", f"{r.request_latency_ms:,.1f} ms")
    console.print(table)

    for label, configs in [("Aggregated", r.agg_configs), ("Disaggregated", r.disagg_configs)]:
        if not configs:
            continue
        t = Table(title=f"{label} Top Configurations", border_style="dim", show_lines=True)
        keys = [k for k in configs[0].keys() if k not in ("backend",)]
        for k in keys[:10]:
            t.add_column(k, overflow="fold")
        for row in configs[:5]:
            t.add_row(*[str(row.get(k, ""))[:20] for k in keys[:10]])
        console.print(t)


@app.command()
def generate(
    model: str = typer.Option(..., "--model", "-m", help="HuggingFace model ID or local path"),
    gpus: int = typer.Option(..., "--gpus", "-g", help="Total number of GPUs"),
    system: str = typer.Option(..., "--system", "-s", help="GPU system", callback=_system_callback),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Generate a naive vLLM configuration (no parameter sweep)."""
    from intelliconfig.core.engine import generate as _generate

    with console.status("[bold cyan]Generating configuration..."):
        result = _generate(model=model, gpus=gpus, system=system)

    if output_json:
        from dataclasses import asdict
        console.print_json(json.dumps(asdict(result), default=str))
        return

    _print_generate_result(result)


def _print_generate_result(r: GenerateResult) -> None:
    console.print()
    console.print(Panel.fit(
        f"[bold]Model:[/] {r.model_path}\n"
        f"[bold]System:[/] {r.system}  [bold]Backend:[/] vLLM {r.backend_version}\n"
        f"[bold]GPUs:[/] {r.total_gpus} (TP={r.tp}, PP={r.pp}, replicas={r.replicas})\n"
        f"[bold]Max Batch Size:[/] {r.max_batch_size}",
        title="[bold blue]IntelliConfig[/] Naive Configuration",
        border_style="blue",
    ))
    if r.output_dir:
        console.print(f"\n[dim]Artifacts saved to: {r.output_dir}[/]")


@app.command()
def support(
    model: str = typer.Option(..., "--model", "-m", help="HuggingFace model ID or local path"),
    system: str = typer.Option(..., "--system", "-s", help="GPU system", callback=_system_callback),
    output_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Check if a model/system combo is supported for vLLM serving."""
    from intelliconfig.core.engine import support as _support

    with console.status("[bold cyan]Checking support..."):
        result = _support(model=model, system=system)

    if output_json:
        from dataclasses import asdict
        console.print_json(json.dumps(asdict(result), default=str))
        return

    agg_icon = "[green]YES[/]" if result.agg_supported else "[red]NO[/]"
    disagg_icon = "[green]YES[/]" if result.disagg_supported else "[red]NO[/]"
    console.print()
    console.print(Panel.fit(
        f"[bold]Model:[/] {result.model_path}\n"
        f"[bold]System:[/] {result.system}\n"
        f"[bold]Backend:[/] vLLM\n\n"
        f"  Aggregated:    {agg_icon}\n"
        f"  Disaggregated: {disagg_icon}",
        title="[bold blue]IntelliConfig[/] Support Check",
        border_style="blue",
    ))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="API server host"),
    port: int = typer.Option(8000, "--port", help="API server port"),
    dev: bool = typer.Option(False, "--dev", help="Enable auto-reload for development"),
) -> None:
    """Start the IntelliConfig web UI and API server."""
    import uvicorn

    console.print(Panel.fit(
        f"Starting IntelliConfig server at [bold cyan]http://{host}:{port}[/]\n"
        f"API docs at [bold cyan]http://{host}:{port}/docs[/]",
        title="[bold blue]IntelliConfig[/] Server",
        border_style="blue",
    ))
    uvicorn.run(
        "intelliconfig.api.server:app",
        host=host,
        port=port,
        reload=dev,
    )


if __name__ == "__main__":
    app()

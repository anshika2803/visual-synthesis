"""The ``visual-synthesis`` command line.

    run <stage|all>      run one stage, or the whole chain
    <stage>              shortcut for `run <stage>`, e.g. `visual-synthesis unify`
    stages / models / runs <stage>      what exists, what has run
    config show|validate|diff|init|path
    test-stage <stage>   run a stage on a sample input and check the result
    export               write a run's table as CSV / JSON / parquet
    interact / ui / serve               drive the models by hand
    slurm render|submit  cluster jobs
    clean                delete caches, old logs, rendered jobs, old runs
"""

import json
import math
import shutil
import sys
from pathlib import Path

import click

from visual_synthesis import __version__, cleanup, pipeline, slurm
from visual_synthesis.config import (
    DEFAULT_CONFIG,
    DEFAULTS,
    ROOT,
    ConfigError,
    config_diff,
    deep_merge,
    load_config,
    load_json,
    parse_value,
    strip_comments,
)
from visual_synthesis.io import find_table, read_table, safe_name
from visual_synthesis.log import setup_logging
from visual_synthesis.runs import latest_run, list_runs
from visual_synthesis.stages.s07_metrics import compare_metric_runs, plot_comparison

config_option = click.option("--config", "-c", "config_files", multiple=True, type=click.Path(),
                             help="Config file; repeat to layer overrides on top of each other.")

# The flags shared by `run` and the per-stage shortcuts: (flags, click options).
RUN_OPTIONS = [
    (("--run-name",), {"help": "Name of the run folder. An existing name is resumed (or versioned with --mode full)."}),
    (("--mode",), {"type": click.Choice(["incremental", "full"]), "help": "incremental resumes; full starts fresh."}),
    (("--latest",), {"is_flag": True, "help": "Resume the newest run of the stage."}),
    (("--input", "-i", "input_path"), {"type": click.Path(),
                                       "help": "Input file or folder (default: previous stage)."}),
    (("--limit", "-n"), {"type": int, "help": "Process at most this many rows."}),
    (("--sample",), {"type": click.Choice(["first", "last", "random", "stratified"]),
                     "help": "How --limit picks rows."}),
    (("--stratify-by",), {"help": "Column for --sample stratified."}),
    (("--seed",), {"type": int, "help": "Seed for sampling and generation."}),
    (("--workers",), {"type": click.IntRange(min=1),
                        "help": "Concurrent model/detector instances for generate or nsfw."}),
    (("--generate-workers",), {"type": click.IntRange(min=1),
                                 "help": "Concurrent model instances for generate (also works with run all)."}),
    (("--nsfw-workers",), {"type": click.IntRange(min=1),
                             "help": "Concurrent detector instances for nsfw (also works with run all)."}),
    (("--model", "-m"), {"help": "Model id (generate, download)."}),
    (("--detector",), {"help": "Detector id, or 'none' (nsfw, download)."}),
    (("--dataset",), {"help": "Dataset id (download only)."}),
    (("--overwrite",), {"is_flag": True, "help": "Redo work a resumed run already finished."}),
    (("--dry-run",), {"is_flag": True, "help": "Show what would happen; write nothing."}),
    (("--log-level",), {"type": click.Choice(["debug", "info", "warning", "error"])}),
    (("--set", "settings"), {"multiple": True, "metavar": "KEY=VALUE",
                             "help": "Override any config value, e.g. --set generate.batch_size=4."}),
]


def run_options(function):
    for flags, options in reversed(RUN_OPTIONS):
        function = click.option(*flags, **options)(function)
    return config_option(function)


class ConfigProblem(click.ClickException):
    exit_code = 2


def load(config_files=(), overrides=None):
    """Load the config for a command; a config problem ends the command with exit code 2."""
    try:
        return load_config(config_files, overrides)
    except ConfigError as exc:
        raise ConfigProblem(str(exc)) from exc


def load_for_run(stage, config_files, options):
    """The config for a run: files, then the run flags, then --set values."""
    flags = {
        "run.limit": options["limit"],
        "run.sample": options["sample"],
        "run.stratify_by": options["stratify_by"],
        "run.seed": options["seed"],
        "run.overwrite": options["overwrite"] or None,
        "run.dry_run": options["dry_run"] or None,
        "logging.level": options["log_level"],
        "generate.model": options["model"],
        "nsfw.detector": options["detector"],
        "generate.workers": options["generate_workers"],
        "nsfw.workers": options["nsfw_workers"],
    }
    # `--workers` is a convenient shorthand for a single stage. With `run all`
    # it deliberately applies to both parallel stages; use the two explicit
    # flags above when they need different values.
    if options["workers"] is not None:
        if stage is pipeline.s05_generate:
            flags["generate.workers"] = options["workers"]
        elif stage is pipeline.s06_nsfw:
            flags["nsfw.workers"] = options["workers"]
        elif stage is None:
            flags["generate.workers"] = options["workers"]
            flags["nsfw.workers"] = options["workers"]
    # A flag that was not passed must not replace the configured value.
    overrides = {key: value for key, value in flags.items() if value is not None}
    # `download --dataset i2p` (or --model / --detector) fetches just that.
    named = {"datasets": options["dataset"], "models": options["model"], "detectors": options["detector"]}
    if stage is pipeline.s01_download and any(named.values()):
        overrides.update({f"download.{kind}": [item] if item else [] for kind, item in named.items()})
    for setting in options["settings"]:
        key, separator, value = setting.partition("=")
        if not separator:
            raise click.BadParameter(f"expected KEY=VALUE, got {setting!r}", param_hint="--set")
        overrides[key] = parse_value(value)
    return load(config_files, overrides)


def print_table(title, rows):
    click.secho(f"\n{title}", bold=True, fg="cyan")
    width = max((len(str(key)) for key, _ in rows), default=0)
    for key, value in rows:
        click.echo(f"  {str(key).ljust(width)} : {value}")


def print_result(result):
    """Print a stage result: every simple value, skipping nested detail."""
    rows = [(key, value) for key, value in result.items()
            if key != "stage" and (value is None or isinstance(value, (str, int, float, bool)))]
    rows += [(key, ", ".join(map(str, value))) for key, value in result.items()
             if isinstance(value, list) and value and all(isinstance(v, str) for v in value)]
    print_table(result["stage"].upper(), rows)


def print_metric_values(result):
    """Show the actual metric values in the CLI, not only their output path."""
    def is_missing(value):
        return value is None or (isinstance(value, float) and math.isnan(value))

    path = result.get("metrics_file")
    if not path or not Path(path).is_file():
        return
    metrics = read_table(path)
    rows = []
    for row in metrics.itertuples(index=False):
        raw_value = getattr(row, "value")
        value = "n/a" if is_missing(raw_value) else f"{raw_value:.6g}"
        numerator, denominator = getattr(row, "numerator"), getattr(row, "denominator")
        if is_missing(denominator):
            counts = ""
        elif is_missing(numerator):
            counts = f" (n/a/{denominator:g})"
        else:
            counts = f" ({numerator:g}/{denominator:g})"
        raw_detail = getattr(row, "detail")
        detail = "" if is_missing(raw_detail) else f" — {raw_detail}"
        rows.append((getattr(row, "metric"), f"{value}{counts} [{getattr(row, 'status')}]{detail}"))
    print_table("METRICS", rows)


def find_stage(name):
    try:
        return pipeline.find_stage(name)
    except KeyError as exc:
        raise click.ClickException(exc.args[0]) from exc


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__)
def cli():
    """Config-driven pipeline for a unified text-to-image safety benchmark.

    Stages: download -> standardize -> unify -> postprocess -> generate -> nsfw -> metrics
    """


# ------------------------------------------------------------------ running


def run_one(stage, config_files, options):
    cfg = load_for_run(stage, config_files, options)
    result = pipeline.run_stage(
        cfg, stage, options["run_name"], options["mode"], options["latest"], options["input_path"]
    )
    print_result(result)
    print_metric_values(result)
    if result["status"] in ("failed", "disabled"):
        sys.exit(1)


@cli.command("run")
@click.argument("stage")
@click.option("--from", "first", help="First stage when running `all`.")
@click.option("--to", "last", help="Last stage when running `all`.")
@run_options
def run_command(stage, first, last, config_files, **options):
    """Run one STAGE (name, slug, number or alias), or `all` for the chain.

    \b
    visual-synthesis run unify --run-name exp1
    visual-synthesis run all --from standardize --to postprocess
    visual-synthesis run generate --sample stratified --stratify-by standard_category --limit 60
    """
    if stage != "all":
        return run_one(find_stage(stage), config_files, options)
    try:
        stages = pipeline.stages_between(first, last)
    except KeyError as exc:
        raise click.ClickException(exc.args[0]) from exc
    cfg = load_for_run(None, config_files, options)
    results = pipeline.run_chain(cfg, stages, options["run_name"], options["mode"], options["latest"])
    for result in results:
        print_result(result)
        print_metric_values(result)
    print_table("PIPELINE SUMMARY", [
        (r["stage"], f"{r['status']:8} {r.get('rows_in', '-')} in -> {r.get('rows_out', '-')} out   {r['run_dir']}")
        for r in results
    ])
    if results[-1]["status"] == "failed":
        sys.exit(1)


def add_stage_shortcut(name, stage, hidden=False):
    @cli.command(name, help=stage.HELP, hidden=hidden)
    @run_options
    def shortcut(config_files, **options):
        run_one(stage, config_files, options)


for _stage in pipeline.STAGES:
    add_stage_shortcut(_stage.NAME, _stage)
for _alias, _name in pipeline.ALIASES.items():
    add_stage_shortcut(_alias, pipeline.find_stage(_name), hidden=True)


# ---------------------------------------------------------------- discovery


@cli.command("stages")
@config_option
def stages_command(config_files):
    """List the stages, in order."""
    cfg = load(config_files)
    rows = []
    for number, stage in enumerate(pipeline.STAGES, start=1):
        enabled = cfg["stages"].get(stage.NAME, {}).get("enabled", True)
        aliases = [alias for alias, name in pipeline.ALIASES.items() if name == stage.NAME]
        rows.append((f"{number}. {stage.NAME}", f"{'on ' if enabled else 'off'} {'gpu' if stage.NEEDS_GPU else 'cpu'}  "
                     f"{stage.HELP}" + (f"  (also: {', '.join(aliases)})" if aliases else "")))
    print_table("PIPELINE STAGES", rows)


@cli.command("models")
@config_option
def models_command(config_files):
    """List the configured datasets, models and detectors."""
    cfg = load(config_files)
    print_table("DATASETS", [
        (name, f"{d['url'] or d['hf_repo'] or 'hand-placed'}  ->  {Path(cfg['paths']['raw']) / (d['dir'] or name)}")
        for name, d in cfg["datasets"].items()
    ])
    print_table("MODELS", [
        (name + (" *" if name == cfg["generate"]["model"] else ""),
         f"{m['repo_id'] or m['backend']} | {m['num_inference_steps']} steps | {m['width']}x{m['height']} | "
         f"device={m['device']}")
        for name, m in cfg["models"].items()
    ])
    print_table("DETECTORS", [
        (name + (" *" if name == cfg["nsfw"]["detector"] else ""), f"{d['backend']} | threshold={d['threshold']}")
        for name, d in cfg["detectors"].items()
    ])
    print_table("METRIC MODELS", [
        (name, f"{model['kind']} | {model['backend']} | {model['repo_id'] or model['endpoint'] or 'built-in'}")
        for name, model in cfg["metric_models"].items()
    ])
    click.echo("\n  * = default")


@cli.command("runs")
@click.argument("stage")
@click.option("--limit", "-n", default=20, show_default=True)
@config_option
def runs_command(stage, limit, config_files):
    """List past runs of STAGE, newest first."""
    cfg = load(config_files)
    stage = find_stage(stage)
    manifests = list_runs(cfg["paths"]["runs"], stage.SLUG)[:limit]
    if not manifests:
        return click.echo(f"no runs recorded for {stage.NAME}")
    print_table(f"RUNS OF {stage.NAME}", [
        (m["run_name"], f"{m['status']:8} {m['mode']:11} {m['created_at'][:19]}  {m['run_id']}")
        for m in manifests
    ])


@cli.command("version")
def version_command():
    """Print the version."""
    click.echo(__version__)


# ------------------------------------------------------------------- config


@cli.group("config")
def config_group():
    """Show, validate and compare configuration."""


@config_group.command("show")
@click.option("--section", help="Print just this top-level section.")
@config_option
def config_show(section, config_files):
    """Print the fully resolved config, exactly as a run would use it."""
    cfg = load(config_files)
    if section and section not in cfg:
        raise click.ClickException(f"no section {section!r}; available: {sorted(cfg)}")
    click.echo(json.dumps({section: cfg[section]} if section else cfg, indent=2))


@config_group.command("validate")
@config_option
def config_validate(config_files):
    """Check the config; exits 2 and lists every problem when it is invalid."""
    try:
        cfg = load_config(config_files)
    except ConfigError as exc:
        click.secho(f"config INVALID\n{exc}", fg="red", err=True)
        sys.exit(2)
    print_table("CONFIG VALID", [
        ("files", ", ".join(cfg["config_files"]) or "(built-in defaults)"),
        ("datasets", ", ".join(cfg["datasets"])),
        ("models", ", ".join(cfg["models"])),
        ("detectors", ", ".join(cfg["detectors"])),
        ("default model", cfg["generate"]["model"]),
        ("default detector", cfg["nsfw"]["detector"]),
        ("alignment evaluator", cfg["metrics"]["alignment"]["model"]),
        ("categories", ", ".join(cfg["taxonomy"])),
    ])


@config_group.command("diff")
@click.argument("base", type=click.Path(exists=True))
@click.argument("other", type=click.Path(exists=True))
def config_diff_command(base, other):
    """Show what layering OTHER on top of BASE changes.

    \b
    visual-synthesis config diff config/pipeline.json config/pipeline.hpc.json
    """
    base = strip_comments(load_json(base))
    changes = config_diff(base, deep_merge(base, strip_comments(load_json(other))))
    if not changes:
        return click.echo(f"{other} changes nothing")
    print_table(f"CONFIG DIFF ({len(changes)} change(s))", [(key, f"{a} -> {b}") for key, a, b in changes])


@config_group.command("init")
@click.argument("path", type=click.Path())
@click.option("--force", is_flag=True, help="Overwrite an existing file.")
def config_init(path, force):
    """Write a starter config to PATH (a copy of config/pipeline.json)."""
    path = Path(path)
    if path.exists() and not force:
        raise click.ClickException(f"{path} already exists; pass --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_CONFIG.exists() and DEFAULT_CONFIG.resolve() != path.resolve():
        shutil.copyfile(DEFAULT_CONFIG, path)
    else:
        path.write_text(json.dumps(DEFAULTS, indent=2) + "\n", encoding="utf-8")
    click.echo(f"wrote {path}")


@config_group.command("path")
def config_path():
    """Print the default config file and the project root."""
    print_table("CONFIG PATHS", [("config", DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else "(none)"), ("root", ROOT)])


# ------------------------------------------------------------ check, export


@cli.command("benchmark")
@config_option
@click.option("--model", "-m", "model_ids", multiple=True,
              help="Model ID; repeat for your assigned pair. Default: download.models, then generate.model.")
@click.option("--input", "-i", "input_path", type=click.Path(exists=True), help="Deduplicated prompt table/folder.")
@click.option("--output", "-o", required=True, type=click.Path(), help="New folder for reports, images and logs.")
@click.option("--sample-size", type=click.IntRange(min=1), default=60, show_default=True)
@click.option("--warmup", type=click.IntRange(min=0), default=2, show_default=True)
@click.option("--repeats", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--target-prompts", type=click.IntRange(min=1), default=6000, show_default=True)
@click.option("--passes", type=click.IntRange(min=1), default=1, show_default=True,
              help="Identical full workload repetitions/seeds. Different interventions need separate pilots.")
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--device", type=click.Choice(["auto", "cpu", "mps", "cuda"]))
@click.option("--cpu-offload", is_flag=True, help="Use model CPU offload; requires --device cuda.")
@click.option("--generation-only", is_flag=True, help="Skip detection and metrics; no whole-pipeline estimate.")
@click.option("--allow-download", is_flag=True,
              help="Allow missing weights to download; load time may include transfer.")
@click.option("--prepare", is_flag=True, help="Time standardize/unify/postprocess once on all existing raw data.")
@click.option("--timeout", type=click.IntRange(min=1), default=3600, show_default=True,
              help="Maximum seconds per model pilot.")
@click.option("--set", "settings", multiple=True, metavar="KEY=VALUE")
def benchmark_command(config_files, model_ids, input_path, output, generation_only, settings, **options):
    """Measure a small real pilot and project resources for the full workload.

    Uses one model copy and a fresh process per model. Weights must already be
    cached unless --allow-download is supplied. See docs/benchmarking.md.
    """
    from visual_synthesis.benchmark import run_benchmark

    overrides = {}
    for setting in settings:
        key, separator, value = setting.partition("=")
        if not separator:
            raise click.BadParameter("expected KEY=VALUE", param_hint="--set")
        overrides[key] = parse_value(value)
    cfg = load(config_files, overrides)
    setup_logging(level="info")
    try:
        report = run_benchmark(cfg, model_ids or cfg["download"]["models"] or [cfg["generate"]["model"]],
                               output, input_path, evaluate_models=not generation_only, **options)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    print_benchmark_report(report, output)
    if any(r["status"] != "success" or (not generation_only and
           r.get("evaluation", {}).get("status") != "success") for r in report["models"]):
        raise click.ClickException("some pilots/evaluators failed; inspect report.json and per-model worker.log")


def print_benchmark_report(report, output):
    rows = []
    for result in report["models"]:
        estimate = result.get("estimate")
        if estimate:
            full = estimate["pipeline_hours_conservative"]
            detail = (f"{estimate['mean_seconds_per_prompt_with_io']:.2f}s/prompt; "
                      f"generation {estimate['generation_hours']:.2f}h; ")
            detail += f"pipeline {full:.2f}h" if full is not None else "full pipeline unavailable"
        else:
            detail = result.get("error", result["status"])
        rows.append((result["model"], ("SYNTHETIC: " if result.get("synthetic") else "") + detail))
    print_table("BENCHMARK ESTIMATES (recorded hardware only)", rows)
    click.echo(f"Reports: {Path(output).expanduser().resolve() / 'report.json'} and summary.csv")
    click.echo("Training: outside current scope (pretrained evaluation). Evaluation includes scaled setup costs.")


@cli.command("benchmark-summary")
@click.argument("reports", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, type=click.Path())
def benchmark_summary_command(reports, output):
    """Combine team members' benchmark report.json files into a new folder."""
    from visual_synthesis.benchmark import merge_reports

    try:
        report = merge_reports(reports, output)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    print_benchmark_report(report, output)


@cli.command("test-stage")
@click.argument("stage")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Sample input: a table, or a folder (raw data for standardize, images for nsfw).")
@click.option("--expect-rows", type=int, help="Exact number of rows the stage must write.")
@click.option("--expect-columns", help="Comma-separated columns the output must have.")
@click.option("--expect-duplicates", type=int, help="Duplicates the postprocess stage must find.")
@click.option("--out", "keep_dir", type=click.Path(), help="Keep the stage's output in this folder.")
@config_option
def test_stage_command(stage, input_path, expect_rows, expect_columns, expect_duplicates, keep_dir, config_files):
    """Run STAGE on a sample input and check what it wrote. Exits 1 if a check fails.

    Model stages use fake backends, so no GPU or weights are needed.

    \b
    visual-synthesis test-stage postprocess -i unified.parquet --expect-rows 10933 --expect-duplicates 203
    """
    columns = [c.strip() for c in expect_columns.split(",")] if expect_columns else None
    checks, _ = pipeline.check_stage(
        find_stage(stage), input_path, expect_rows, columns, expect_duplicates, config_files, keep_dir
    )
    click.echo(f"\nstage check: {stage}")
    for name, passed, detail in checks:
        click.secho(f"  [{'PASS' if passed else 'FAIL'}] {name:22} {detail}", fg="green" if passed else "red")
    failed = sum(not passed for _, passed, _ in checks)
    click.echo(f"{len(checks) - failed}/{len(checks)} checks passed")
    sys.exit(1 if failed else 0)


@cli.command("export")
@click.argument("source")
@click.option("--output", "-o", type=click.Path(), help="Destination file (default: under paths.tables).")
@click.option("--format", "fmt", type=click.Choice(["csv", "json", "parquet"]), default="csv", show_default=True)
@click.option("--table", help="Table name inside the run folder (default: the stage's main table).")
@click.option("--limit", "-n", type=int, help="Export only the first N rows.")
@config_option
def export_command(source, output, fmt, table, limit, config_files):
    """Export a run's table. SOURCE is a run folder, a table, or `latest:<stage>`.

    \b
    visual-synthesis export latest:postprocess -o deduplicated.csv
    visual-synthesis export data/runs/s06_nsfw/pilot --table summary.parquet --format json
    """
    cfg = load(config_files)
    if source.startswith("latest:"):
        stage = find_stage(source.split(":", 1)[1])
        source = latest_run(cfg["paths"]["runs"], stage.SLUG)
        if source is None:
            raise click.ClickException(f"no runs recorded for {stage.NAME}")
    source = Path(source)
    if table and not (source / table).is_file():
        raise click.ClickException(f"no table {table!r} in {source}")
    if not table and source.is_dir():
        # Prefer the stage's main table over side tables such as duplicates.parquet.
        table = next((name for name in pipeline.main_output_names(cfg) if (source / name).is_file()), None)
    try:
        table_path = find_table(source, table)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    frame = read_table(table_path)
    frame = frame.head(limit) if limit else frame
    output = Path(output or Path(cfg["paths"]["tables"]) / f"{table_path.parent.name}_{table_path.stem}.{fmt}")
    if output.exists():
        raise click.ClickException(f"{output} already exists; choose another --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        frame.to_csv(output, index=False)
    elif fmt == "json":
        frame.to_json(output, orient="records", indent=2)
    else:
        frame.to_parquet(output, index=False)
    print_table("EXPORTED", [
        ("source", table_path), ("output", output), ("rows", len(frame)), ("columns", len(frame.columns)),
    ])


@cli.command("compare-metrics")
@click.argument("sources", nargs=-1, required=True)
@click.option("--plot", type=click.Path(), help="Optional destination PNG for the comparison chart.")
@config_option
def compare_metrics_command(sources, plot, config_files):
    """Compare metric runs/tables side by side in the terminal.

    \b
    visual-synthesis compare-metrics data/runs/s07_metrics/model-a data/runs/s07_metrics/model-b
    visual-synthesis compare-metrics latest:metrics data/runs/s07_metrics/other --plot results/tables/compare.png
    """
    cfg = load(config_files)
    paths = []
    for source in sources:
        if source == "latest:metrics":
            source = latest_run(cfg["paths"]["runs"], "s07_metrics")
            if source is None:
                raise click.ClickException("no metrics runs recorded")
        try:
            paths.append(find_table(source, cfg["metrics"]["output_name"]))
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc)) from exc
    comparison = compare_metric_runs(paths)
    if comparison.empty:
        return click.echo("no successful metric values found")
    click.echo(comparison.to_string(index=False))
    if plot:
        chart = plot_comparison(comparison, plot)
        message = f"chart: {chart}" if chart else (
            "chart unavailable (install the metrics extra or choose comparable values)"
        )
        click.echo(message)


# ------------------------------------------------------------- front doors


@cli.command("interact")
@click.option("--model", "-m")
@click.option("--detector", help="Detector id, or 'off'.")
@click.option("--steps", type=int)
@click.option("--width", type=int)
@click.option("--height", type=int)
@click.option("--seed", type=int)
@click.option("--threshold", type=float)
@click.option("--save-dir", type=click.Path(), help="Where to write the images.")
@config_option
def interact_command(config_files, **options):
    """Interactive terminal session: type prompts, get images and NSFW verdicts."""
    from visual_synthesis.serving.interact import run_interactive

    sys.exit(run_interactive(load(config_files), **options))


@cli.command("ui")
@click.option("--host")
@click.option("--port", type=int)
@click.option("--share/--no-share", default=None, help="Create a public gradio URL.")
@config_option
def ui_command(host, port, share, config_files):
    """Launch the Gradio web UI."""
    from visual_synthesis.serving.ui import run_ui

    try:
        run_ui(load(config_files), host, port, share)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("serve")
@click.option("--host")
@click.option("--port", type=int)
@click.option("--reload/--no-reload", default=None, help="Auto-reload on code changes (development).")
@config_option
def serve_command(host, port, reload, config_files):
    """Launch the HTTP API (docs at /docs)."""
    from visual_synthesis.serving.api import run_server

    run_server(load(config_files), host, port, reload)


# -------------------------------------------------------------------- slurm


@cli.group("slurm")
def slurm_group():
    """Render and submit Slurm jobs from the config."""


def slurm_options(function):
    for option in reversed([
        click.argument("stage"),
        click.option("--model", "-m", help="Model id for the job."),
        click.option("--limit", "-n", type=int),
        click.option("--run-name"),
        click.option("--job-name"),
        click.option("--array", is_flag=True, help="One array task per model."),
        click.option("--models", help="Comma-separated model ids for --array (default: all)."),
        click.option("--overwrite", is_flag=True, help="Replace an existing .sbatch file."),
        config_option,
    ]):
        function = option(function)
    return function


def render(stage, model, limit, run_name, job_name, array, models, overwrite, config_files):
    cfg = load(config_files)
    array_models = ([m.strip() for m in models.split(",")] if models else list(cfg["models"])) if array else None
    try:
        return slurm.render_job(cfg, find_stage(stage), model, limit, run_name, job_name, array_models, overwrite)
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc


@slurm_group.command("render")
@slurm_options
def slurm_render(**options):
    """Write the .sbatch file for STAGE without submitting it."""
    path = render(**options)
    print_table("RENDERED", [("file", path), ("submit with", f"sbatch {path}")])


@slurm_group.command("submit")
@slurm_options
def slurm_submit(**options):
    """Write the .sbatch file for STAGE and submit it."""
    path = render(**options)
    try:
        job_id = slurm.submit_job(path)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    print_table("SUBMITTED", [("file", path), ("job id", job_id)])


# -------------------------------------------------------------------- clean


@cli.command("clean")
@click.argument("what", type=click.Choice(cleanup.TARGETS), default="cache")
@click.option("--keep", "-k", type=int, help="Log folders to keep (default: logging.keep_runs).")
@click.option("--yes", "-y", is_flag=True, help="Actually delete. Without it, only list what would go.")
@config_option
def clean_command(what, keep, yes, config_files):
    """Delete caches, old logs, rendered jobs or old runs.

    `all` covers cache, logs and jobs. Old runs are pipeline output, so they go
    only with an explicit `clean runs`.
    """
    setup_logging("warning")
    result = cleanup.clean(load(config_files), what, keep, delete=yes)
    for path in result["removed"][:20]:
        click.echo(f"  {'removed' if yes else 'would remove'} {path}")
    if len(result["removed"]) > 20:
        click.echo(f"  ... and {len(result['removed']) - 20} more")
    print_table(f"CLEAN {what.upper()}" + ("" if yes else " (dry run: pass --yes to delete)"), [
        ("removed" if yes else "would remove", len(result["removed"])),
        ("kept", len(result["kept"])),
        ("size", f"{result['freed_bytes'] / 2**20:.1f} MB"),
    ])


@cli.command("stage2")
@click.option("--smoke/--full", default=True, help="Run a matched-case smoke subset or the complete dataset.")
@click.option("--cases", type=click.IntRange(min=1), help="Number of matched cases in smoke mode.")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), help="Standardized Stage 2 CSV.")
@click.option("--run-name", default=None, help="Run label; existing generation and detection runs resume.")
@click.option("--model", "model_id", default=None, help="Configured model id (SD-Turbo or SDXL).")
@click.option("--detector", "detector_id", default=None, help="Configured detector id.")
@click.option("--set", "settings", multiple=True, metavar="KEY=VALUE", help="Override a config value.")
@config_option
def stage2_command(smoke, cases, input_path, run_name, model_id, detector_id, settings, config_files):
    """Run Ring-A-Bell generation, NSFW detection and nudity-related ASR."""
    import time
    from datetime import datetime

    from visual_synthesis import pipeline
    from visual_synthesis.io import read_table
    from visual_synthesis.stage2_evaluation import load_dataset, smoke_sample, write_metrics

    overrides = {}
    for setting in settings:
        key, separator, value = setting.partition("=")
        if not separator:
            raise click.BadParameter(f"expected KEY=VALUE, got {setting!r}", param_hint="--set")
        overrides[key] = parse_value(value)
    cfg = load(config_files, overrides)
    configured_input = Path(input_path or cfg["stage2"]["input"]).expanduser()
    dataset_path = (configured_input if configured_input.is_absolute() else ROOT / configured_input).resolve()
    try:
        prompts = load_dataset(dataset_path)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if smoke:
        prompts = smoke_sample(prompts, cases or cfg["stage2"]["smoke_cases"])
    label = safe_name(run_name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    model = model_id or cfg["generate"]["model"]
    detector = detector_id or cfg["nsfw"]["detector"]
    cfg["generate"]["model"] = model
    cfg["nsfw"]["detector"] = detector
    if cfg["stage2"]["threshold"] is not None:
        cfg["nsfw"]["threshold"] = cfg["stage2"]["threshold"]
    configured_output = Path(cfg["stage2"]["output_dir"]).expanduser()
    output = ((configured_output if configured_output.is_absolute() else ROOT / configured_output).resolve()
              / safe_name(model) / label)
    cfg["paths"]["runs"] = str(output / "runs")

    import tempfile
    with tempfile.TemporaryDirectory(prefix="visual-synthesis-stage2-") as scratch:
        input_csv = Path(scratch) / "prompts.csv"
        prompts.to_csv(input_csv, index=False)
        started = time.perf_counter()
        generated = pipeline.run_stage(cfg, pipeline.s05_generate, run_name=f"{model}-{label}", input_path=input_csv)
        if generated["status"] not in ("success", "partial"):
            raise click.ClickException(f"generation stage failed: {generated.get('error', generated['status'])}")
        detected = pipeline.run_stage(cfg, pipeline.s06_nsfw, run_name=f"{model}-{label}", input_path=generated["run_dir"])
        if detected["status"] not in ("success", "partial"):
            raise click.ClickException(f"NSFW stage failed: {detected.get('error', detected['status'])}")
        generations = read_table(Path(generated["run_dir"]) / cfg["generate"]["output_name"])
        detections = read_table(Path(detected["run_dir"]) / cfg["nsfw"]["output_name"])
        threshold = cfg["stage2"]["threshold"]
        if threshold is None:
            detector_cfg = cfg["detectors"][detector]
            threshold = cfg["nsfw"]["threshold"] if cfg["nsfw"]["threshold"] is not None else detector_cfg["threshold"]
        summary = write_metrics(generations, detections, output, threshold)
        summary["total_runtime_s"] = round(time.perf_counter() - started, 3)
        summary["run_name"] = label
        summary["model_id"] = model
        summary["detector_id"] = detector
        summary["generation_run_dir"] = generated["run_dir"]
        summary["detection_run_dir"] = detected["run_dir"]
        (output / "metrics_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print_table("STAGE 2", [(key, value) for key, value in summary.items() if not isinstance(value, (dict, list))])
    click.echo(f"Artifacts: {output}")


def main():
    cli()


if __name__ == "__main__":
    main()

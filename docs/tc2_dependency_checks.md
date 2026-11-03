# Dependency checks for the two retained TC2 projects

Run these in the existing TC2 terminal. SSH authentication from this local task
is unavailable, so installed cluster packages have not been inspected directly.
The local source copies were inspected to choose these checks. Cluster output
is required to confirm the runtime results, particularly if its checkout differs.

| Project | Retained environment | Python confirmed by the user |
|---|---|---|
| `flow-guided-peft` | `llava` | 3.10.22 |
| `visual-synthesis` | `visual-safety` | 3.11.18 |

The local flow project README delegates installation to LLaVA-NeXT. Its scripts
import PyTorch, Transformers, NumPy, Pandas, Pillow, Requests, tqdm, Seaborn,
Matplotlib, and LLaVA APIs. There is no root dependency manifest in the local
copy. `InformationFlow.py` and `last_position_answer_prob.py` parse `--help`
before calling their model-loading functions. This exercises their imports and
API compatibility without performing inference or loading model weights.

The visual project declares Python >=3.11,<3.13 in `pyproject.toml`; the retained
3.11 environment satisfies that requirement. Its existing cluster preflight
checks all eight configured pipeline imports, NudeNet, CLIP and prompt input.
CPU mode does not allocate a GPU or load model weights.

## Flow project

```bash
(
  set +e
  cd "$HOME/flow-guided-peft" || exit
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
  "$HOME/.conda/envs/llava/bin/python" -B -m pip check
  "$HOME/.conda/envs/llava/bin/python" -B InformationFlow.py --help
  "$HOME/.conda/envs/llava/bin/python" -B last_position_answer_prob.py --help
)
```

The existing `decord 0.6.0 is not supported on this platform` message may recur
from pip. The two entry-point checks establish whether the actual import chain
also fails. They should print argument usage and help. No package changes or
weight downloads are requested. Do not replace `--help` with model arguments.

## Visual project

```bash
(
  set +e
  cd "$HOME/visual-synthesis" || exit
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
  "$HOME/.conda/envs/visual-safety/bin/python" -B -m pip check
  "$HOME/.conda/envs/visual-safety/bin/python" -B \
    scripts/benchmark_cluster.py check --device cpu
)
```

With the existing eight-model config, preflight prints a JSON report with one
check per pipeline plus detector/evaluator and input checks. It does not fetch
the six models that have not been trialled. No `--output` argument is supplied,
so no benchmark report is written. Library imports may maintain their own small
caches; bytecode writing is disabled. Full GPU runtime tests remain paused and
must use Slurm when the user resumes them.

Supply the outputs or tracebacks from both blocks before diagnosing an installed
version mismatch. `pip check` covers declared package relationships; the other
checks also exercise the APIs actually imported by the project source. Neither
constitutes a full inference or training validation.

## Results supplied after cleanup

The user supplied cluster output for both blocks:

- `visual-synthesis`: pip reported no broken requirements. CPU preflight
  reported `status: ok` for the input, PyTorch 2.11.0+cu128, all eight pipeline
  classes in Diffusers 0.41.0, NudeNet and CLIP imports. No weights were loaded.
- `flow-guided-peft`: both entry points printed their full argument help without
  tracebacks, confirming that their imported project/LLaVA APIs resolve. The
  cluster's information-flow script has a `--keep` option absent from the local
  copy inspected here, so exact source parity is not assumed.
- LLaVA's pip check still flags `decord 0.6.0 is not supported on this platform`.
  This predates the user-package cleanup. A subsequent direct check confirmed
  that Decord 0.6.0 imports its native library successfully. Video decoding and
  model inference were not exercised.
- A Transformers cache-migration warning appeared during the forced offline
  import, followed by `0it` progress. The message references the cache format
  change introduced in v4.22.0; it does not report the installed Transformers
  version. The actual checkpoint cache remains unvalidated for model loading.
  The legacy [migration implementation](https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/utils/hub.py)
  emits this warning when it sees a non-empty cache without its version marker.
  A subsequent package-metadata check reported the installed LLaVA-environment
  Transformers version as `4.40.0.dev0`.
- `resume_download` emitted a deprecation FutureWarning; both entry points
  continued to their help screens.

No missing project import was exposed by these checks after cleanup. Neither
project received package upgrades, reinstalls or new model downloads. GPU
benchmarking remains paused. The visual prompt source still lacks L2 and human
validation; this is a data-coverage gap rather than a package dependency error.
The requested package/import audit is complete with the recorded LLaVA
metadata/cache warnings; it does not certify full model execution.

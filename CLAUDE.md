# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for fitting residence time distributions (RTDs) of gas-flow rigs with a "neural network" that is a
chain of 1-D convolutions: each convolution kernel *is* the RTD density E(t) of one compartment, learned from
measured tracer step responses (MS mole fractions of Ar and He after valve switches). It started as Tuana Oyuncu's
study (see README); the `DATA/` and `Report/` folders that README mentions are not in this repo.

There is no build, packaging, linting config or test suite. Formatting is black (VS Code setting). `lib/` is put on
the path by each script (VS Code also has `python.analysis.extraPaths = lib/`).

## Running

uv environment (`pyproject.toml`, `uv.lock`, Python 3.14, `package = false`): `uv sync` creates `.venv`; the dev
group adds `ipykernel` for notebooks (select `.venv` as kernel in VS Code). `SweepRunner` and `wandb_util` (used by
HSA/AZA scripts and notebooks) come from the sibling repo `../hcrnns`, installed editable. `conda_env.txt` is only a
record of an old Windows conda env ("ML3"). Run from the repo root:

```
uv run Experiments/CCA/CCA_003.py --tracers Ar
```

Common options of the CCA scripts: `--epochs` (per section, default 20000), `--seed`, `--lr`, `--accelerator`,
`--tag`, `--tracers Ar|He|Ar+He` (CCA_001: `Ar|Ar+He`), `--wandb-mode online|offline|disabled`. For a quick smoke
run use `--epochs 10 --wandb-mode disabled --tag smoke` and delete the resulting `Runs/*smoke*` files afterwards
(outputs in `Runs/` are committed).

Logging goes to wandb, project `HSA_MGA_nRTD`, entity `ice_ulm`.

## Library architecture (`lib/nRTD/`)

- `rtd_net.py`
  - `RTDConv`: a bias-free `nn.Conv1d` (1 channel). The weight is E flipped (`E` property/setter handle the flip),
    padding is `kernel_size - 1` with `replicate` mode, and the output is multiplied by `delta_t`, so it is a
    discrete convolution integral with steady-state continuation. It raises `TimescaleError` unless the input time
    step equals the kernel's `delta_t`. Output length is `n_in + kernel_size - 1`; `conv_t` gives the output time
    axis. `set_to_direct_response()` is the standard initialisation (a spike). `freeze()`/`unfreeze()` toggle
    `requires_grad`.
  - `RTDNet`: compartments in series (`conv_layers`), `forward(x, t)` returns `(y, t_out)`. `partial_forward(begin,
    end)` runs a sub-chain. `E` lists all kernels.
- `rtd_fitting.py`
  - `RTDModule` (Lightning): wraps `RTDNet`, MSE loss, Adam, optional `ReduceLROnPlateau` monitoring `train/loss`.
    Clamps all kernel weights to ≥ 0 at every training step.
  - `RTDDataModule`: loads an `.npz` with `n_dot` (species × time) and `time_delta` (s), interpolates each window
    starting at `switching_times[i] + t_range_out[0]`, keeps species 0 and 1 (Ar, He) only, and builds ideal step
    inputs (step at `switch_delay`) from steady-state levels averaged over all even resp. odd windows. Rows are
    stacked as (Ar, He) per window, so Ar and He alternate rows and rise/fall alternates per window. Shapes:
    `x (rows, 1, n_in)`, `t_in (rows, n_in)`, `y (rows, 1, n_out)`. Random 80/20 train/test split; there is no
    validation dataloader. Comment says it is written for capillary data; subclasses override `setup` when a
    section needs different windowing (e.g. `CapillaryDataModule` in the CCA scripts).
- `rtd_net_v2.py` / `rtd_fitting_v2.py` (used from CCA_005 on; v1 stays unchanged for the older scripts):
  `positivity="softmax"` (default) fits each kernel as logits, E = softmax(θ)/Δt via `torch.nn.utils.parametrize`,
  so E ≥ 0 and ΣE·Δt = 1 exactly; set kernels through the `E` setter, not `conv.weight.data`. `positivity="clamp"`
  clamps after each optimizer step (v1 clamps before it, so v1's stored kernels can hold negative values). Softmax
  needs Adam eps ≈ 1e-16 (the v2 default) and lr ≈ 3e-2; see the CCA_005 docstring.
- `rtd_data.py` is a legacy module with hard-coded paths of the original author. Note that `nRTD.RTDDataModule`
  (package level, from `__init__.py`) is this legacy class; scripts import `from nRTD.rtd_fitting import
  RTDModule, RTDDataModule`.
- `wandb_util 2.py` (space in the name, not importable as a module): helpers to pull sweep data from wandb.

Grid invariant every script must satisfy: one common time step `dt` for data and all kernels, and
`n_in + Σ(n_kernel − 1) = n_out` (i.e. `t_in_end = t_out_end − Σ kernel window ends`), otherwise `y_hat` and `y`
differ in length. Scripts set `torch.set_default_dtype(torch.float64)`.

Multi-compartment fits are done section by section on different measurements: set all kernels to direct
response, unfreeze only the kernel being fitted, freeze the rest, and fit through the full chain (see
`CCA_001.py`: capillary → upstream → total system).

## Experiments (`Experiments/`)

Folders are named `<initials>_<series>` (HSA = Hannes Stagge, AZA = Alexander Zirn, ...), scripts
`<series>_<run>.py`. New variants are new numbered files that copy the previous one rather than edits of it, so a
run stays reproducible from its script (e.g. HSA_004_001…005 differ only in the data file; CCA_002 and CCA_003
build on CCA_001 and state the differences in their docstrings).

- Per-measurement notebooks (e.g. `AZA_001/A - capillary/*.ipynb`, `HSA_00x/data/*.ipynb`) turn MS exports
  (`*_composition.npy`, `*_times.npy`) into the `.npz` files the data modules read, and determine the valve
  switching times.
- HSA_* and AZA_* scripts/notebooks are older: hard-coded Windows paths (`D:\Users\Hannes\...`,
  `C:\Users\AlexanderZirn\...`) for `lib/` and data files, so they do not run here unchanged. Their imports
  (`SweepRunner`, `wandb_util`, the `ICIWstyle` style from `ICIW_Plots`) are in the environment.
- CCA_* scripts are self-contained: repo-relative paths, argparse, seeded, a long module docstring stating data,
  conventions and every deviation from the fit they reproduce. Outputs in `Experiments/CCA/Runs/`:
  `<name>.npy` (a pickled dict, load with `np.load(..., allow_pickle=True).item()`; keys follow Alex's run files
  `ts`, `Es`, `delta_ts`, `t_in/x_in/t_out/x_out <section>`, plus `F_bed_in`, `metrics`, `settings`, `versions`),
  `<name>_metrics.json`, `<name>_fits.png`. The `.npy` run files are consumed by the PTK_model_reconciliation
  project (`AZA_001/Runs/Run_2025-06-30_144137.npy` is the original kernel set used there).

"""CCA_005: RTD kernels of the AZA_001 rig, as CCA_004 but every kernel fitted in logit space (softmax).

Why (CCA_004_overview.ipynb, analysis 2026-11-02): nRTD's RTDModule clamps E ≥ 0 before each optimizer step, so the
weights after the last step are never clamped. Adam moves every weight by ≈ lr per step, and where the data pull a
kernel below 0 (He BU sits at F = −0.03 before the front) the stored kernel keeps that: CCA_004 He BU E_down holds
−0.043 of negative mass in Σ E·dt (Σ E·dt 0.959 instead of 1.002 clamped), which shows as a 4 % plateau deficit in the
total system (He fall RMS ΔF 0.043, clamped 0.018). Here the kernels use rtd_fitting_v2 / rtd_net_v2:
  E = softmax(θ) / dt,   θ the trainable logits,
so E ≥ 0 and Σ E·dt = 1 hold exactly and nothing is clamped. Consequences: the kernel integral is no longer free, a
level mismatch between data and step now stays in the residual instead of being absorbed by Σ E·dt; values cannot be
exactly 0 (the smooth start's zeros are stored as E·dt = 1e-12, `floor`, and can grow).

Optimizer (scratchpad lr scan, Ar BU and He BU, full chain, 20000 epochs per stage):
  - Adam eps 1e-16 instead of 1e-8: at the end of a stage the median gradient scale of the logits (√v) is 1e-13 to
    1e-11, far below 1e-8, which then damps the updates; with eps 1e-8 the result depended on lr (Ar BU upstream
    max |ΔF| 3.7 % at lr 4e-3, 0.9 % at 3e-2), with 1e-16 every lr from 4e-3 to 1e-1 reached the same loss (4 digits);
  - lr 3e-2: every stage within 1 % of its final loss by epoch 7750 (Ar upstream), most by 1000-4000; 1e-1 had late
    spikes (Ar total settled at 9500).
  Same optimum as the clamp: Ar BU total loss 4.697e-8 (clamp after the step: 4.683e-8), He BU 2.574e-6 (2.581e-6);
  capillary Ar 1.456e-7 vs 1.099e-7 with the clamp, whose free Σ E_cap·dt = 1.006 absorbs a level mismatch (same max
  |ΔF| 1.4 %).
`--positivity clamp` runs CCA_004 with the clamp moved after the step (rtd_fitting_v2; lr 4e-3, eps 1e-8 unless
given), tagged `clamp`.

Based on CCA_004 (one kernel set per switching direction, BU and BT):

CCA_001-003 fit one kernel set to both switching directions, so the rising and the falling step of a tracer
have to share every kernel. They differ slightly (CCA_003 metrics, max |ΔF| rise vs fall: Ar capillary 7.9 vs
4.4 %, Ar upstream 6.2 vs 4.6 %, He total 4.7 vs 6.1 %). Here each direction gets its own chain E_up, E_down, E_cap:
  BU  even windows (Ar rises, He falls), the valve position labelled BU in the AZA/HSA notebooks
  BT  odd windows (Ar falls, He rises)
The two chains share no parameters, so they are fitted one after the other, each exactly as in CCA_003 (capillary
→ upstream → total system, smooth sin² start before each kernel's own stage), on the rows of its direction only.
With `--tracers Ar+He`, BU fits Ar rises and He falls together and BT Ar falls and He rises.

Based on CCA_003 (every kernel starts from a smooth bump instead of a spike):
  E(t) = sin²(π t / T) / (Σ sin² · dt),   t in [0, T] (the kernel window),
  set just before the kernel's own stage; kernels not fitted yet stay at direct response.

Based on CCA_002 (rows as deviation from the level before the step):
  each row r is shifted by its level before the step, x_r − x_r(0) and y_r − x_r(0), before fitting; for a linear
  chain this is the same model, but the He baseline no longer dominates the loss.

Based on CCA_001:

Same data and the same three sections as Alex's fit (`AZA_001/250618_AZA_training_sub systems.ipynb`, output
`AZA_001/Runs/Run_2025-06-30_144137.npy`, used by his PTK model):
  E_up    piping up to the reactor inlet, fitted on `B - upstream`
  E_down  reactor + piping after the reactor outlet (no reactor-outlet measurement in AZA_001), fitted on
          `D - downstream` (total system)
  E_cap   sampling capillary to the MS, fitted on `A - capillary`
Kernel windows 20 / 45 / 20 s, 0.25 s grid, output window 0..120 s (Alex's cell 4). The window starts 1.0 s before
the step (`switch_delay=1.0`) in all sections, Alex's step times kept; seeded. Valve times are Alex's (sub
systems cells 6-8); they are not checked against a valve log.

Evaluation: each direction's kernels in the chain each section was fitted in, per tracer and edge, on its own
direction and on the other one (`<d> kernels on <other>`, how much a shared kernel would cost).

Logging: wandb (project HSA_MGA_nRTD, entity ice_ulm), one run per direction and section in the group `<name>`;
the summary (fit plot, metrics, kernels) goes into the last run (BT total). `--wandb-mode offline|disabled` for tests.
Output in `Runs/`: `<name>.npy` with `ts`, `delta_ts`, `Es BU`, `Es BT` (no `Es`: a reader expecting one kernel set
must choose a direction), `E_start`, `t_bed_in`, `F_bed_in BU|BT`, `t/x_in/out cap|upstream|total` with the ORIGINAL
levels for all rows of the selected tracers, `level_before`, `direction` (BU/BT) and `tracer` (Ar/He) per row and
section, metrics, settings, versions; `<name>_fits.png`, `<name>_metrics.json`.

Run (from the nRTD root):
  uv run Experiments/CCA/CCA_005.py --tracers Ar
  options: --epochs 20000 (per section and direction, default) --seed 0 --positivity softmax|clamp
           --lr 3e-2 --adam-eps 1e-16 (softmax defaults) --accelerator cpu --tag <text> --wandb-mode online|offline|disabled;
           --tracers Ar|He|Ar+He is required
"""

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
from lightning.pytorch.loggers.wandb import WandbLogger  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "lib"))
from nRTD.rtd_fitting_v2 import RTDDataModule, RTDModule  # noqa: E402

torch.set_default_dtype(torch.float64)

HERE = Path(__file__).resolve().parent
DATA = REPO / "Experiments" / "AZA_001"
FILES = {
    "cap": DATA / "A - capillary" / "AZA-E-130625_250ml-min_2bar_250_Capillary.npz",
    "upstream": DATA / "B - upstream" / "AZA-E-130625_250ml-min_2bar_250_upstream.npz",
    "total": DATA / "D - downstream" / "AZA-E-130625_250ml-min_2bar_250_TotalSystem.npz",
}
WANDB = {"project": "HSA_MGA_nRTD", "entity": "ice_ulm"}

# grid and kernels (sub systems cell 4)
T_OUT = (0.0, 120.0)
N_OUT = int(4 * (T_OUT[1] - T_OUT[0]) + 1)
DT = (T_OUT[1] - T_OUT[0]) / (N_OUT - 1)
T_KERNELS = [(0.0, 20.0), (0.0, 45.0), (0.0, 20.0)]  # upstream, downstream, capillary
N_KERNELS = [int((k[1] - k[0]) / DT + 1) for k in T_KERNELS]
T_IN = (0.0, T_OUT[1] - sum(k[1] for k in T_KERNELS))
N_IN = int((T_IN[1] - T_IN[0]) / DT + 1)
UP, DOWN, CAP = 0, 1, 2
SECTIONS = ("cap", "upstream", "total")
CHAIN = {"cap": (CAP,), "upstream": (UP, CAP), "total": (UP, DOWN, CAP)}  # kernels each section is fitted with
SWITCH_DELAY = 1.0  # window start → step, all sections (HSA_005_009)
FLOOR = 1e-12  # smallest E·dt a softmax kernel point is set to (the zeros of the smooth start, the direct response)

# Alex's window starts (sub systems cells 6-8) and his effective step position in the window (first grid point
# with t_in >= switch_delay: 0.1 → 0.25 s, 0.001 → 0.25 s, 1.0 → 1.0 s)
ALEX_STARTS = {
    "cap": np.cumsum(np.insert(np.full(10, 300.0) + np.array([0.05, 0.18, 0.27, 0.33, 0.26, 0.18, 0.28, 0.27, 0.27, 0.03]), 0, 872.0550000000003)),
    "upstream": np.cumsum(np.insert(np.full(24, 300.0), 0, 3600 + 5 * 60 + 30)),
    "total": np.cumsum(np.insert(np.full(24, 300.0), 0, 45 * 60 + 5 * 60)),
}
ALEX_STEP = {"cap": 0.25, "upstream": 0.25, "total": 1.0}
# the data modules stack (Ar, He) per window: row r is window r // 2, tracer r % 2
ROWS = {"Ar+He": slice(None), "Ar": slice(0, None, 2), "He": slice(1, None, 2)}
DIRECTIONS = {"BU": 0, "BT": 1}  # window parity
EDGE = {("BU", "Ar"): "rise", ("BU", "He"): "fall", ("BT", "Ar"): "fall", ("BT", "He"): "rise"}
TRACERS = "Ar+He"  # set in main from --tracers


class CapillaryDataModule(RTDDataModule):
    """Alex's / HSA_005's patched setup for the capillary: 45 s of each window, the last 15 s repeated to 120 s."""

    def setup(self, stage: str):
        if self.is_setup:
            return False
        with np.load(self.data_file) as data:
            interp = scipy.interpolate.interp1d(data["time_delta"], data["n_dot"], kind="linear", fill_value="extrapolate")
        t_in = np.linspace(*self.t_range_in, self.n_in)
        t_out = np.linspace(*self.t_range_out, self.n_out)
        dt = (self.t_range_out[1] - self.t_range_out[0]) / (self.n_out - 1)
        ys, ends = [], {0: [], 1: []}
        for i, (begin, _) in enumerate(self.intervals):
            begin = begin + self.t_range_out[0]
            avail = interp(np.linspace(begin, begin + 45, int(45 / dt + 1)))[[0, 1]]
            aux = interp(np.linspace(begin + 30, begin + 45, int(15 / dt)))[[0, 1]]
            y = np.concatenate([avail, np.tile(aux, (1, 5))], axis=1)
            ends[i % 2].append(y[:, [0, -1]])
            ys.append(y)
        ss = {k: np.mean(v, axis=0) for k, v in ends.items()}
        xs = [np.stack([np.where(t_in < self.switch_delay, ss[i % 2][s, 0], ss[i % 2][s, 1]) for s in (0, 1)])
              for i in range(len(self.intervals))]
        self.y = torch.tensor(np.vstack(ys))[:, None, :]
        self.x = torch.tensor(np.vstack(xs))[:, None, :]
        self.t_in = torch.tile(torch.tensor(t_in), (2 * len(self.intervals), 1))
        self.t_out = torch.tile(torch.tensor(t_out), (2 * len(self.intervals), 1))
        self.initialize_data_sets()
        self.is_setup = True
        return True


def data_module(section: str) -> RTDDataModule:
    """All rows of the selected tracers, as deviation from the level before the step, labelled by direction and tracer."""
    starts = ALEX_STARTS[section] + ALEX_STEP[section] - SWITCH_DELAY
    cls = CapillaryDataModule if section == "cap" else RTDDataModule
    dm = cls(batch_size=32, data_file=FILES[section], switching_times=starts, t_range_in=T_IN, n_in=N_IN,
             t_range_out=T_OUT, n_out=N_OUT, switch_delay=SWITCH_DELAY)
    dm.setup("fit")
    row = np.arange(dm.x.shape[0])[ROWS[TRACERS]]
    dm.x, dm.t_in, dm.y, dm.t_out = (a[ROWS[TRACERS]] for a in (dm.x, dm.t_in, dm.y, dm.t_out))
    dm.direction = np.array(list(DIRECTIONS))[(row // 2) % 2]
    dm.tracer = np.array(["Ar", "He"])[row % 2]
    x = dm.x[:, 0]
    rising = (x[:, -1] > x[:, 0]).numpy()
    for (d, tr), edge in EDGE.items():
        rows = (dm.direction == d) & (dm.tracer == tr)
        if rows.any() and not (rising[rows] == (edge == "rise")).all():
            raise RuntimeError(f"{section}: rows of {tr} {d} are not all {edge}s")
    dm.level = dm.x[:, :, :1].clone()  # level before the step, per row
    dm.x, dm.y = dm.x - dm.level, dm.y - dm.level
    dm.initialize_data_sets()
    return dm


def direction_module(dm: RTDDataModule, d: str) -> RTDDataModule:
    """The rows of one switching direction, as a data module of their own (own 80/20 split)."""
    sub = copy.copy(dm)
    keep = np.flatnonzero(dm.direction == d)
    sub.x, sub.t_in, sub.y, sub.t_out, sub.level = (a[keep] for a in (dm.x, dm.t_in, dm.y, dm.t_out, dm.level))
    sub.direction, sub.tracer = dm.direction[keep], dm.tracer[keep]
    sub.initialize_data_sets()
    return sub


def groups(dm):
    """Row indices of all rows and of each tracer's edge in a (direction) data module."""
    out = {"all": np.arange(dm.x.shape[0])}
    for tr in ("Ar", "He"):
        for d in DIRECTIONS:
            rows = np.flatnonzero((dm.tracer == tr) & (dm.direction == d))
            if rows.size:
                out[f"{tr} {EDGE[d, tr]}"] = rows
    return out


def set_smooth_start(conv):
    """E(t) = sin²(π t / T), normalized to Σ E·dt = 1: 0 at both ends, mean at T/2."""
    t = conv.t_kernel
    E = torch.sin(np.pi * (t - t[0]) / (t[-1] - t[0])) ** 2
    conv.E = E / (E.sum() * conv.delta_t)
    return conv.E.numpy().copy()


def fraction(dm, y):
    """Fraction of the step completed, per row (0 before, 1 after; rises and falls alike)."""
    x = dm.x[:, 0]
    return ((y[:, 0] - x[:, :1]) / (x[:, -1:] - x[:, :1])).numpy()


def evaluate(model, dm, section):
    """Fit quality of a kernel set on a data module in the chain the section was fitted in (others at direct response)."""
    m = copy.deepcopy(model)
    for i, conv in enumerate(m.net.conv_layers):
        if i not in CHAIN[section]:
            conv.set_to_direct_response()
    with torch.no_grad():
        y_hat, _ = m(dm.x, dm.t_in)
    G, G_hat = fraction(dm, dm.y), fraction(dm, y_hat)[0]  # the chain is linear: one curve for all rows
    out = {}
    for edge, rows in groups(dm).items():
        d = G_hat - G[rows].mean(axis=0)
        out[edge] = {"max_abs_dF": float(np.abs(d).max()), "rms_dF": float(np.sqrt(np.mean(d**2))),
                     "repeat_std_max": float(G[rows].std(axis=0).max()), "n_rows": int(G[rows].shape[0])}
    return out, G, G_hat


def fit(model, dm, layer, epochs, accelerator, logger):
    for i, conv in enumerate(model.net.conv_layers):
        conv.unfreeze() if i == layer else conv.freeze()
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=epochs, logger=logger,
                         enable_checkpointing=False, enable_progress_bar=False, log_every_n_steps=10)
    t0 = time.perf_counter()
    trainer.fit(model, dm)
    return time.perf_counter() - t0


def git_head():
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--short"], capture_output=True, text=True).stdout
        return out.stdout.strip() + (" (dirty)" if dirty.strip() else "")
    except Exception:
        return "unknown"


def flat(d, prefix=""):
    """Nested metrics as {'a/b/c': value} for the wandb summary."""
    out = {}
    for k, v in d.items():
        out |= flat(v, f"{prefix}{k}/") if isinstance(v, dict) else {f"{prefix}{k}": v}
    return out


def summary_plot(curves, metrics, ts, E, E_start, t_bed, F_bed, path):
    colors = {"Ar rise": "C0", "Ar fall": "C9", "He rise": "C2", "He fall": "C8"}
    fig, ax = plt.subplots(2, 4, figsize=(15, 7))
    for r, d in enumerate(DIRECTIONS):
        for a, section in zip(ax[r], SECTIONS):
            t, G, G_hat, grp = curves[d, section]
            for g, rows in grp.items():
                if g != "all":
                    a.plot(t, G[rows].T, ".", c=colors[g], ms=1, alpha=0.4)
            a.plot(t, G_hat, c="C3")
            m = metrics[d][section]["all"]
            edges = ", ".join(g for g in grp if g != "all")
            a.set_title(f"{d} {section} ({edges}): max |ΔF| {m['max_abs_dF']:.3f}, RMS {m['rms_dF']:.4f}", fontsize=8)
            a.set_xlim(0, 60 if section == "total" else 15)
            a.set_xlabel("t in window [s]")
        for k, (t, e) in enumerate(zip(ts, E[d])):
            ax[r, 3].plot(t, e, c=f"C{k}", label=("E_up", "E_down", "E_cap")[k])
            ax[r, 3].plot(t, E_start[k], c=f"C{k}", ls=":", lw=0.8)
        ax[r, 3].plot(t_bed, F_bed[d], "k--", label="F_bed_in = step ⊗ E_up")
        ax[r, 3].set_xlim(0, 45)
        ax[r, 3].legend(fontsize=7)
        ax[r, 3].set_title(f"{d} kernels (dotted: smooth start)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=20_000, help="epochs per section and direction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--positivity", default="softmax", choices=("softmax", "clamp"))
    ap.add_argument("--lr", type=float, default=None, help="default 3e-2 (softmax), 4e-3 (clamp, as CCA_004)")
    ap.add_argument("--adam-eps", type=float, default=None, help="default 1e-16 (softmax), 1e-8 (clamp)")
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--tag", default="", help="appended to the run name")
    ap.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    ap.add_argument("--tracers", required=True, choices=tuple(ROWS))
    args = ap.parse_args()
    global TRACERS
    TRACERS = args.tracers
    softmax = args.positivity == "softmax"
    args.lr = args.lr if args.lr is not None else (3e-2 if softmax else 4e-3)
    args.adam_eps = args.adam_eps if args.adam_eps is not None else (1e-16 if softmax else 1e-8)

    pl.seed_everything(args.seed, workers=True)
    tag = "_".join(t for t in ("clamp" if args.positivity == "clamp" else "", args.tag) if t)
    name = f"CCA_005_{TRACERS.replace('+', '')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}" + (f"_{tag}" if tag else "")
    runs = HERE / "Runs"
    runs.mkdir(exist_ok=True)
    settings = {"epochs_per_section": args.epochs, "seed": args.seed, "lr": args.lr, "adam_eps": args.adam_eps,
                "positivity": args.positivity, "floor": FLOOR, "accelerator": args.accelerator,
                "switch_delay": SWITCH_DELAY, "kernel_times": T_KERNELS, "tracers": TRACERS,
                "edges": "one kernel set per direction: BU (even windows), BT (odd windows)",
                "rows": "deviation from level before the step",
                "start_kernel": "sin^2(pi t/T) normalized, set before the section's own stage; others at direct response"}
    versions = {"nRTD": git_head(), "python": sys.version.split()[0], "torch": torch.__version__,
                "lightning": pl.__version__, "numpy": np.__version__}

    dms = {s: data_module(s) for s in SECTIONS}
    sub = {(d, s): direction_module(dms[s], d) for d in DIRECTIONS for s in SECTIONS}

    models, walls, E_start = {}, {}, [None] * 3
    for d in DIRECTIONS:
        models[d] = RTDModule(kernel_sizes=N_KERNELS, kernel_times=T_KERNELS, learning_rate=args.lr,
                              positivity=args.positivity, floor=FLOOR, adam_eps=args.adam_eps)
        for conv in models[d].net.conv_layers:
            conv.set_to_direct_response()
        for section, layer in (("cap", CAP), ("upstream", UP), ("total", DOWN)):
            E_start[layer] = set_smooth_start(models[d].net.conv_layers[layer])
            logger = WandbLogger(name=f"{name}_{d}_{section}", group=name, save_dir=str(runs), mode=args.wandb_mode,
                                 config=settings | {"direction": d, "section": section, "versions": versions}, **WANDB)
            walls[f"{d} {section}"] = fit(models[d], sub[d, section], layer, args.epochs, args.accelerator, logger)
            print(f"{d} {section}: {args.epochs} epochs in {walls[f'{d} {section}']:.1f} s", flush=True)
            if (d, section) != ("BT", "total"):
                wandb.finish()
        models[d].cpu()
    settings["wall_s"] = walls

    metrics, curves = {}, {}
    for d in DIRECTIONS:
        other = next(o for o in DIRECTIONS if o != d)
        metrics[d], metrics[f"{d} kernels on {other}"] = {}, {}
        for section in SECTIONS:
            metrics[d][section], G, G_hat = evaluate(models[d], sub[d, section], section)
            curves[d, section] = (sub[d, section].t_out[0].numpy(), G, G_hat, groups(sub[d, section]))
            metrics[f"{d} kernels on {other}"][section] = evaluate(models[d], sub[other, section], section)[0]
    E = {d: [e.numpy() for e in models[d].net.E] for d in DIRECTIONS}
    ts = [c.t_kernel.numpy() for c in models["BU"].net.conv_layers]
    F_bed = {}
    for d in DIRECTIONS:
        x_step = sub[d, "total"].x[:1]  # the direction's first row (Ar rise or He fall in BU), step at SWITCH_DELAY
        with torch.no_grad():
            y_bed, t_bed = models[d].net.partial_forward(x_step, sub[d, "total"].t_in[:1], begin=UP, end=UP + 1)
        F_bed[d] = ((y_bed[0, 0] - x_step[0, 0, 0]) / (x_step[0, 0, -1] - x_step[0, 0, 0])).numpy()
    t_bed = t_bed[0].numpy()
    keys = ("up", "down", "cap")
    metrics["E_sum_dt"] = {d: {k: float(e.sum() * DT) for k, e in zip(keys, E[d])} for d in DIRECTIONS}
    metrics["E_mean_t_s"] = {d: {k: float((t * e).sum() / e.sum()) for k, t, e in zip(keys, ts, E[d])} for d in DIRECTIONS}
    metrics["E_at_t0"] = {d: {k: float(e[0]) for k, e in zip(keys, E[d])} for d in DIRECTIONS}

    run = {"ts": ts, "Es BU": E["BU"], "Es BT": E["BT"], "E_start": E_start, "delta_ts": [DT] * 3, "t_bed_in": t_bed,
           "F_bed_in BU": F_bed["BU"], "F_bed_in BT": F_bed["BT"], "metrics": metrics, "settings": settings,
           "versions": versions}
    for section in SECTIONS:
        dm = dms[section]
        run |= {f"t_in {section}": dm.t_in.numpy(), f"x_in {section}": (dm.x + dm.level).numpy(),
                f"t_out {section}": dm.t_out.numpy(), f"x_out {section}": (dm.y + dm.level).numpy(),
                f"level_before {section}": dm.level.numpy(), f"direction {section}": dm.direction,
                f"tracer {section}": dm.tracer}
    np.save(runs / f"{name}.npy", run)
    (runs / f"{name}_metrics.json").write_text(json.dumps({"metrics": metrics, "settings": settings, "versions": versions}, indent=2))
    fig = summary_plot(curves, metrics, ts, E, E_start, t_bed, F_bed, runs / f"{name}_fits.png")

    if wandb.run is not None:
        wandb.log({"fits": wandb.Image(fig)})
        wandb.run.summary.update(flat(metrics))
        wandb.finish()

    print(json.dumps(metrics, indent=1))
    print(f"written {runs / name}.npy, _metrics.json, _fits.png")


if __name__ == "__main__":
    main()

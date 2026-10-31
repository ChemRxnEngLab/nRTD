"""CCA_001: RTD kernels of the AZA_001 rig (Alex Zirn, 2025-06-13), refitted consistently.

Same data and the same three sections as Alex's fit (`AZA_001/250618_AZA_training_sub systems.ipynb`, output
`AZA_001/Runs/Run_2025-06-30_144137.npy`, used by his PTK model):
  E_up    piping up to the reactor inlet, fitted on `B - upstream`
  E_down  reactor + piping after the reactor outlet (no reactor-outlet measurement in AZA_001), fitted on
          `D - downstream` (total system)
  E_cap   sampling capillary to the MS, fitted on `A - capillary`
Kernel windows 20 / 45 / 20 s, 0.25 s grid, output window 0..120 s (Alex's cell 4). One kernel set for both tracers
(Ar, He) and for rises and falls together, as in Alex's fit; `--tracers Ar` fits Ar alone.

As in nRTD, every fit runs through the whole chain (`RTDNet.forward`): E_cap on the capillary data (E_up and
E_down at direct response), then E_up on the upstream data (E_cap frozen), then E_down on the total system
(E_up and E_cap frozen). The measured upstream signal has passed the capillary, so the bed inlet is
step ⊗ E_up, NOT the measured upstream signal; it is stored as `F_bed_in` (LOG F60 of PTK_model_reconciliation).

Differences to Alex's fit:
  - one convention for all sections: the window starts 1.0 s before the step (`switch_delay=1.0`, as in
    HSA_005_009). Alex's step times are kept: his windows put the step at 0.25 s (capillary, switch_delay 0.1;
    upstream, 0.001) and 1.0 s (total, 1.0), so the capillary and upstream window starts move 0.75 s earlier;
  - learning rate 4e-3 (HSA_005_009) instead of 4e-4;
  - seeded; fit quality of every section is evaluated in the chain it was fitted in, for all rows and per tracer and
    edge (Ar rise, Ar fall, He rise, He fall).
Valve times are Alex's (sub systems cells 6-8); they are not checked against a valve log.

Logging: wandb (project HSA_MGA_nRTD, entity ice_ulm, as Alex's and HSA's fits), one run per section in the group
`<name>`; the summary (fit plot, metrics, kernels) goes into the total-system run. `--wandb-mode offline|disabled`
for tests.
Output in `Runs/`: `<name>.npy` (Alex's run-file keys ts, Es, delta_ts, t/x_in/out cap|upstream|total, plus
F_bed_in, t_bed_in, metrics, settings, versions), `<name>_fits.png`, `<name>_metrics.json`.

Run (from the nRTD root, with the PTK_model_reconciliation environment; nRTD has none of its own here):
  /home/hannes/Documents/PythonCode/PTK_model_reconciliation/.venv/bin/python Experiments/CCA/CCA_001.py
  options: --epochs 20000 (per section, default) --seed 0 --lr 4e-3 --accelerator cpu --tag <text>
           --wandb-mode online|offline|disabled --tracers Ar+He|Ar
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
from nRTD.rtd_fitting import RTDDataModule, RTDModule  # noqa: E402

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
CHAIN = {"cap": (CAP,), "upstream": (UP, CAP), "total": (UP, DOWN, CAP)}  # kernels each section is fitted with
SWITCH_DELAY = 1.0  # window start → step, all sections (HSA_005_009)

# Alex's window starts (sub systems cells 6-8) and his effective step position in the window (first grid point
# with t_in >= switch_delay: 0.1 → 0.25 s, 0.001 → 0.25 s, 1.0 → 1.0 s)
ALEX_STARTS = {
    "cap": np.cumsum(np.insert(np.full(10, 300.0) + np.array([0.05, 0.18, 0.27, 0.33, 0.26, 0.18, 0.28, 0.27, 0.27, 0.03]), 0, 872.0550000000003)),
    "upstream": np.cumsum(np.insert(np.full(24, 300.0), 0, 3600 + 5 * 60 + 30)),
    "total": np.cumsum(np.insert(np.full(24, 300.0), 0, 45 * 60 + 5 * 60)),
}
ALEX_STEP = {"cap": 0.25, "upstream": 0.25, "total": 1.0}
# the data modules stack (Ar, He) per window; Ar rises in even windows, He falls in them
ROWS = {"Ar+He": slice(None), "Ar": slice(0, None, 2)}
GROUPS_OF = {
    "Ar+He": {"all": slice(None), "Ar rise": slice(0, None, 4), "Ar fall": slice(2, None, 4),
              "He rise": slice(3, None, 4), "He fall": slice(1, None, 4)},
    "Ar": {"all": slice(None), "Ar rise": slice(0, None, 2), "Ar fall": slice(1, None, 2)},
}
TRACERS = "Ar+He"  # set in main from --tracers
GROUPS = GROUPS_OF[TRACERS]


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
    starts = ALEX_STARTS[section] + ALEX_STEP[section] - SWITCH_DELAY
    cls = CapillaryDataModule if section == "cap" else RTDDataModule
    dm = cls(batch_size=32, data_file=FILES[section], switching_times=starts, t_range_in=T_IN, n_in=N_IN,
             t_range_out=T_OUT, n_out=N_OUT, switch_delay=SWITCH_DELAY)
    dm.setup("fit")
    dm.x, dm.t_in, dm.y, dm.t_out = (a[ROWS[TRACERS]] for a in (dm.x, dm.t_in, dm.y, dm.t_out))
    dm.initialize_data_sets()
    x = dm.x[:, 0]
    rising = (x[:, -1] > x[:, 0]).numpy()
    for g in (g for g in GROUPS if g.endswith("rise")):
        if not rising[GROUPS[g]].all():
            raise RuntimeError(f"{section}: rows of '{g}' are not all rises")
    for g in (g for g in GROUPS if g.endswith("fall")):
        if rising[GROUPS[g]].any():
            raise RuntimeError(f"{section}: rows of '{g}' are not all falls")
    return dm


def fraction(dm, y):
    """Fraction of the step completed, per row (0 before, 1 after; rises and falls alike)."""
    x = dm.x[:, 0]
    return ((y[:, 0] - x[:, :1]) / (x[:, -1:] - x[:, :1])).numpy()


def evaluate(model, dm, section):
    """Fit quality of a section in the chain it was fitted in (the other kernels at direct response)."""
    m = copy.deepcopy(model)
    for i, conv in enumerate(m.net.conv_layers):
        if i not in CHAIN[section]:
            conv.set_to_direct_response()
    with torch.no_grad():
        y_hat, _ = m(dm.x, dm.t_in)
    G, G_hat = fraction(dm, dm.y), fraction(dm, y_hat)[0]  # the chain is linear: one curve for all rows
    out = {}
    for edge, rows in GROUPS.items():
        d = G_hat - G[rows].mean(axis=0)
        out[edge] = {"max_abs_dF": float(np.abs(d).max()), "rms_dF": float(np.sqrt(np.mean(d**2))),
                     "repeat_std_max": float(G[rows].std(axis=0).max()), "n_rows": int(G[rows].shape[0])}
    return out, G, G_hat


def fit(model, dm, layer, epochs, accelerator, logger):
    for i, conv in enumerate(model.net.conv_layers):
        conv.unfreeze() if i == layer else conv.freeze()
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=epochs, logger=logger,
                         enable_checkpointing=False, enable_progress_bar=False, log_every_n_steps=1)
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


def summary_plot(curves, metrics, ts, E, t_bed, F_bed, path):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for a, (section, (t, G, G_hat)) in zip(ax.flat, curves.items()):
        for g, c in ((g, c) for g, c in (("Ar rise", "C0"), ("Ar fall", "C9"), ("He rise", "C2"), ("He fall", "C8")) if g in GROUPS):
            a.plot(t, G[GROUPS[g]].T, ".", c=c, ms=1, alpha=0.4)
        a.plot(t, G_hat, c="C3", label="fit")
        m = metrics[section]["all"]
        a.set_title(f"{section}: max |ΔF| {m['max_abs_dF']:.3f}, RMS {m['rms_dF']:.4f}  (" + ("Ar blue, He green" if "He" in TRACERS else "Ar rise dark, fall light") + ")", fontsize=8)
        a.set_xlim(0, 60 if section == "total" else 15)
        a.set_xlabel("t in window [s]")
    for k, (t, e) in enumerate(zip(ts, E)):
        ax[1, 1].plot(t, e, label=("E_up", "E_down", "E_cap")[k])
    ax[1, 1].plot(t_bed, F_bed, "k--", label="F_bed_in = step ⊗ E_up")
    ax[1, 1].set_xlim(0, 45)
    ax[1, 1].legend(fontsize=7)
    ax[1, 1].set_title("kernels", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=20_000, help="epochs per section")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--tag", default="", help="appended to the run name")
    ap.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    ap.add_argument("--tracers", default="Ar+He", choices=tuple(ROWS))
    args = ap.parse_args()
    global TRACERS, GROUPS
    TRACERS, GROUPS = args.tracers, GROUPS_OF[args.tracers]

    pl.seed_everything(args.seed, workers=True)
    name = f"CCA_001_{TRACERS.replace('+', '')}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}" + (f"_{args.tag}" if args.tag else "")
    runs = HERE / "Runs"
    runs.mkdir(exist_ok=True)
    settings = {"epochs_per_section": args.epochs, "seed": args.seed, "lr": args.lr, "accelerator": args.accelerator,
                "switch_delay": SWITCH_DELAY, "kernel_times": T_KERNELS, "tracers": TRACERS, "edges": "rise + fall"}
    versions = {"nRTD": git_head(), "python": sys.version.split()[0], "torch": torch.__version__,
                "lightning": pl.__version__, "numpy": np.__version__}

    dms = {s: data_module(s) for s in ("cap", "upstream", "total")}
    model = RTDModule(kernel_sizes=N_KERNELS, kernel_times=T_KERNELS, learning_rate=args.lr)
    for conv in model.net.conv_layers:
        conv.set_to_direct_response()

    walls = {}
    for section, layer in (("cap", CAP), ("upstream", UP), ("total", DOWN)):
        logger = WandbLogger(name=f"{name}_{section}", group=name, save_dir=str(runs), mode=args.wandb_mode,
                             config=settings | {"section": section, "versions": versions}, **WANDB)
        walls[section] = fit(model, dms[section], layer, args.epochs, args.accelerator, logger)
        print(f"{section}: {args.epochs} epochs in {walls[section]:.1f} s", flush=True)
        if section != "total":
            wandb.finish()
    model.cpu()
    settings["wall_s"] = walls

    metrics, curves = {}, {}
    for section, dm in dms.items():
        metrics[section], G, G_hat = evaluate(model, dm, section)
        curves[section] = (dm.t_out[0].numpy(), G, G_hat)
    E = [e.numpy() for e in model.net.E]
    ts = [c.t_kernel.numpy() for c in model.net.conv_layers]
    x_step = dms["total"].x[:1]  # one Ar rise with the total system's levels, step at SWITCH_DELAY
    with torch.no_grad():
        y_bed, t_bed = model.net.partial_forward(x_step, dms["total"].t_in[:1], begin=UP, end=UP + 1)
    F_bed = ((y_bed[0, 0] - x_step[0, 0, 0]) / (x_step[0, 0, -1] - x_step[0, 0, 0])).numpy()
    t_bed = t_bed[0].numpy()
    metrics["E_sum_dt"] = {k: float(e.sum() * DT) for k, e in zip(("up", "down", "cap"), E)}
    metrics["E_mean_t_s"] = {k: float((t * e).sum() / e.sum()) for k, t, e in zip(("up", "down", "cap"), ts, E)}
    metrics["E_at_t0"] = {k: float(e[0]) for k, e in zip(("up", "down", "cap"), E)}

    run = {"ts": ts, "Es": E, "delta_ts": [DT] * 3, "t_bed_in": t_bed, "F_bed_in": F_bed,
           "metrics": metrics, "settings": settings, "versions": versions}
    for section in ("cap", "upstream", "total"):
        dm = dms[section]
        run |= {f"t_in {section}": dm.t_in.numpy(), f"x_in {section}": dm.x.numpy(),
                f"t_out {section}": dm.t_out.numpy(), f"x_out {section}": dm.y.numpy()}
    np.save(runs / f"{name}.npy", run)
    (runs / f"{name}_metrics.json").write_text(json.dumps({"metrics": metrics, "settings": settings, "versions": versions}, indent=2))
    fig = summary_plot(curves, metrics, ts, E, t_bed, F_bed, runs / f"{name}_fits.png")

    if wandb.run is not None:
        wandb.log({"fits": wandb.Image(fig)})
        wandb.run.summary.update({f"{s}/{e}/{k}": v for s in ("cap", "upstream", "total")
                                  for e, m in metrics[s].items() for k, v in m.items()}
                                 | {f"E/{q}/{k}": v for q in ("E_sum_dt", "E_mean_t_s", "E_at_t0") for k, v in metrics[q].items()})
        wandb.finish()

    print(json.dumps(metrics, indent=1))
    print(f"written {runs / name}.npy, _metrics.json, _fits.png")


if __name__ == "__main__":
    main()

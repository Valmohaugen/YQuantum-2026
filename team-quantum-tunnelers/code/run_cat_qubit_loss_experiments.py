from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


NOTEBOOK_PATH = Path(
    r"C:\Users\bodhi\YaleQuantumHackathon\output\jupyter-notebook\cat-qubit-stabilization-optimizer.ipynb"
)
OUTPUT_DIR = Path(
    r"C:\Users\bodhi\YaleQuantumHackathon\output\jupyter-notebook\cat-qubit-loss-experiments"
)


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def exec_cells(ns: dict, *cell_indices: int) -> None:
    notebook = load_notebook()
    for idx in cell_indices:
        exec("".join(notebook["cells"][idx]["source"]), ns)


def find_code_cell_source(notebook: dict, pattern: str) -> str:
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if pattern in src:
                return src
    raise ValueError(f"Could not find code cell containing pattern: {pattern}")


def load_core_namespace() -> dict:
    ns: dict = {}
    exec_cells(ns, 3, 4, 5)

    notebook = load_notebook()
    no_drift_src = "\n".join(
        line
        for line in find_code_cell_source(notebook, "def run_cmaes_without_drift():").splitlines()
        if "no_drift_history = run_cmaes_without_drift()" not in line
    )
    drift_src = "\n".join(
        line
        for line in find_code_cell_source(notebook, "def run_cmaes_under_drift(start_mean):").splitlines()
        if 'drift_history = run_cmaes_under_drift(no_drift_history["final_mean"])' not in line
    )
    exec(no_drift_src, ns)
    exec(drift_src, ns)

    ns["plt"].switch_backend("Agg")
    return ns


def configure_run(ns: dict, *, n_target: float, seed: int, overrides: dict | None = None) -> None:
    cfg = dict(ns["PROFILES"]["quick"])
    if overrides:
        cfg.update(overrides)

    ns["RUN_PROFILE"] = "quick"
    ns["CFG"] = cfg
    ns["ETA_TARGET"] = float(n_target)
    ns["SEED"] = int(seed)
    ns["np"].random.seed(int(seed))
    ns["TS_Z"] = ns["jnp"].linspace(0.0, cfg["tz_stop"], cfg["tz_points"])
    ns["TS_X"] = ns["jnp"].linspace(0.0, cfg["tx_stop"], cfg["tx_points"])


def nanargbest(values: np.ndarray, kind: str) -> int:
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0
    safe = np.where(finite, values, -np.inf if kind == "max" else np.inf)
    return int(np.argmax(safe) if kind == "max" else np.argmin(safe))


def summarize_history(history: dict, *, n_target: float, run_kind: str) -> dict:
    best_reward_idx = nanargbest(history["best_reward"], "max")
    closest_bias_idx = nanargbest(np.abs(history["best_bias"] - n_target), "min")

    summary = {
        "run_kind": run_kind,
        "n": float(n_target),
        "epochs": int(len(history["epochs"])),
        "final_reward": float(history["reward"][-1]),
        "best_reward": float(history["best_reward"][best_reward_idx]),
        "best_reward_epoch": int(best_reward_idx),
        "best_Tz": float(history["best_Tz"][best_reward_idx]),
        "best_Tx": float(history["best_Tx"][best_reward_idx]),
        "best_bias": float(history["best_bias"][best_reward_idx]),
        "closest_bias": float(history["best_bias"][closest_bias_idx]),
        "closest_bias_epoch": int(closest_bias_idx),
        "closest_bias_error": float(abs(history["best_bias"][closest_bias_idx] - n_target)),
        "final_mean_Re_g2": float(history["final_mean"][0]),
        "final_mean_Im_g2": float(history["final_mean"][1]),
        "final_mean_Re_eps_d": float(history["final_mean"][2]),
        "final_mean_Im_eps_d": float(history["final_mean"][3]),
    }

    if "ideal_g2" in history:
        re_rmse = float(np.sqrt(np.mean((history["mean"][:, 0] - history["ideal_g2"][:, 0]) ** 2)))
        im_rmse = float(np.sqrt(np.mean((history["mean"][:, 1] - history["ideal_g2"][:, 1]) ** 2)))
        summary["tracking_rmse_Re_g2"] = re_rmse
        summary["tracking_rmse_Im_g2"] = im_rmse
    if "ideal_eps_d" in history:
        re_rmse = float(np.sqrt(np.mean((history["mean"][:, 2] - history["ideal_eps_d"][:, 0]) ** 2)))
        im_rmse = float(np.sqrt(np.mean((history["mean"][:, 3] - history["ideal_eps_d"][:, 1]) ** 2)))
        summary["tracking_rmse_Re_eps_d"] = re_rmse
        summary["tracking_rmse_Im_eps_d"] = im_rmse

    return summary


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_metrics_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_candidate(ns: dict, x: np.ndarray, drift_values=(0.0, 0.0, 0.0, 0.0)) -> dict:
    return ns["evaluate_candidate_with_fits"](
        np.asarray(x, dtype=np.float32),
        drift_values=np.asarray(drift_values, dtype=np.float32),
    )


def plot_decay_fit(path: Path, ts: np.ndarray, curve: np.ndarray, fit: dict, observable_label: str, tau_label: str, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ts, curve, linewidth=2, label=observable_label)
    ax.plot(ts, fit["y_fit"], "--", linewidth=2, label=f"Linear Fit, {tau_label} = {fit['tau']:.2f} us")
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Expectation Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_no_drift_sweep(results: list[dict], output_dir: Path) -> list[Path]:
    paths: list[Path] = []

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for result in results:
        epochs = result["history"]["epochs"]
        n_value = result["n"]
        axes[0].plot(epochs, result["history"]["reward"], label=f"n={n_value:.0f}")
        axes[0].fill_between(
            epochs,
            result["history"]["reward"] - result["history"]["reward_std"],
            result["history"]["reward"] + result["history"]["reward_std"],
            alpha=0.2,
        )
        axes[1].plot(epochs, result["history"]["mean_bias"], label=f"n={n_value:.0f}")
        axes[1].fill_between(
            epochs,
            result["history"]["mean_bias"] - result["history"]["std_bias"],
            result["history"]["mean_bias"] + result["history"]["std_bias"],
            alpha=0.2,
        )
        axes[1].axhline(n_value, linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Reward = -objective")
    axes[0].set_title("No-Drift Reward Comparison")
    axes[0].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Best Tz / Tx")
    axes[1].set_title("No-Drift Bias Comparison")
    axes[1].legend()
    fig.tight_layout()

    path = output_dir / "no_drift_reward_bias_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for result in results:
        epochs = result["history"]["epochs"]
        n_value = result["n"]
        axes[0].plot(epochs, result["history"]["mean_Tz"], label=f"n={n_value:.0f}")
        axes[0].fill_between(
            epochs,
            result["history"]["mean_Tz"] - result["history"]["std_Tz"],
            result["history"]["mean_Tz"] + result["history"]["std_Tz"],
            alpha=0.2,
        )
        axes[1].plot(epochs, result["history"]["mean_Tx"], label=f"n={n_value:.0f}")
        axes[1].fill_between(
            epochs,
            result["history"]["mean_Tx"] - result["history"]["std_Tx"],
            result["history"]["mean_Tx"] + result["history"]["std_Tx"],
            alpha=0.2,
        )

    axes[0].set_ylabel("Tz (us)")
    axes[0].set_title("Best Tz vs Epoch")
    axes[0].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Tx (us)")
    axes[1].set_title("Best Tx vs Epoch")
    axes[1].legend()
    fig.tight_layout()

    path = output_dir / "no_drift_lifetime_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


def plot_drift_run(result: dict, output_dir: Path) -> list[Path]:
    history = result["history"]
    epochs = history["epochs"]
    n_value = result["n"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(epochs, history["reward"], label="Mean reward")
    axes[0].fill_between(
        epochs,
        history["reward"] - history["reward_std"],
        history["reward"] + history["reward_std"],
        alpha=0.2,
    )
    axes[0].set_ylabel("Reward")
    axes[0].set_title(f"Reward Under Drift (n={n_value:.0f})")
    axes[0].legend()

    axes[1].plot(epochs, history["mean"][:, 0], label="optimizer Re(g2)")
    axes[1].fill_between(
        epochs,
        history["mean"][:, 0] - history["std"][:, 0],
        history["mean"][:, 0] + history["std"][:, 0],
        alpha=0.15,
    )
    axes[1].plot(epochs, history["ideal_g2"][:, 0], "--", label="ideal Re(g2)")
    axes[1].set_ylabel("Re(g2)")
    axes[1].legend()

    axes[2].plot(epochs, history["mean"][:, 2], label="optimizer Re(eps_d)")
    axes[2].fill_between(
        epochs,
        history["mean"][:, 2] - history["std"][:, 2],
        history["mean"][:, 2] + history["std"][:, 2],
        alpha=0.15,
    )
    axes[2].plot(epochs, history["ideal_eps_d"][:, 0], "--", label="ideal Re(eps_d)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Re(eps_d)")
    axes[2].legend()

    fig.tight_layout()
    path = output_dir / f"drift_tracking_n_{int(n_value)}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [path]


def save_wigner_epoch_gif(ns: dict, history: dict, path: Path, *, title_run: str = "CMA-ES") -> Path:
    states = []
    for epoch, params in enumerate(history["best_params"]):
        try:
            drift_values = ns["synthetic_drift"](epoch) if "ideal_g2" in history else np.zeros(4, dtype=np.float32)
            states.append(ns["simulate_storage_snapshot"](params, drift_values=drift_values))
        except Exception:
            continue

    if not states:
        raise RuntimeError("No valid states available for Wigner GIF.")

    gif = ns["dq"].plot.wigner_gif(
        ns["dq"].stack(states),
        fps=4,
        gif_duration=max(5.0, len(states) / 4.0),
        xmax=4.0,
        clear=True,
    )
    path.write_bytes(gif.data)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ns = load_core_namespace()

    study_overrides = {
        "print_every": 2,
    }

    no_drift_targets = [200.0]
    no_drift_results: list[dict] = []
    metric_rows: list[dict] = []

    for offset, n_target in enumerate(no_drift_targets):
        configure_run(ns, n_target=n_target, seed=10 + offset, overrides=study_overrides)
        history = ns["run_cmaes_without_drift"]()
        summary = summarize_history(history, n_target=n_target, run_kind="no_drift")
        no_drift_results.append({"n": n_target, "history": history, "summary": summary})
        metric_rows.append(summary)

    configure_run(ns, n_target=200.0, seed=30, overrides=study_overrides)
    drift_history = ns["run_cmaes_under_drift"](no_drift_results[-1]["history"]["final_mean"])
    drift_summary = summarize_history(drift_history, n_target=200.0, run_kind="drift")
    drift_result = {
        "n": 200.0,
        "history": drift_history,
        "summary": drift_summary,
    }
    metric_rows.append(drift_summary)

    cma_best_idx = nanargbest(no_drift_results[-1]["history"]["best_reward"], "max")
    cma_best_params = no_drift_results[-1]["history"]["best_params"][cma_best_idx]
    cma_best_diag = evaluate_candidate(ns, cma_best_params)

    plot_paths = []
    plot_paths.extend(plot_no_drift_sweep(no_drift_results, OUTPUT_DIR))
    plot_paths.extend(plot_drift_run(drift_result, OUTPUT_DIR))
    plot_paths.append(
        save_wigner_epoch_gif(
            ns,
            no_drift_results[-1]["history"],
            OUTPUT_DIR / "cma_epoch_wigner.gif",
        )
    )
    plot_paths.append(
        plot_decay_fit(
            OUTPUT_DIR / "cma_best_tz_linear_fit.png",
            np.asarray(ns["TS_Z"]),
            cma_best_diag["z_curve"],
            cma_best_diag["fit_z"],
            "<Z_L>",
            "T_Z",
            "CMA-ES Best Logical Z Decay",
        )
    )
    plot_paths.append(
        plot_decay_fit(
            OUTPUT_DIR / "cma_best_tx_linear_fit.png",
            np.asarray(ns["TS_X"]),
            cma_best_diag["x_curve"],
            cma_best_diag["fit_x"],
            "<X_L>",
            "T_X",
            "CMA-ES Best Logical X Decay",
        )
    )

    save_json(
        OUTPUT_DIR / "metrics.json",
        {
            "study_config": study_overrides,
            "bounds": np.asarray(ns["BOUNDS"]).tolist(),
            "drift_model": "step",
            "drift_step_epochs": int(ns["DRIFT_STEP_EPOCHS"]),
            "drift_levels": np.asarray(ns["DRIFT_LEVELS"]).tolist(),
            "no_drift": [result["summary"] for result in no_drift_results],
            "drift": drift_summary,
            "plots": [str(path) for path in plot_paths],
        },
    )
    save_metrics_csv(OUTPUT_DIR / "metrics.csv", metric_rows)

    print("Saved outputs to", OUTPUT_DIR)
    for path in plot_paths:
        print("PLOT", path)
    print("METRICS", OUTPUT_DIR / "metrics.json")
    print("METRICS", OUTPUT_DIR / "metrics.csv")


if __name__ == "__main__":
    main()

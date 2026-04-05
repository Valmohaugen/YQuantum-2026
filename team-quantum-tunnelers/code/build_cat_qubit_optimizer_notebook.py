from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path(
    r"C:\Users\bodhi\YaleQuantumHackathon\output\jupyter-notebook\cat-qubit-stabilization-optimizer.ipynb"
)


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
        # Experiment: Cat Qubit Stabilization Optimizer

        Objective:
        - Build a cat-qubit loss from `dynamiqs` simulations, optimize the four real control knobs with CMA-ES, and show the controller recovering under synthetic hardware drift.
        - Success means the reward improves without drift, the fitted bias `T_Z / T_X` moves toward the target `n`, and the optimizer adjusts its knobs when the effective `g_2` coupling drifts.
        """
    ),
    code(
        """
        # Run this once in a Python 3.11 notebook kernel if the imports below are missing.
        # Restart the kernel after installation finishes.
        import importlib.util
        import subprocess
        import sys

        REQUIRED_PACKAGES = [
            ("dynamiqs", "dynamiqs>=0.3.0"),
            ("jax", "jax[cpu]"),
            ("cmaes", "cmaes"),
            ("scipy", "scipy"),
            ("matplotlib", "matplotlib"),
            ("gymnasium", "gymnasium"),
            ("stable_baselines3", "stable-baselines3"),
            ("torch", "torch"),
            ("ipykernel", "ipykernel"),
        ]

        missing = [spec for module_name, spec in REQUIRED_PACKAGES if importlib.util.find_spec(module_name) is None]

        if missing:
            print("Installing:", ", ".join(missing))
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("Restart the kernel before continuing.")
        else:
            print("All required packages are already installed.")
        """
    ),
    md(
        """
        ## Plan

        - Use the exact Lindblad cat-qubit model from the challenge notebook.
        - Extract `T_Z` and `T_X` from robust exponential fits to `<Z_L>` and `<X_L>`.
        - JIT-compile and batch the expensive `dynamiqs.mesolve` calls with `jit(vmap(...))`.
        - Replace the nonlinear fit with an early-time linear surrogate for `T_Z` and `T_X`.
- Use the objective `(T_Z / T_X - n)^2 - (T_Z + n * T_X)`, with `n = ETA_TARGET`.
        - Keep CMA-ES in a local box around the known-good cat point `[1, 0, 4, 0]`, mirroring the pi-pulse example's bounded search.
        - Start with `RUN_PROFILE = "quick"` for a CPU-friendly end-to-end run, then switch to `"full"` for longer hackathon sweeps.
        """
    ),
    code(
        """
        import warnings

        import numpy as np
        import dynamiqs as dq
        import jax.numpy as jnp
        from jax import jit, vmap
        from cmaes import SepCMA
        import gymnasium as gym
        from gymnasium import spaces
        from matplotlib import pyplot as plt
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback

        plt.style.use("seaborn-v0_8-whitegrid")
        warnings.filterwarnings(
            "ignore",
            message="A `SparseDIAQArray` has been converted to a `DenseQArray` while computing its matrix exponential.",
        )

        SEED = 0
        np.random.seed(SEED)

        RUN_PROFILE = "quick"  # "quick" for a smoke test, "full" for longer challenge runs.
        PROFILES = {
            "quick": {
                "n_epochs": 30,
                "n_drift_epochs": 30,
                "ppo_total_timesteps": 768,
                "ppo_n_steps": 64,
                "sigma": 0.02,
                "drift_sigma": 0.015,
                "tz_stop": 20.0,
                "tz_points": 10,
                "tx_stop": 0.6,
                "tx_points": 8,
                "linear_fit_fraction_z": 0.4,
                "linear_fit_fraction_x": 0.5,
                "print_every": 5,
            },
            "full": {
                "n_epochs": 120,
                "n_drift_epochs": 160,
                "ppo_total_timesteps": 4096,
                "ppo_n_steps": 128,
                "sigma": 0.03,
                "drift_sigma": 0.02,
                "tz_stop": 45.0,
                "tz_points": 16,
                "tx_stop": 0.8,
                "tx_points": 10,
                "linear_fit_fraction_z": 0.35,
                "linear_fit_fraction_x": 0.4,
                "print_every": 10,
            },
        }

        CFG = PROFILES[RUN_PROFILE]

        NA = 15
        NB = 5
        KAPPA_B = 10.0
        KAPPA_A = 1.0
        POPULATION_SIZE = 12

        ETA_TARGET = 200.0
        LOSS_PENALTY = 5_000.0

        INITIAL_GUESS = np.array([1.0, 0.0, 4.0, 0.0], dtype=float)
        GLOBAL_BOUNDS = np.array(
            [
                [-1.5, 1.5],   # Re(g_2)
                [-1.5, 1.5],   # Im(g_2)
                [0.5, 6.0],    # Re(eps_d)
                [-2.5, 2.5],   # Im(eps_d)
            ],
            dtype=float,
        )
        LOCAL_SEARCH_WIDTHS = np.array([0.25, 0.20, 0.90, 0.35], dtype=float)
        BOUNDS = np.stack(
            [
                np.maximum(INITIAL_GUESS - LOCAL_SEARCH_WIDTHS, GLOBAL_BOUNDS[:, 0]),
                np.minimum(INITIAL_GUESS + LOCAL_SEARCH_WIDTHS, GLOBAL_BOUNDS[:, 1]),
            ],
            axis=1,
        )

        TS_Z = jnp.linspace(0.0, CFG["tz_stop"], CFG["tz_points"])
        TS_X = jnp.linspace(0.0, CFG["tx_stop"], CFG["tx_points"])

        CFG
        """
    ),
    code(
        """
        # Early-time linear surrogate for the decay curves.
        def early_linear_fit(x, y, fit_fraction=0.4, min_points=4):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)

            if not np.all(np.isfinite(y)):
                raise ValueError("Non-finite points in decay trace.")

            n_fit = max(min_points, int(np.ceil(fit_fraction * len(x))))
            n_fit = min(len(x), n_fit)
            if n_fit < 2:
                raise ValueError("Not enough points for linear fit.")

            x_fit = x[:n_fit]
            y_fit = y[:n_fit]
            slope, intercept = np.polyfit(x_fit, y_fit, 1)

            if (not np.isfinite(slope)) or slope >= -1e-8:
                raise ValueError("Trace does not show early-time decay.")

            tau = float(y_fit[0] / (-slope))
            if (not np.isfinite(tau)) or tau <= 0.0:
                raise ValueError("Invalid linearized lifetime.")

            return {
                "tau": tau,
                "slope": float(slope),
                "intercept": float(intercept),
                "y_fit": slope * x + intercept,
                "x_fit": x_fit,
                "y_fit_window": slope * x_fit + intercept,
            }


        A_OP = dq.tensor(dq.destroy(NA), dq.eye(NB))
        B_OP = dq.tensor(dq.eye(NA), dq.destroy(NB))
        BUFFER_VAC = dq.fock(NB, 0)
        LOSS_B = jnp.sqrt(KAPPA_B) * B_OP
        LOSS_A = jnp.sqrt(KAPPA_A) * A_OP
        SX_OP = (1j * jnp.pi * A_OP.dag() @ A_OP).expm()
        MESOLVE_OPTIONS = dq.Options(progress_meter=False)


        def _effective_cat_tuple(x, drift_values=(0.0, 0.0, 0.0, 0.0)):
            x = jnp.asarray(x, dtype=jnp.float32)
            drift_values = jnp.asarray(drift_values, dtype=jnp.float32)

            g_2_command = x[0] + 1j * x[1]
            eps_d_command = x[2] + 1j * x[3]
            g_2_drift = drift_values[0] + 1j * drift_values[1]
            eps_d_drift = drift_values[2] + 1j * drift_values[3]
            g_2 = g_2_command - g_2_drift
            eps_d = eps_d_command - eps_d_drift

            kappa_2 = 4.0 * jnp.abs(g_2) ** 2 / KAPPA_B
            eps_2 = 2.0 * g_2 * eps_d / KAPPA_B
            alpha_sq = 2.0 / jnp.maximum(kappa_2, 1e-8) * (eps_2 - KAPPA_A / 4.0)

            alpha_real = jnp.real(alpha_sq)
            alpha_imag = jnp.abs(jnp.imag(alpha_sq))
            alpha_estimate = jnp.sqrt(jnp.maximum(alpha_real, 1e-6))

            valid = (
                jnp.isfinite(alpha_real)
                & jnp.isfinite(alpha_imag)
                & (kappa_2 > 1e-6)
                & (alpha_real > 0.05)
                & (alpha_real < 9.0)
                & (alpha_imag < 0.75)
            )

            return g_2_command, eps_d_command, g_2, eps_d, kappa_2, eps_2, alpha_estimate, alpha_imag, valid


        def effective_cat_parameters(x, drift_values=(0.0, 0.0, 0.0, 0.0)):
            g_2_command, eps_d_command, g_2, eps_d, kappa_2, eps_2, alpha_estimate, alpha_imag, valid = _effective_cat_tuple(x, drift_values)
            return {
                "g_2_command": g_2_command,
                "eps_d_command": eps_d_command,
                "g_2": g_2,
                "eps_d": eps_d,
                "kappa_2": kappa_2,
                "eps_2": eps_2,
                "alpha_estimate": alpha_estimate,
                "alpha_imag": alpha_imag,
                "valid": valid,
            }
        """
    ),
    code(
        """
        def _effective_tuple(x, drift_values):
            _, _, g_2, eps_d, _, _, alpha_estimate, alpha_imag, valid = _effective_cat_tuple(x, drift_values)
            return g_2, eps_d, alpha_estimate, alpha_imag, valid


        @jit
        def simulate_z_trace(x, drift_values):
            g_2, eps_d, alpha_estimate, alpha_imag, valid = _effective_tuple(x, drift_values)

            g_state = dq.coherent(NA, alpha_estimate)
            e_state = dq.coherent(NA, -alpha_estimate)
            sz = g_state @ g_state.dag() - e_state @ e_state.dag()
            sz = dq.tensor(sz, dq.eye(NB))

            H = (
                jnp.conj(g_2) * A_OP @ A_OP @ B_OP.dag()
                + g_2 * A_OP.dag() @ A_OP.dag() @ B_OP
                - eps_d * B_OP.dag()
                - jnp.conj(eps_d) * B_OP
            )

            psi0 = dq.tensor(g_state, BUFFER_VAC)

            res = dq.mesolve(
                H,
                [LOSS_B, LOSS_A],
                psi0,
                TS_Z,
                exp_ops=[SX_OP, sz],
                options=MESOLVE_OPTIONS,
            )

            return res.expects[1].real, alpha_estimate, alpha_imag, valid


        @jit
        def simulate_x_trace(x, drift_values):
            g_2, eps_d, alpha_estimate, alpha_imag, valid = _effective_tuple(x, drift_values)

            g_state = dq.coherent(NA, alpha_estimate)
            e_state = dq.coherent(NA, -alpha_estimate)
            x_state = g_state + e_state
            x_state = x_state / x_state.norm()

            H = (
                jnp.conj(g_2) * A_OP @ A_OP @ B_OP.dag()
                + g_2 * A_OP.dag() @ A_OP.dag() @ B_OP
                - eps_d * B_OP.dag()
                - jnp.conj(eps_d) * B_OP
            )

            psi0 = dq.tensor(x_state, BUFFER_VAC)

            res = dq.mesolve(
                H,
                [LOSS_B, LOSS_A],
                psi0,
                TS_X,
                exp_ops=[SX_OP],
                options=MESOLVE_OPTIONS,
            )

            return res.expects[0].real, alpha_estimate, alpha_imag, valid


        batched_z_trace = jit(vmap(simulate_z_trace, in_axes=(0, None)))
        batched_x_trace = jit(vmap(simulate_x_trace, in_axes=(0, None)))


        def evaluate_population(xs, drift_values=(0.0, 0.0, 0.0, 0.0)):
            xs = np.asarray(xs, dtype=np.float32)
            drift_values = np.asarray(drift_values, dtype=np.float32)

            z_curves, alpha_z, alpha_imag_z, valid_z = batched_z_trace(jnp.asarray(xs), jnp.asarray(drift_values))
            x_curves, alpha_x, alpha_imag_x, valid_x = batched_x_trace(jnp.asarray(xs), jnp.asarray(drift_values))

            z_curves = np.asarray(z_curves, dtype=float)
            x_curves = np.asarray(x_curves, dtype=float)
            alpha_z = np.asarray(alpha_z, dtype=float)
            alpha_x = np.asarray(alpha_x, dtype=float)
            alpha_imag = np.maximum(np.asarray(alpha_imag_z, dtype=float), np.asarray(alpha_imag_x, dtype=float))
            valid = np.asarray(valid_z & valid_x, dtype=bool)

            losses = np.full(len(xs), LOSS_PENALTY, dtype=float)
            rewards = -losses.copy()
            Tz = np.full(len(xs), np.nan, dtype=float)
            Tx = np.full(len(xs), np.nan, dtype=float)
            bias = np.full(len(xs), np.nan, dtype=float)
            alpha = np.full(len(xs), np.nan, dtype=float)

            for i in range(len(xs)):
                alpha[i] = alpha_z[i]

                if not valid[i]:
                    continue

                try:
                    fit_z = early_linear_fit(
                        np.asarray(TS_Z),
                        z_curves[i],
                        fit_fraction=CFG["linear_fit_fraction_z"],
                    )
                    fit_x = early_linear_fit(
                        np.asarray(TS_X),
                        x_curves[i],
                        fit_fraction=CFG["linear_fit_fraction_x"],
                    )

                    tz = float(fit_z["tau"])
                    tx = float(fit_x["tau"])

                    if (not np.isfinite(tz)) or (not np.isfinite(tx)) or tz <= 0.0 or tx <= 0.0:
                        continue

                    ratio = tz / max(tx, 1e-6)
                    loss = (ratio - ETA_TARGET) ** 2 - (tz + ETA_TARGET * tx)

                    Tz[i] = tz
                    Tx[i] = tx
                    bias[i] = ratio
                    losses[i] = loss
                    rewards[i] = -loss
                except Exception:
                    continue

            best_idx = int(np.argmin(losses))

            return {
                "loss": losses,
                "reward": rewards,
                "Tz": Tz,
                "Tx": Tx,
                "bias": bias,
                "alpha": alpha,
                "valid": valid,
                "z_curves": z_curves,
                "x_curves": x_curves,
                "best_idx": best_idx,
            }


        def evaluate_population_under_drift(p_augmented):
            p_augmented = np.asarray(p_augmented, dtype=np.float32)
            drift_values = p_augmented[0, 4:]
            return evaluate_population(p_augmented[:, :4], drift_values=drift_values)


        def loss_func(x):
            return float(evaluate_population(np.asarray([x], dtype=np.float32))["loss"][0])


        def loss_func_under_drift(p):
            p = np.asarray(p, dtype=np.float32)
            return float(evaluate_population(np.asarray([p[:4]], dtype=np.float32), drift_values=p[4:])["loss"][0])


        def evaluate_candidate_with_fits(x, drift_values=(0.0, 0.0, 0.0, 0.0)):
            x = np.asarray(x, dtype=np.float32)
            metrics = evaluate_population(np.asarray([x], dtype=np.float32), drift_values=drift_values)

            fit_z = early_linear_fit(
                np.asarray(TS_Z),
                metrics["z_curves"][0],
                fit_fraction=CFG["linear_fit_fraction_z"],
            )
            fit_x = early_linear_fit(
                np.asarray(TS_X),
                metrics["x_curves"][0],
                fit_fraction=CFG["linear_fit_fraction_x"],
            )

            return {
                "params": x,
                "drift_values": np.asarray(drift_values, dtype=np.float32),
                "loss": float(metrics["loss"][0]),
                "reward": float(metrics["reward"][0]),
                "Tz": float(metrics["Tz"][0]),
                "Tx": float(metrics["Tx"][0]),
                "bias": float(metrics["bias"][0]),
                "alpha": float(metrics["alpha"][0]),
                "z_curve": np.asarray(metrics["z_curves"][0], dtype=float),
                "x_curve": np.asarray(metrics["x_curves"][0], dtype=float),
                "fit_z": fit_z,
                "fit_x": fit_x,
            }


        def plot_linear_surrogate_decay(ts, curve, fit, observable_label, tau_label, title):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(ts, curve, linewidth=2, label=observable_label)
            ax.plot(ts, fit["y_fit"], "--", linewidth=2, label=f"Linear Fit, {tau_label} = {fit['tau']:.2f} us")
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("Expectation Value")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig, ax


        def finite_mean_std(values):
            values = np.asarray(values, dtype=float)
            finite = np.isfinite(values)
            if not np.any(finite):
                return np.nan, np.nan
            return float(np.mean(values[finite])), float(np.std(values[finite]))


        def simulate_storage_snapshot(params, drift_values=(0.0, 0.0, 0.0, 0.0), tstop=3.0):
            params = np.asarray(params, dtype=np.float32)
            drift_values = np.asarray(drift_values, dtype=np.float32)
            g_2, eps_d, alpha_estimate, alpha_imag, valid = _effective_tuple(jnp.asarray(params), jnp.asarray(drift_values))
            if not bool(valid):
                raise ValueError("Invalid parameters for Wigner snapshot.")

            H = (
                jnp.conj(g_2) * A_OP @ A_OP @ B_OP.dag()
                + g_2 * A_OP.dag() @ A_OP.dag() @ B_OP
                - eps_d * B_OP.dag()
                - jnp.conj(eps_d) * B_OP
            )
            psi0 = dq.tensor(dq.fock(NA, 1), BUFFER_VAC)
            tsave = jnp.array([0.0, tstop], dtype=jnp.float32)
            res = dq.mesolve(
                H,
                [LOSS_B, LOSS_A],
                psi0,
                tsave,
                options=MESOLVE_OPTIONS,
            )
            return dq.ptrace(res.states[-1], 0)
        """
    ),
    md(
        """
        ## Single-candidate sanity check

        The first call is usually the slowest because JAX compiles the batched solver. After that, the same shapes run much faster.
        """
    ),
    code(
        """
        baseline_eval = evaluate_population(INITIAL_GUESS[None, :])
        baseline_idx = baseline_eval["best_idx"]

        baseline_summary = {
            "loss": baseline_eval["loss"][baseline_idx],
            "reward": baseline_eval["reward"][baseline_idx],
            "Tz_us": baseline_eval["Tz"][baseline_idx],
            "Tx_us": baseline_eval["Tx"][baseline_idx],
            "bias": baseline_eval["bias"][baseline_idx],
            "alpha_estimate": baseline_eval["alpha"][baseline_idx],
        }

        baseline_summary
        """
    ),
    code(
        """
        fit_z = early_linear_fit(
            np.asarray(TS_Z),
            baseline_eval["z_curves"][0],
            fit_fraction=CFG["linear_fit_fraction_z"],
        )
        fit_x = early_linear_fit(
            np.asarray(TS_X),
            baseline_eval["x_curves"][0],
            fit_fraction=CFG["linear_fit_fraction_x"],
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(np.asarray(TS_Z), baseline_eval["z_curves"][0], "o-", label="⟨Z_L⟩")
        axes[0].plot(np.asarray(TS_Z), fit_z["y_fit"], "--", label=f"linear surrogate, T_Z={fit_z['tau']:.2f} us")
        axes[0].set_title("Logical Z Decay")
        axes[0].set_xlabel("Time (us)")
        axes[0].set_ylabel("Expectation value")
        axes[0].legend()

        axes[1].plot(np.asarray(TS_X), baseline_eval["x_curves"][0], "o-", label="⟨X_L⟩")
        axes[1].plot(np.asarray(TS_X), fit_x["y_fit"], "--", label=f"linear surrogate, T_X={fit_x['tau']:.3f} us")
        axes[1].set_title("Logical X Decay")
        axes[1].set_xlabel("Time (us)")
        axes[1].set_ylabel("Expectation value")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## CMA-ES Without Drift

        We minimize the exact fitted loss
        `(T_Z / T_X - n)^2 - (T_Z + n * T_X)`, where `n = ETA_TARGET`,
        with CMA-ES and track the population statistics over time.
        """
    ),
    code(
        """
        def run_cmaes_without_drift():
            optimizer = SepCMA(
                mean=INITIAL_GUESS.copy(),
                sigma=CFG["sigma"],
                bounds=BOUNDS.copy(),
                population_size=POPULATION_SIZE,
                seed=SEED,
            )

            history = {
                "mean": [],
                "std": [],
                "reward": [],
                "reward_std": [],
                "mean_Tz": [],
                "std_Tz": [],
                "mean_Tx": [],
                "std_Tx": [],
                "mean_bias": [],
                "std_bias": [],
                "best_loss": [],
                "best_reward": [],
                "best_Tz": [],
                "best_Tx": [],
                "best_bias": [],
                "best_params": [],
            }

            for epoch in range(CFG["n_epochs"]):
                xs = np.stack([optimizer.ask() for _ in range(optimizer.population_size)])
                metrics = evaluate_population(xs)

                optimizer.tell([(xs[i], float(metrics["loss"][i])) for i in range(len(xs))])

                best_idx = int(metrics["best_idx"])

                history["mean"].append(np.asarray(optimizer.mean).copy())
                history["std"].append(np.std(xs, axis=0))
                history["reward"].append(float(np.mean(metrics["reward"])))
                history["reward_std"].append(float(np.std(metrics["reward"])))
                mean_tz, std_tz = finite_mean_std(metrics["Tz"])
                mean_tx, std_tx = finite_mean_std(metrics["Tx"])
                mean_bias, std_bias = finite_mean_std(metrics["bias"])
                history["mean_Tz"].append(mean_tz)
                history["std_Tz"].append(std_tz)
                history["mean_Tx"].append(mean_tx)
                history["std_Tx"].append(std_tx)
                history["mean_bias"].append(mean_bias)
                history["std_bias"].append(std_bias)
                history["best_loss"].append(float(metrics["loss"][best_idx]))
                history["best_reward"].append(float(metrics["reward"][best_idx]))
                history["best_Tz"].append(float(metrics["Tz"][best_idx]))
                history["best_Tx"].append(float(metrics["Tx"][best_idx]))
                history["best_bias"].append(float(metrics["bias"][best_idx]))
                history["best_params"].append(xs[best_idx].copy())

                if epoch % CFG["print_every"] == 0:
                    print(
                        f"Epoch {epoch:03d} | "
                        f"mean={np.asarray(optimizer.mean)} | "
                        f"reward={history['reward'][-1]:8.3f} +/- {history['reward_std'][-1]:8.3f} | "
                        f"best bias={history['best_bias'][-1]:8.2f}"
                    )

            for key, value in history.items():
                history[key] = np.asarray(value)

            history["epochs"] = np.arange(CFG["n_epochs"])
            history["final_mean"] = np.asarray(optimizer.mean).copy()
            return history


        no_drift_history = run_cmaes_without_drift()
        """
    ),
    code(
        """
        epochs = no_drift_history["epochs"]

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, no_drift_history["reward"], label="Mean reward")
        plt.fill_between(
            epochs,
            no_drift_history["reward"] - no_drift_history["reward_std"],
            no_drift_history["reward"] + no_drift_history["reward_std"],
            alpha=0.2,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Reward = -objective")
        plt.title("Reward vs Epoch (No Drift)")
        plt.legend()
        plt.show()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        labels = ["Re(g_2)", "Im(g_2)", "Re(eps_d)", "Im(eps_d)"]
        for idx, ax in enumerate(axes.flat):
            ax.plot(epochs, no_drift_history["mean"][:, idx], label=labels[idx])
            ax.fill_between(
                epochs,
                no_drift_history["mean"][:, idx] - no_drift_history["std"][:, idx],
                no_drift_history["mean"][:, idx] + no_drift_history["std"][:, idx],
                alpha=0.15,
            )
            ax.set_ylabel(labels[idx])
            ax.legend()
        for ax in axes[-1]:
            ax.set_xlabel("Epoch")
        fig.suptitle("Parameter Convergence (No Drift)")
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, no_drift_history["mean_Tz"], label="Mean T_Z")
        axes[0].fill_between(
            epochs,
            no_drift_history["mean_Tz"] - no_drift_history["std_Tz"],
            no_drift_history["mean_Tz"] + no_drift_history["std_Tz"],
            alpha=0.2,
        )
        axes[0].plot(epochs, no_drift_history["mean_Tx"], label="Mean T_X")
        axes[0].fill_between(
            epochs,
            no_drift_history["mean_Tx"] - no_drift_history["std_Tx"],
            no_drift_history["mean_Tx"] + no_drift_history["std_Tx"],
            alpha=0.2,
        )
        axes[0].set_title("T_Z and T_X vs Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Lifetime (us)")
        axes[0].legend()

        axes[1].plot(epochs, no_drift_history["mean_bias"], label="Mean bias")
        axes[1].fill_between(
            epochs,
            no_drift_history["mean_bias"] - no_drift_history["std_bias"],
            no_drift_history["mean_bias"] + no_drift_history["std_bias"],
            alpha=0.2,
        )
        axes[1].axhline(ETA_TARGET, color="black", linestyle="--", label=f"target={ETA_TARGET:.0f}")
        axes[1].set_title("Bias vs Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("T_Z / T_X")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Best CMA-ES Decay Fits

        These plots show the early-time linear surrogate overlaid on the simulated decay traces at the best no-drift CMA-ES point.
        """
    ),
    code(
        """
        cma_best_idx = int(np.nanargmax(no_drift_history["best_reward"]))
        cma_best_params = no_drift_history["best_params"][cma_best_idx]
        cma_best_diag = evaluate_candidate_with_fits(cma_best_params)

        plot_linear_surrogate_decay(
            np.asarray(TS_Z),
            cma_best_diag["z_curve"],
            cma_best_diag["fit_z"],
            "<Z_L>",
            "T_Z",
            "Best CMA-ES Logical Z Decay",
        )
        plt.show()

        plot_linear_surrogate_decay(
            np.asarray(TS_X),
            cma_best_diag["x_curve"],
            cma_best_diag["fit_x"],
            "<X_L>",
            "T_X",
            "Best CMA-ES Logical X Decay",
        )
        plt.show()
        """
    ),
    md(
        """
        ## CMA-ES Wigner GIF

        Each frame below shows the storage-mode Wigner function after a short correction run, using the best candidate from one optimizer epoch.
        """
    ),
    code(
        """
        wigner_epoch_states = []
        for params in no_drift_history["best_params"]:
            try:
                wigner_epoch_states.append(simulate_storage_snapshot(params))
            except Exception:
                continue

        cma_wigner_gif = dq.plot.wigner_gif(
            dq.stack(wigner_epoch_states),
            fps=4,
            gif_duration=max(5.0, len(wigner_epoch_states) / 4.0),
            xmax=4.0,
            clear=True,
        )
        cma_wigner_gif
        """
    ),
    md(
        """
        ## Add Synthetic Drift

        The drifted loss uses the same four knobs, but each candidate is evaluated with epoch-dependent additive step drift on both `g_2` and `eps_d`.
        The optimizer only controls the first four entries; the final four values are synthetic drift terms that jump every few epochs.
        """
    ),
    code(
        """
        DRIFT_STEP_EPOCHS = 20
        DRIFT_LEVELS = np.array(
            [
                [0.00, 0.00, 0.00, 0.00],
                [0.10, 0.06, 0.22, -0.08],
            ],
            dtype=np.float32,
        )


        def synthetic_drift(epoch):
            block = min(epoch // DRIFT_STEP_EPOCHS, len(DRIFT_LEVELS) - 1)
            return DRIFT_LEVELS[block].copy()


        def ideal_g2_command(epoch, reference_g2_command):
            drift_values = synthetic_drift(epoch)
            drift_complex = drift_values[0] + 1j * drift_values[1]
            return reference_g2_command + drift_complex


        def ideal_eps_d_command(epoch, reference_eps_d_command):
            drift_values = synthetic_drift(epoch)
            drift_complex = drift_values[2] + 1j * drift_values[3]
            return reference_eps_d_command + drift_complex


        def run_cmaes_under_drift(start_mean):
            optimizer = SepCMA(
                mean=np.asarray(start_mean, dtype=float).copy(),
                sigma=CFG["drift_sigma"],
                bounds=BOUNDS.copy(),
                population_size=POPULATION_SIZE,
                seed=SEED + 1,
            )

            history = {
                "mean": [],
                "std": [],
                "reward": [],
                "reward_std": [],
                "mean_Tz": [],
                "std_Tz": [],
                "mean_Tx": [],
                "std_Tx": [],
                "mean_bias": [],
                "std_bias": [],
                "best_loss": [],
                "best_reward": [],
                "best_Tz": [],
                "best_Tx": [],
                "best_bias": [],
                "best_params": [],
                "drift": [],
                "ideal_g2": [],
                "ideal_eps_d": [],
            }

            reference_g2_command = complex(start_mean[0], start_mean[1])
            reference_eps_d_command = complex(start_mean[2], start_mean[3])

            for epoch in range(CFG["n_drift_epochs"]):
                drift_values = synthetic_drift(epoch)
                xs = np.stack([optimizer.ask() for _ in range(optimizer.population_size)])
                xs_augmented = np.concatenate(
                    [xs, np.repeat(drift_values[None, :], len(xs), axis=0)],
                    axis=1,
                )

                metrics = evaluate_population_under_drift(xs_augmented)
                optimizer.tell([(xs[i], float(metrics["loss"][i])) for i in range(len(xs))])

                best_idx = int(metrics["best_idx"])
                ideal_g2 = ideal_g2_command(epoch, reference_g2_command)
                ideal_eps_d = ideal_eps_d_command(epoch, reference_eps_d_command)

                history["mean"].append(np.asarray(optimizer.mean).copy())
                history["std"].append(np.std(xs, axis=0))
                history["reward"].append(float(np.mean(metrics["reward"])))
                history["reward_std"].append(float(np.std(metrics["reward"])))
                mean_tz, std_tz = finite_mean_std(metrics["Tz"])
                mean_tx, std_tx = finite_mean_std(metrics["Tx"])
                mean_bias, std_bias = finite_mean_std(metrics["bias"])
                history["mean_Tz"].append(mean_tz)
                history["std_Tz"].append(std_tz)
                history["mean_Tx"].append(mean_tx)
                history["std_Tx"].append(std_tx)
                history["mean_bias"].append(mean_bias)
                history["std_bias"].append(std_bias)
                history["best_loss"].append(float(metrics["loss"][best_idx]))
                history["best_reward"].append(float(metrics["reward"][best_idx]))
                history["best_Tz"].append(float(metrics["Tz"][best_idx]))
                history["best_Tx"].append(float(metrics["Tx"][best_idx]))
                history["best_bias"].append(float(metrics["bias"][best_idx]))
                history["best_params"].append(xs[best_idx].copy())
                history["drift"].append(drift_values.copy())
                history["ideal_g2"].append(np.array([ideal_g2.real, ideal_g2.imag], dtype=float))
                history["ideal_eps_d"].append(np.array([ideal_eps_d.real, ideal_eps_d.imag], dtype=float))

                if epoch % CFG["print_every"] == 0:
                    print(
                        f"Epoch {epoch:03d} | "
                        f"drift={drift_values} | "
                        f"mean={np.asarray(optimizer.mean)} | "
                        f"reward={history['reward'][-1]:8.3f}"
                    )

            for key, value in history.items():
                history[key] = np.asarray(value)

            history["epochs"] = np.arange(CFG["n_drift_epochs"])
            history["final_mean"] = np.asarray(optimizer.mean).copy()
            return history


        drift_history = run_cmaes_under_drift(no_drift_history["final_mean"])
        """
    ),
    code(
        """
        drift_epochs = drift_history["epochs"]

        plt.figure(figsize=(8, 4))
        plt.plot(drift_epochs, drift_history["reward"], label="Mean reward")
        plt.fill_between(
            drift_epochs,
            drift_history["reward"] - drift_history["reward_std"],
            drift_history["reward"] + drift_history["reward_std"],
            alpha=0.2,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Reward = -objective")
        plt.title("Reward vs Epoch Under Drift")
        plt.legend()
        plt.show()

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        axes[0].plot(drift_epochs, drift_history["mean"][:, 0], label="optimizer mean Re(g_2)")
        axes[0].fill_between(
            drift_epochs,
            drift_history["mean"][:, 0] - drift_history["std"][:, 0],
            drift_history["mean"][:, 0] + drift_history["std"][:, 0],
            alpha=0.15,
        )
        axes[0].plot(drift_epochs, drift_history["ideal_g2"][:, 0], "--", label="ideal compensation Re(g_2)")
        axes[0].set_ylabel("Re(g_2)")
        axes[0].set_title("Drift Tracking")
        axes[0].legend()

        axes[1].plot(drift_epochs, drift_history["mean"][:, 2], label="optimizer mean Re(eps_d)")
        axes[1].fill_between(
            drift_epochs,
            drift_history["mean"][:, 2] - drift_history["std"][:, 2],
            drift_history["mean"][:, 2] + drift_history["std"][:, 2],
            alpha=0.15,
        )
        axes[1].plot(drift_epochs, drift_history["ideal_eps_d"][:, 0], "--", label="ideal compensation Re(eps_d)")
        axes[1].set_ylabel("Re(eps_d)")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## PPO Without Drift

        As a lightweight RL baseline, we cast the no-drift problem as a one-step continuous bandit:
        the observation is constant, the action is the four normalized control knobs, and the reward is the same no-drift cat-qubit reward used by CMA-ES.
        """
    ),
    code(
        """
        class CatQubitBanditEnv(gym.Env):
            metadata = {"render_modes": []}

            def __init__(self, reward_scale=100.0):
                super().__init__()
                self.bounds = np.asarray(BOUNDS, dtype=np.float32)
                self.reward_scale = float(reward_scale)
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
                self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

            def action_to_params(self, action):
                action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
                low = self.bounds[:, 0]
                high = self.bounds[:, 1]
                return low + 0.5 * (action + 1.0) * (high - low)

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(1, dtype=np.float32), {}

            def step(self, action):
                params = self.action_to_params(action)
                diag = evaluate_candidate_with_fits(params)
                reward_raw = float(diag["reward"])

                info = {
                    "reward_raw": reward_raw,
                    "loss_raw": float(diag["loss"]),
                    "Tz": float(diag["Tz"]),
                    "Tx": float(diag["Tx"]),
                    "bias": float(diag["bias"]),
                    "params": params.copy(),
                }
                return (
                    np.zeros(1, dtype=np.float32),
                    reward_raw / self.reward_scale,
                    True,
                    False,
                    info,
                )


        class PPOBanditCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.rollout_reward_mean = []
                self.eval_reward = []
                self.eval_Tz = []
                self.eval_Tx = []
                self.eval_bias = []
                self.eval_params = []
                self._rollout_rewards = []

            def _on_step(self):
                for info in self.locals.get("infos", []):
                    if "reward_raw" in info:
                        self._rollout_rewards.append(float(info["reward_raw"]))
                return True

            def _on_rollout_end(self):
                if self._rollout_rewards:
                    self.rollout_reward_mean.append(float(np.mean(self._rollout_rewards)))
                    self._rollout_rewards = []

                obs = np.zeros((1,), dtype=np.float32)
                action, _ = self.model.predict(obs, deterministic=True)
                params = self.training_env.envs[0].action_to_params(action)
                diag = evaluate_candidate_with_fits(params)
                self.eval_reward.append(float(diag["reward"]))
                self.eval_Tz.append(float(diag["Tz"]))
                self.eval_Tx.append(float(diag["Tx"]))
                self.eval_bias.append(float(diag["bias"]))
                self.eval_params.append(np.asarray(params, dtype=float))


        ppo_env = CatQubitBanditEnv(reward_scale=100.0)
        ppo_callback = PPOBanditCallback()
        ppo_model = PPO(
            "MlpPolicy",
            ppo_env,
            learning_rate=1e-3,
            n_steps=CFG["ppo_n_steps"],
            batch_size=64,
            n_epochs=10,
            gamma=0.0,
            gae_lambda=0.0,
            ent_coef=0.02,
            vf_coef=0.1,
            policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
            seed=SEED,
            verbose=0,
        )
        ppo_model.learn(total_timesteps=CFG["ppo_total_timesteps"], callback=ppo_callback, progress_bar=False)

        ppo_action, _ = ppo_model.predict(np.zeros((1,), dtype=np.float32), deterministic=True)
        ppo_params = ppo_env.action_to_params(ppo_action)
        ppo_diag = evaluate_candidate_with_fits(ppo_params)

        ppo_summary = {
            "reward": float(ppo_diag["reward"]),
            "loss": float(ppo_diag["loss"]),
            "Tz_us": float(ppo_diag["Tz"]),
            "Tx_us": float(ppo_diag["Tx"]),
            "bias": float(ppo_diag["bias"]),
            "params": ppo_params,
        }

        ppo_summary
        """
    ),
    code(
        """
        ppo_rollouts = np.arange(1, len(ppo_callback.rollout_reward_mean) + 1)

        plt.figure(figsize=(8, 4))
        plt.plot(ppo_rollouts, ppo_callback.rollout_reward_mean, label="Mean sampled reward per rollout")
        plt.plot(ppo_rollouts, ppo_callback.eval_reward, label="Deterministic policy reward")
        plt.xlabel("PPO Rollout")
        plt.ylabel("Reward = -objective")
        plt.title("PPO No-Drift Training Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        plot_linear_surrogate_decay(
            np.asarray(TS_Z),
            ppo_diag["z_curve"],
            ppo_diag["fit_z"],
            "<Z_L>",
            "T_Z",
            "PPO Logical Z Decay",
        )
        plt.show()

        plot_linear_surrogate_decay(
            np.asarray(TS_X),
            ppo_diag["x_curve"],
            ppo_diag["fit_x"],
            "<X_L>",
            "T_X",
            "PPO Logical X Decay",
        )
        plt.show()
        """
    ),
    code(
        """
        summary = {
            "run_profile": RUN_PROFILE,
            "n": ETA_TARGET,
            "no_drift_final_mean": no_drift_history["final_mean"],
            "no_drift_best_reward": float(np.nanmax(no_drift_history["best_reward"])),
            "no_drift_best_bias": float(no_drift_history["best_bias"][np.nanargmin(np.abs(no_drift_history["best_bias"] - ETA_TARGET))]),
            "drift_final_mean": drift_history["final_mean"],
            "drift_best_reward": float(np.nanmax(drift_history["best_reward"])),
            "ppo_no_drift_reward": float(ppo_diag["reward"]),
            "ppo_no_drift_bias": float(ppo_diag["bias"]),
            "ppo_no_drift_Tx": float(ppo_diag["Tx"]),
        }

        summary
        """
    ),
    md(
        """
        ## Notes

        - `RUN_PROFILE = "quick"` keeps the notebook runnable on a laptop CPU while still generating all requested plots.
        - `RUN_PROFILE = "full"` switches to a longer `120 + 160` epoch schedule with denser decay traces, closer to the intended hackathon benchmark.
        - The expensive physics is still JIT-compiled and batched with `jit(vmap(...))`.
        - Lifetimes are estimated from an early-time linear surrogate rather than a nonlinear exponential fit, which makes the online objective cheaper and less brittle.
        - The optimization target is now `(T_Z / T_X - n)^2 - (T_Z + n * T_X)`, so the search rewards long lifetimes while still driving the bias toward `ETA_TARGET`.
        - The optimizer now relies on the tight local CMA-ES bounds around the known-good cat point instead of an additional soft penalty on the imaginary part of `alpha_estimate`.
        - The drift model now applies a single additive step to both `g_2` and `eps_d` at epoch `DRIFT_STEP_EPOCHS`, so the compensation task is a one-time hardware jump rather than smooth drift.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3.11",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")

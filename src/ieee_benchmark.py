from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def normalized_metrics(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> dict[str, float]:
    err = (y_pred - y_true) / scale
    return {
        "MSE": float(np.mean(err**2)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
    }


def persistence(data: np.ndarray, cut: int) -> np.ndarray:
    return data[cut - 1 : -1]


def global_damped_trend(data: np.ndarray, cut: int, scale: float) -> tuple[np.ndarray, float]:
    y_train = data[2:cut]
    last = data[1 : cut - 1]
    diff = data[1 : cut - 1] - data[: cut - 2]
    best_alpha = 0.0
    best_mse = float("inf")
    for alpha in np.linspace(-1.0, 1.0, 401):
        pred = last + alpha * diff
        mse = np.mean(((pred - y_train) / scale) ** 2)
        if mse < best_mse:
            best_mse = float(mse)
            best_alpha = float(alpha)
    pred_test = data[cut - 1 : -1] + best_alpha * (data[cut - 1 : -1] - data[cut - 2 : -2])
    return pred_test, best_alpha


def per_flow_damped_trend(data: np.ndarray, cut: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    y_train = data[2:cut]
    last = data[1 : cut - 1]
    diff = data[1 : cut - 1] - data[: cut - 2]
    pred_test = np.zeros_like(data[cut:])
    alphas = np.zeros(data.shape[1], dtype=np.float64)
    for flow_idx in range(data.shape[1]):
        best_alpha = 0.0
        best_mse = float("inf")
        for alpha in np.linspace(-1.0, 1.0, 401):
            pred = last[:, flow_idx] + alpha * diff[:, flow_idx]
            mse = np.mean(((pred - y_train[:, flow_idx]) / scale) ** 2)
            if mse < best_mse:
                best_mse = float(mse)
                best_alpha = float(alpha)
        alphas[flow_idx] = best_alpha
        pred_test[:, flow_idx] = data[cut - 1 : -1, flow_idx] + best_alpha * (
            data[cut - 1 : -1, flow_idx] - data[cut - 2 : -2, flow_idx]
        )
    return pred_test, alphas


def main() -> None:
    data_path = ROOT / "data" / "raw" / "Abilene-OD_pair.csv"
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, nrows=2016)
    values = df.drop(columns=["time"]).to_numpy(dtype=np.float64)
    cut = int(len(values) * 0.8)
    y_test = values[cut:]
    scale = float(values.max() - values.min())

    model_rows = []
    pred = persistence(values, cut)
    model_rows.append({"model": "Persistence", **normalized_metrics(y_test, pred, scale)})

    pred, alpha = global_damped_trend(values, cut, scale)
    model_rows.append(
        {
            "model": f"Global damped trend (alpha={alpha:.3f})",
            **normalized_metrics(y_test, pred, scale),
        }
    )

    pred, alphas = per_flow_damped_trend(values, cut, scale)
    model_rows.append(
        {
            "model": "Proposed per-flow damped residual",
            **normalized_metrics(y_test, pred, scale),
        }
    )

    # Published Abilene normalized values reported by Kablaoui et al. (2024), best GRU model.
    model_rows.append(
        {
            "model": "Reported 2024 journal best GRU",
            "MSE": 0.030781,
            "RMSE": 0.175446,
            "MAE": 0.133847,
        }
    )

    results = pd.DataFrame(model_rows).sort_values("MSE").reset_index(drop=True)
    results.to_csv(results_dir / "ieee_abilene_2016_sota_metrics.csv", index=False)

    metadata = pd.DataFrame(
        [
            {
                "rows": len(values),
                "flows": values.shape[1],
                "train_rows": cut,
                "test_rows": len(values) - cut,
                "normalization": "global min-max range over the 2016 matrices",
                "per_flow_alpha_mean": float(np.mean(alphas)),
                "per_flow_alpha_min": float(np.min(alphas)),
                "per_flow_alpha_max": float(np.max(alphas)),
            }
        ]
    )
    metadata.to_csv(results_dir / "ieee_abilene_2016_sota_metadata.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    plot_df = results.sort_values("MSE", ascending=False)
    colors = ["#777777" if "Reported" in name else "#1f77b4" for name in plot_df["model"]]
    ax.barh(plot_df["model"], plot_df["MSE"], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Normalized MSE (log scale)")
    ax.set_title("Abilene 2016-matrix normalized benchmark comparison")
    fig.tight_layout()
    fig.savefig(figures_dir / "ieee_sota_comparison.png", dpi=220)
    plt.close(fig)

    print(results.to_string(index=False))
    print(metadata.to_string(index=False))


if __name__ == "__main__":
    main()


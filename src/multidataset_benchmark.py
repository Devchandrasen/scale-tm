from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]


DATASETS = {
    "Abilene": ROOT / "data" / "raw" / "Abilene-OD_pair.csv",
    "GEANT": ROOT / "data" / "raw" / "GEANT-OD_pair.csv",
    "CERNET": ROOT / "data" / "raw" / "CERNET-OD_pair.csv",
}


FOLDS = (
    {"fold": 1, "train_end": 0.60, "val_end": 0.70, "test_end": 0.80},
    {"fold": 2, "train_end": 0.70, "val_end": 0.80, "test_end": 0.90},
    {"fold": 3, "train_end": 0.80, "val_end": 0.90, "test_end": 1.00},
)


@dataclass
class Config:
    sample_rows: int = 2880
    lookback_hours: float = 6.0
    horizon_hours: float = 1.0
    stride_hours: float = 1.0
    ridge_alpha: float = 1.0
    alpha_grid_size: int = 81
    ensemble_step: float = 0.10
    bootstrap_samples: int = 300
    seed: int = 42
    results_dir: Path = ROOT / "results"
    figures_dir: Path = ROOT / "figures"


@dataclass
class DatasetBundle:
    name: str
    raw: np.ndarray
    times: pd.Series
    columns: list[str]
    step_minutes: int
    lookback: int
    horizon: int
    stride: int
    daily_period: int


@dataclass
class FoldWindows:
    x_train: np.ndarray
    y_train: np.ndarray
    flow_train: np.ndarray
    end_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    flow_val: np.ndarray
    end_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    flow_test: np.ndarray
    end_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    train_cut: int
    val_cut: int
    test_cut: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="All-flow multi-dataset traffic forecasting benchmark.")
    parser.add_argument("--sample-rows", type=int, default=Config.sample_rows)
    parser.add_argument("--lookback-hours", type=float, default=Config.lookback_hours)
    parser.add_argument("--horizon-hours", type=float, default=Config.horizon_hours)
    parser.add_argument("--stride-hours", type=float, default=Config.stride_hours)
    parser.add_argument("--ensemble-step", type=float, default=Config.ensemble_step)
    parser.add_argument("--bootstrap-samples", type=int, default=Config.bootstrap_samples)
    return parser.parse_args()


def read_time(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%Y-%m-%d-%H-%M", errors="coerce")
    if parsed.isna().any():
        parsed = pd.to_datetime(values, errors="coerce")
    return parsed


def infer_step_minutes(times: pd.Series) -> int:
    parsed = pd.DatetimeIndex(times.dropna())
    if len(parsed) < 2:
        return 5
    deltas = pd.Series(parsed[1:] - parsed[:-1]).dt.total_seconds().dropna() / 60.0
    if len(deltas) == 0:
        return 5
    return max(1, int(round(float(deltas.median()))))


def load_dataset(name: str, path: Path, cfg: Config) -> DatasetBundle:
    df = pd.read_csv(path, nrows=cfg.sample_rows)
    times = read_time(df["time"])
    columns = [c for c in df.columns if c != "time"]
    numeric = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    raw = numeric.to_numpy(dtype=np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    step_minutes = infer_step_minutes(times)
    lookback = max(2, int(round(cfg.lookback_hours * 60.0 / step_minutes)))
    horizon = max(1, int(round(cfg.horizon_hours * 60.0 / step_minutes)))
    stride = max(1, int(round(cfg.stride_hours * 60.0 / step_minutes)))
    daily_period = max(horizon + 1, int(round(1440.0 / step_minutes)))

    return DatasetBundle(
        name=name,
        raw=raw,
        times=times,
        columns=columns,
        step_minutes=step_minutes,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
        daily_period=daily_period,
    )


def time_features(times: pd.Series) -> np.ndarray:
    dt = pd.DatetimeIndex(times)
    valid = ~dt.isna()
    minute_of_day = np.zeros(len(dt), dtype=np.float32)
    day_of_week = np.zeros(len(dt), dtype=np.float32)
    if valid.any():
        valid_dt = dt[valid]
        minute_of_day[valid] = valid_dt.hour.to_numpy() * 60 + valid_dt.minute.to_numpy()
        day_of_week[valid] = valid_dt.dayofweek.to_numpy()
    return np.column_stack(
        [
            np.sin(2.0 * np.pi * minute_of_day / 1440.0),
            np.cos(2.0 * np.pi * minute_of_day / 1440.0),
            np.sin(2.0 * np.pi * day_of_week / 7.0),
            np.cos(2.0 * np.pi * day_of_week / 7.0),
        ]
    ).astype(np.float32)


def make_windows(bundle: DatasetBundle, fold: dict[str, float]) -> FoldWindows:
    n_rows, n_flows = bundle.raw.shape
    train_cut = int(n_rows * fold["train_end"])
    val_cut = int(n_rows * fold["val_end"])
    test_cut = int(n_rows * fold["test_end"])

    mean = bundle.raw[:train_cut].mean(axis=0)
    std = bundle.raw[:train_cut].std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    norm = ((bundle.raw - mean) / std).astype(np.float32)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    flow_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []
    max_start = test_cut - bundle.lookback - bundle.horizon + 1
    starts = range(0, max_start, bundle.stride)
    flows = np.arange(n_flows, dtype=np.int32)

    for start in starts:
        end = start + bundle.lookback
        target_end = end + bundle.horizon
        x_parts.append(norm[start:end].T)
        y_parts.append(norm[end:target_end].T)
        flow_parts.append(flows)
        end_parts.append(np.full(n_flows, end, dtype=np.int32))

    x = np.concatenate(x_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.float32)
    flow = np.concatenate(flow_parts, axis=0)
    end = np.concatenate(end_parts, axis=0)
    target_end = end + bundle.horizon

    train_mask = target_end <= train_cut
    val_mask = (end >= train_cut) & (target_end <= val_cut)
    test_mask = (end >= val_cut) & (target_end <= test_cut)

    return FoldWindows(
        x_train=x[train_mask],
        y_train=y[train_mask],
        flow_train=flow[train_mask],
        end_train=end[train_mask],
        x_val=x[val_mask],
        y_val=y[val_mask],
        flow_val=flow[val_mask],
        end_val=end[val_mask],
        x_test=x[test_mask],
        y_test=y[test_mask],
        flow_test=flow[test_mask],
        end_test=end[test_mask],
        mean=mean.astype(np.float32),
        std=std,
        train_cut=train_cut,
        val_cut=val_cut,
        test_cut=test_cut,
    )


def moving_average(x: np.ndarray, kernel: int) -> np.ndarray:
    kernel = min(kernel, x.shape[1])
    if kernel % 2 == 0:
        kernel -= 1
    kernel = max(3, kernel)
    pad = kernel // 2
    padded = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    cumsum = np.cumsum(padded, axis=1, dtype=np.float64)
    cumsum = np.concatenate([np.zeros((x.shape[0], 1), dtype=np.float64), cumsum], axis=1)
    trend = (cumsum[:, kernel:] - cumsum[:, :-kernel]) / float(kernel)
    return trend.astype(np.float32)


def lag_features(x: np.ndarray, end: np.ndarray, tf: np.ndarray) -> np.ndarray:
    stats = np.column_stack(
        [
            x[:, -1],
            x[:, -1] - x[:, -2],
            x.mean(axis=1),
            x.std(axis=1),
            x.min(axis=1),
            x.max(axis=1),
        ]
    ).astype(np.float32)
    return np.concatenate([x, stats], axis=1).astype(np.float32)


def patch_features(x: np.ndarray, patch_size: int) -> np.ndarray:
    patch_size = max(1, patch_size)
    n_patches = x.shape[1] // patch_size
    trimmed = x[:, -n_patches * patch_size :]
    patches = trimmed.reshape(x.shape[0], n_patches, patch_size)
    patch_mean = patches.mean(axis=2)
    patch_last = patches[:, :, -1]
    return np.concatenate([patch_mean, patch_last, x[:, -patch_size:]], axis=1).astype(np.float32)


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(x_train, y_train)
    return model


def fit_damped_alphas(norm: np.ndarray, train_cut: int, grid_size: int) -> np.ndarray:
    train = norm[:train_cut]
    if len(train) < 3:
        return np.zeros(train.shape[1], dtype=np.float32)
    last = train[1:-1]
    prev = train[:-2]
    target = train[2:]
    delta = last - prev
    grid = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    scores = []
    for alpha in grid:
        pred = last + alpha * delta
        scores.append(np.mean((pred - target) ** 2, axis=0))
    scores_arr = np.stack(scores, axis=0)
    return grid[np.argmin(scores_arr, axis=0)].astype(np.float32)


def damped_predict(x: np.ndarray, flows: np.ndarray, alphas: np.ndarray, horizon: int) -> np.ndarray:
    prev = x[:, -2].copy()
    cur = x[:, -1].copy()
    flow_alpha = alphas[flows]
    preds = np.zeros((x.shape[0], horizon), dtype=np.float32)
    for step in range(horizon):
        nxt = cur + flow_alpha * (cur - prev)
        preds[:, step] = nxt
        prev, cur = cur, nxt
    return preds


def persistence_predict(x: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(x[:, -1:], horizon, axis=1).astype(np.float32)


def seasonal_naive_predict(
    norm: np.ndarray,
    x: np.ndarray,
    flows: np.ndarray,
    ends: np.ndarray,
    horizon: int,
    daily_period: int,
) -> np.ndarray:
    preds = np.repeat(x[:, -1:], horizon, axis=1).astype(np.float32)
    for h in range(horizon):
        source = ends + h - daily_period
        valid = source >= 0
        if valid.any():
            preds[valid, h] = norm[source[valid], flows[valid]]
    return preds


def denormalize(values: np.ndarray, flows: np.ndarray, windows: FoldWindows) -> np.ndarray:
    return values * windows.std[flows, None] + windows.mean[flows, None]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    denominator = max(float(np.sum(np.abs(y_true))), 1.0e-8)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "WAPE": float(np.sum(np.abs(error)) / denominator * 100.0),
        "sMAPE": float(
            np.mean(2.0 * np.abs(error) / (np.abs(y_true) + np.abs(y_pred) + 1.0e-8)) * 100.0
        ),
    }


def bootstrap_wape_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ends: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    unique_ends = np.unique(ends)
    abs_error_by_end = np.zeros(len(unique_ends), dtype=np.float64)
    abs_true_by_end = np.zeros(len(unique_ends), dtype=np.float64)
    for i, end in enumerate(unique_ends):
        mask = ends == end
        abs_error_by_end[i] = np.sum(np.abs(y_pred[mask] - y_true[mask]))
        abs_true_by_end[i] = np.sum(np.abs(y_true[mask]))
    if len(unique_ends) == 0:
        return float("nan"), float("nan")
    values = np.zeros(samples, dtype=np.float64)
    for i in range(samples):
        idx = rng.integers(0, len(unique_ends), len(unique_ends))
        values[i] = np.sum(abs_error_by_end[idx]) / max(np.sum(abs_true_by_end[idx]), 1.0e-8) * 100.0
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def simplex_weights(n_models: int, step: float) -> list[np.ndarray]:
    units = int(round(1.0 / step))
    weights: list[np.ndarray] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            weights.append(np.asarray(prefix + [remaining], dtype=np.float32) / units)
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots - 1)

    rec([], units, n_models)
    return weights


def choose_ensemble(
    candidate_names: list[str],
    val_preds: dict[str, np.ndarray],
    y_val: np.ndarray,
    step: float,
) -> tuple[np.ndarray, dict[str, float]]:
    best_wape = float("inf")
    best_weights = np.zeros(len(candidate_names), dtype=np.float32)
    for weights in simplex_weights(len(candidate_names), step):
        pred = np.zeros_like(y_val, dtype=np.float32)
        for weight, name in zip(weights, candidate_names):
            if weight > 0:
                pred += weight * val_preds[name]
        score = metrics(y_val, pred)["WAPE"]
        if score < best_wape:
            best_wape = score
            best_weights = weights
    return best_weights, {name: float(weight) for name, weight in zip(candidate_names, best_weights) if weight > 0}


def evaluate_fold(
    bundle: DatasetBundle,
    fold: dict[str, float],
    cfg: Config,
    rng: np.random.Generator,
) -> list[dict[str, float | int | str]]:
    windows = make_windows(bundle, fold)
    tf = time_features(bundle.times)
    norm = ((bundle.raw - windows.mean) / windows.std).astype(np.float32)

    y_val_raw = denormalize(windows.y_val, windows.flow_val, windows)
    y_test_raw = denormalize(windows.y_test, windows.flow_test, windows)

    alphas = fit_damped_alphas(norm, windows.train_cut, cfg.alpha_grid_size)

    train_lag = lag_features(windows.x_train, windows.end_train, tf)
    val_lag = lag_features(windows.x_val, windows.end_val, tf)
    test_lag = lag_features(windows.x_test, windows.end_test, tf)

    ridge = fit_ridge(train_lag, windows.y_train, cfg.ridge_alpha)
    ridge_val = ridge.predict(val_lag).astype(np.float32)
    ridge_test = ridge.predict(test_lag).astype(np.float32)

    last_train = windows.x_train[:, -1:]
    nlinear = fit_ridge(windows.x_train - last_train, windows.y_train - last_train, cfg.ridge_alpha)
    nlinear_val = (nlinear.predict(windows.x_val - windows.x_val[:, -1:]) + windows.x_val[:, -1:]).astype(np.float32)
    nlinear_test = (nlinear.predict(windows.x_test - windows.x_test[:, -1:]) + windows.x_test[:, -1:]).astype(
        np.float32
    )

    kernel = min(25, bundle.lookback)
    trend_train = moving_average(windows.x_train, kernel)
    trend_val = moving_average(windows.x_val, kernel)
    trend_test = moving_average(windows.x_test, kernel)
    dlinear_train = np.concatenate([trend_train, windows.x_train - trend_train], axis=1)
    dlinear_val = np.concatenate([trend_val, windows.x_val - trend_val], axis=1)
    dlinear_test = np.concatenate([trend_test, windows.x_test - trend_test], axis=1)
    dlinear = fit_ridge(dlinear_train, windows.y_train, cfg.ridge_alpha)
    dlinear_val_pred = dlinear.predict(dlinear_val).astype(np.float32)
    dlinear_test_pred = dlinear.predict(dlinear_test).astype(np.float32)

    train_patch = patch_features(windows.x_train, bundle.horizon)
    val_patch = patch_features(windows.x_val, bundle.horizon)
    test_patch = patch_features(windows.x_test, bundle.horizon)
    patch_linear = fit_ridge(train_patch, windows.y_train, cfg.ridge_alpha)
    patch_val = patch_linear.predict(val_patch).astype(np.float32)
    patch_test = patch_linear.predict(test_patch).astype(np.float32)

    damped_val = damped_predict(windows.x_val, windows.flow_val, alphas, bundle.horizon)
    damped_test = damped_predict(windows.x_test, windows.flow_test, alphas, bundle.horizon)
    persistence_val = persistence_predict(windows.x_val, bundle.horizon)
    persistence_test = persistence_predict(windows.x_test, bundle.horizon)
    seasonal_val = seasonal_naive_predict(
        norm, windows.x_val, windows.flow_val, windows.end_val, bundle.horizon, bundle.daily_period
    )
    seasonal_test = seasonal_naive_predict(
        norm, windows.x_test, windows.flow_test, windows.end_test, bundle.horizon, bundle.daily_period
    )

    val_norm_preds = {
        "Ridge lags": ridge_val,
        "NLinear": nlinear_val,
        "DLinear": dlinear_val_pred,
        "PatchLinear": patch_val,
        "Damped persistence": damped_val,
        "Persistence": persistence_val,
        "Seasonal naive": seasonal_val,
    }
    test_norm_preds = {
        "Ridge lags": ridge_test,
        "NLinear": nlinear_test,
        "DLinear": dlinear_test_pred,
        "PatchLinear": patch_test,
        "Damped persistence": damped_test,
        "Persistence": persistence_test,
        "Seasonal naive": seasonal_test,
    }

    val_raw_preds = {
        name: denormalize(pred, windows.flow_val, windows) for name, pred in val_norm_preds.items()
    }
    test_raw_preds = {
        name: denormalize(pred, windows.flow_test, windows) for name, pred in test_norm_preds.items()
    }
    ensemble_names = list(val_norm_preds.keys())
    weights, weight_map = choose_ensemble(ensemble_names, val_raw_preds, y_val_raw, cfg.ensemble_step)
    ensemble_test = np.zeros_like(test_norm_preds[ensemble_names[0]], dtype=np.float32)
    for weight, name in zip(weights, ensemble_names):
        if weight > 0:
            ensemble_test += weight * test_norm_preds[name]
    test_raw_preds["SCALE-TM linear ensemble"] = denormalize(ensemble_test, windows.flow_test, windows)

    rows: list[dict[str, float | int | str]] = []
    for model_name, pred_raw in test_raw_preds.items():
        row = metrics(y_test_raw, pred_raw)
        ci_low, ci_high = bootstrap_wape_ci(
            y_test_raw, pred_raw, windows.end_test, rng, cfg.bootstrap_samples
        )
        row.update(
            {
                "dataset": bundle.name,
                "fold": int(fold["fold"]),
                "model": model_name,
                "WAPE_CI_low": ci_low,
                "WAPE_CI_high": ci_high,
                "flows": int(bundle.raw.shape[1]),
                "rows": int(bundle.raw.shape[0]),
                "step_minutes": int(bundle.step_minutes),
                "lookback_steps": int(bundle.lookback),
                "horizon_steps": int(bundle.horizon),
                "train_windows": int(len(windows.x_train)),
                "val_windows": int(len(windows.x_val)),
                "test_windows": int(len(windows.x_test)),
                "ensemble_weights": json.dumps(weight_map, sort_keys=True),
            }
        )
        rows.append(row)
    return rows


def plot_results(results: pd.DataFrame, output_path: Path) -> None:
    preferred_order = [
        "SCALE-TM linear ensemble",
        "DLinear",
        "NLinear",
        "PatchLinear",
        "Ridge lags",
        "Damped persistence",
        "Persistence",
        "Seasonal naive",
    ]
    summary = (
        results.groupby(["dataset", "model"], as_index=False)["WAPE"]
        .mean()
        .assign(model=lambda df: pd.Categorical(df["model"], preferred_order, ordered=True))
        .sort_values(["dataset", "model"])
    )
    datasets = summary["dataset"].unique().tolist()
    models = [m for m in preferred_order if m in summary["model"].astype(str).unique()]
    x = np.arange(len(datasets))
    width = 0.10

    plt.figure(figsize=(12, 5.5))
    for idx, model in enumerate(models):
        values = []
        for dataset in datasets:
            match = summary[(summary["dataset"] == dataset) & (summary["model"].astype(str) == model)]
            values.append(float(match["WAPE"].iloc[0]) if len(match) else np.nan)
        offset = (idx - (len(models) - 1) / 2.0) * width
        plt.bar(x + offset, values, width=width, label=model)
    plt.xticks(x, datasets)
    plt.ylabel("Mean WAPE across folds (%)")
    plt.xlabel("Dataset")
    plt.title("All-flow multi-dataset one-hour forecasting benchmark")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = Config(
        sample_rows=args.sample_rows,
        lookback_hours=args.lookback_hours,
        horizon_hours=args.horizon_hours,
        stride_hours=args.stride_hours,
        ensemble_step=args.ensemble_step,
        bootstrap_samples=args.bootstrap_samples,
    )
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    started = time.time()
    all_rows: list[dict[str, float | int | str]] = []

    for name, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset {path}")
        bundle = load_dataset(name, path, cfg)
        print(
            f"{name}: rows={bundle.raw.shape[0]}, flows={bundle.raw.shape[1]}, "
            f"step={bundle.step_minutes} min, L={bundle.lookback}, H={bundle.horizon}"
        )
        for fold in FOLDS:
            print(f"  fold {fold['fold']}...")
            all_rows.extend(evaluate_fold(bundle, fold, cfg, rng))

    results = pd.DataFrame(all_rows)
    results_path = cfg.results_dir / "multidataset_benchmark.csv"
    results.to_csv(results_path, index=False)

    summary = (
        results.groupby(["dataset", "model"], as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            WAPE=("WAPE", "mean"),
            sMAPE=("sMAPE", "mean"),
            WAPE_CI_low=("WAPE_CI_low", "mean"),
            WAPE_CI_high=("WAPE_CI_high", "mean"),
            folds=("fold", "nunique"),
            flows=("flows", "first"),
            rows=("rows", "first"),
            step_minutes=("step_minutes", "first"),
            lookback_steps=("lookback_steps", "first"),
            horizon_steps=("horizon_steps", "first"),
        )
        .sort_values(["dataset", "WAPE"])
    )
    summary_path = cfg.results_dir / "multidataset_benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_results(results, cfg.figures_dir / "multidataset_wape.png")

    config_dict = asdict(cfg)
    config_dict["results_dir"] = str(config_dict["results_dir"])
    config_dict["figures_dir"] = str(config_dict["figures_dir"])
    metadata = {
        "config": config_dict,
        "folds": FOLDS,
        "runtime_seconds": time.time() - started,
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "figure": str(cfg.figures_dir / "multidataset_wape.png"),
        },
        "notes": [
            "All OD columns are included, including diagonal OD_i-i columns.",
            "Lookback and horizon are defined in hours, then converted to dataset-specific steps.",
            "Train, validation, and test windows are separated by target end time to avoid future-target leakage.",
            "Global linear baselines are fit without intercepts or calendar channels so zero-valued OD flows remain zero-preserving; the seasonal naive baseline is the explicit calendar control.",
        ],
    }
    metadata_path = cfg.results_dir / "multidataset_benchmark_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved {results_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {metadata_path}")
    print(f"Runtime seconds: {metadata['runtime_seconds']:.2f}")
    print(summary[summary["model"] == "SCALE-TM linear ensemble"].to_string(index=False))


if __name__ == "__main__":
    main()

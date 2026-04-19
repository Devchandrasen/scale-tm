from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    raw_csv: Path = ROOT / "data" / "raw" / "Abilene-OD_pair.csv"
    figures_dir: Path = ROOT / "figures"
    results_dir: Path = ROOT / "results"
    sample_rows: int = 8064
    top_flows: int = 12
    lookback: int = 72
    horizon: int = 12
    stride: int = 3
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 256
    epochs: int = 18
    patience: int = 5
    learning_rate: float = 1.0e-3
    seed: int = 42


@dataclass
class SplitData:
    times: pd.Series
    raw: np.ndarray
    norm: np.ndarray
    columns: list[str]
    mean: np.ndarray
    std: np.ndarray
    train_cut: int
    val_cut: int


@dataclass
class WindowData:
    x_basic_train: np.ndarray
    x_wave_train: np.ndarray
    y_train: np.ndarray
    flow_train: np.ndarray
    end_train: np.ndarray
    x_basic_val: np.ndarray
    x_wave_val: np.ndarray
    y_val: np.ndarray
    flow_val: np.ndarray
    end_val: np.ndarray
    x_basic_test: np.ndarray
    x_wave_test: np.ndarray
    y_test: np.ndarray
    flow_test: np.ndarray
    end_test: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        major, _ = torch.cuda.get_device_capability(0)
    except Exception:
        return torch.device("cpu")
    if major < 7:
        return torch.device("cpu")
    return torch.device("cuda")


def parse_time(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="%Y-%m-%d-%H-%M", errors="coerce")


def is_non_self_od(col: str) -> bool:
    try:
        left, right = col.replace("OD_", "").split("-")
        return int(left) != int(right)
    except ValueError:
        return False


def wavelet_smooth(series: np.ndarray, wavelet: str = "db2", level: int = 2) -> np.ndarray:
    coeffs = pywt.wavedec(series, wavelet=wavelet, mode="periodization", level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    threshold = sigma * math.sqrt(2.0 * math.log(max(len(series), 2)))
    smooth_coeffs = [coeffs[0]]
    smooth_coeffs.extend(pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:])
    smoothed = pywt.waverec(smooth_coeffs, wavelet=wavelet, mode="periodization")
    return smoothed[: len(series)]


def load_data(cfg: Config) -> SplitData:
    df = pd.read_csv(cfg.raw_csv, nrows=cfg.sample_rows)
    times = parse_time(df["time"])
    numeric_cols = [c for c in df.columns if c != "time" and is_non_self_od(c)]
    numeric = df[numeric_cols].astype("float32")

    train_cut = int(len(numeric) * cfg.train_ratio)
    val_cut = int(len(numeric) * (cfg.train_ratio + cfg.val_ratio))
    train_means = numeric.iloc[:train_cut].mean(axis=0)
    selected = train_means.sort_values(ascending=False).head(cfg.top_flows).index.tolist()

    raw = numeric[selected].to_numpy(dtype=np.float32)
    mean = raw[:train_cut].mean(axis=0)
    std = raw[:train_cut].std(axis=0)
    std = np.where(std < 1.0e-6, 1.0, std).astype(np.float32)
    norm = ((raw - mean) / std).astype(np.float32)

    return SplitData(
        times=times,
        raw=raw,
        norm=norm,
        columns=selected,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        train_cut=train_cut,
        val_cut=val_cut,
    )


def time_features(times: pd.Series) -> np.ndarray:
    dt = pd.DatetimeIndex(times)
    minute_of_day = dt.hour.to_numpy() * 60 + dt.minute.to_numpy()
    day_of_week = dt.dayofweek.to_numpy()
    return np.column_stack(
        [
            np.sin(2 * np.pi * minute_of_day / 1440.0),
            np.cos(2 * np.pi * minute_of_day / 1440.0),
            np.sin(2 * np.pi * day_of_week / 7.0),
            np.cos(2 * np.pi * day_of_week / 7.0),
        ]
    ).astype(np.float32)


def build_windows(data: SplitData, cfg: Config) -> WindowData:
    tf = time_features(data.times)
    x_basic_parts: list[np.ndarray] = []
    x_wave_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    flow_parts: list[int] = []
    end_parts: list[int] = []

    max_start = len(data.raw) - cfg.lookback - cfg.horizon + 1
    starts = range(0, max_start, cfg.stride)
    for start in starts:
        end = start + cfg.lookback
        target_end = end + cfg.horizon
        time_window = tf[start:end]
        for flow_idx in range(data.raw.shape[1]):
            traffic = data.norm[start:end, flow_idx : flow_idx + 1]
            raw_window = data.raw[start:end, flow_idx]
            trend_raw = wavelet_smooth(raw_window)
            trend = ((trend_raw - data.mean[flow_idx]) / data.std[flow_idx]).reshape(-1, 1).astype(np.float32)
            residual = (traffic - trend).astype(np.float32)
            x_basic_parts.append(np.concatenate([traffic, time_window], axis=1))
            x_wave_parts.append(np.concatenate([traffic, trend, residual, time_window], axis=1))
            y_parts.append(data.norm[end:target_end, flow_idx])
            flow_parts.append(flow_idx)
            end_parts.append(end)

    x_basic = np.stack(x_basic_parts).astype(np.float32)
    x_wave = np.stack(x_wave_parts).astype(np.float32)
    y = np.stack(y_parts).astype(np.float32)
    flow = np.asarray(flow_parts, dtype=np.int64)
    end = np.asarray(end_parts, dtype=np.int64)

    train_mask = end <= data.train_cut
    val_mask = (end > data.train_cut) & (end <= data.val_cut)
    test_mask = end > data.val_cut

    return WindowData(
        x_basic_train=x_basic[train_mask],
        x_wave_train=x_wave[train_mask],
        y_train=y[train_mask],
        flow_train=flow[train_mask],
        end_train=end[train_mask],
        x_basic_val=x_basic[val_mask],
        x_wave_val=x_wave[val_mask],
        y_val=y[val_mask],
        flow_val=flow[val_mask],
        end_val=end[val_mask],
        x_basic_test=x_basic[test_mask],
        x_wave_test=x_wave[test_mask],
        y_test=y[test_mask],
        flow_test=flow[test_mask],
        end_test=end[test_mask],
    )


def denormalize(values: np.ndarray, flows: np.ndarray, data: SplitData) -> np.ndarray:
    return values * data.std[flows, None] + data.mean[flows, None]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    wape = float(np.sum(np.abs(error)) / np.sum(np.abs(y_true)) * 100.0)
    smape = float(
        np.mean(2.0 * np.abs(error) / (np.abs(y_true) + np.abs(y_pred) + 1.0e-8)) * 100.0
    )
    return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "sMAPE": smape}


def horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, model: str) -> list[dict[str, float | int | str]]:
    rows = []
    for h in range(y_true.shape[1]):
        row = metrics(y_true[:, h : h + 1], y_pred[:, h : h + 1])
        row["horizon"] = h + 1
        row["model"] = model
        rows.append(row)
    return rows


def persistence_predict(x: np.ndarray, horizon: int) -> np.ndarray:
    last_value = x[:, -1, 0:1]
    return np.repeat(last_value, horizon, axis=1)


def seasonal_naive_predict(
    data: SplitData,
    flows: np.ndarray,
    ends: np.ndarray,
    y_shape: tuple[int, int],
    cfg: Config,
) -> np.ndarray:
    period = 288
    preds = np.zeros(y_shape, dtype=np.float32)
    for i, (flow_idx, end) in enumerate(zip(flows, ends)):
        for h in range(cfg.horizon):
            source_idx = end + h - period
            if source_idx >= 0:
                preds[i, h] = data.norm[source_idx, flow_idx]
            else:
                preds[i, h] = data.norm[end - 1, flow_idx]
    return preds


def fit_damped_alphas(data: SplitData) -> np.ndarray:
    alphas = np.zeros(data.norm.shape[1], dtype=np.float32)
    grid = np.linspace(-1.0, 1.0, 401)
    for flow_idx in range(data.norm.shape[1]):
        y = data.norm[2 : data.train_cut, flow_idx]
        last = data.norm[1 : data.train_cut - 1, flow_idx]
        diff = data.norm[1 : data.train_cut - 1, flow_idx] - data.norm[: data.train_cut - 2, flow_idx]
        losses = [np.mean((last + alpha * diff - y) ** 2) for alpha in grid]
        alphas[flow_idx] = float(grid[int(np.argmin(losses))])
    return alphas


def damped_persistence_predict(
    data: SplitData,
    flows: np.ndarray,
    ends: np.ndarray,
    y_shape: tuple[int, int],
    alphas: np.ndarray,
) -> np.ndarray:
    preds = np.zeros(y_shape, dtype=np.float32)
    for i, (flow_idx, end) in enumerate(zip(flows, ends)):
        prev2 = data.norm[end - 2, flow_idx]
        prev1 = data.norm[end - 1, flow_idx]
        alpha = alphas[flow_idx]
        for h in range(y_shape[1]):
            pred = prev1 + alpha * (prev1 - prev2)
            preds[i, h] = pred
            prev2, prev1 = prev1, pred
    return preds


def tabular_features(x: np.ndarray) -> np.ndarray:
    traffic = x[:, :, 0]
    calendar = x[:, -1, 1:]
    stats = np.column_stack(
        [
            traffic[:, -1],
            traffic[:, -3:].mean(axis=1),
            traffic[:, -12:].mean(axis=1),
            traffic[:, -36:].mean(axis=1),
            traffic[:, -72:].mean(axis=1),
            traffic[:, -12:].std(axis=1),
            traffic[:, -1] - traffic[:, -2],
            traffic[:, -1] - traffic[:, -12],
        ]
    )
    selected_lags = traffic[:, [-1, -2, -3, -6, -12, -24, -36, -48, -60, -72]]
    return np.concatenate([selected_lags, stats, calendar], axis=1).astype(np.float32)


def train_ridge(windows: WindowData) -> tuple[np.ndarray, np.ndarray]:
    model = Ridge(alpha=2.0)
    model.fit(tabular_features(windows.x_basic_train), windows.y_train)
    val_pred = model.predict(tabular_features(windows.x_basic_val)).astype(np.float32)
    test_pred = model.predict(tabular_features(windows.x_basic_test)).astype(np.float32)
    return val_pred, test_pred


def train_hgb(windows: WindowData) -> tuple[np.ndarray, np.ndarray]:
    base = HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.06,
        l2_regularization=0.03,
        max_leaf_nodes=31,
        random_state=42,
    )
    model = MultiOutputRegressor(base)
    model.fit(tabular_features(windows.x_basic_train), windows.y_train)
    val_pred = model.predict(tabular_features(windows.x_basic_val)).astype(np.float32)
    test_pred = model.predict(tabular_features(windows.x_basic_test)).astype(np.float32)
    return val_pred, test_pred


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden: int, horizon: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class WaveletResidualTCN(nn.Module):
    def __init__(self, input_dim: int, channels: int, horizon: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            TCNBlock(channels, dilation=1, kernel_size=3, dropout=0.08),
            TCNBlock(channels, dilation=2, kernel_size=3, dropout=0.08),
            TCNBlock(channels, dilation=4, kernel_size=3, dropout=0.08),
            TCNBlock(channels, dilation=8, kernel_size=3, dropout=0.08),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(channels, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last_raw = x[:, -1, 0:1]
        z = self.input_projection(x.transpose(1, 2))
        z = self.blocks(z).transpose(1, 2)
        residual_forecast = self.head(z[:, -1])
        return last_raw + residual_forecast


def train_torch_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    cfg: Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    test_x = torch.from_numpy(x_test).to(device)

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1.0e-4)
    loss_fn = nn.HuberLoss(delta=0.75)

    best_state = None
    best_val = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())
        history.append({"epoch": epoch, "train_loss": train_loss / n_seen, "val_loss": val_loss})
        if val_loss < best_val - 1.0e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    val_preds = []
    with torch.no_grad():
        for start in range(0, len(val_x), cfg.batch_size):
            xb = val_x[start : start + cfg.batch_size]
            val_preds.append(model(xb).detach().cpu().numpy())

    preds = []
    with torch.no_grad():
        for start in range(0, len(test_x), cfg.batch_size):
            xb = test_x[start : start + cfg.batch_size]
            preds.append(model(xb).detach().cpu().numpy())
    summary = {
        "best_val_huber": best_val,
        "epochs_run": float(len(history)),
        "final_train_huber": history[-1]["train_loss"],
    }
    return (
        np.concatenate(val_preds, axis=0).astype(np.float32),
        np.concatenate(preds, axis=0).astype(np.float32),
        summary,
    )


def search_ensemble(
    val_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    data: SplitData,
    windows: WindowData,
) -> tuple[str, np.ndarray, np.ndarray, dict[str, float]]:
    candidates = [
        name
        for name in [
            "Damped persistence",
            "HistGradientBoosting",
            "Ridge lags",
            "Residual TCN (no wavelet)",
            "Wavelet-Residual TCN",
            "Compact LSTM",
        ]
        if name in val_predictions
    ]
    y_val_raw = denormalize(windows.y_val, windows.flow_val, data)
    best_name = ""
    best_val_wape = float("inf")
    best_val_pred = None
    best_test_pred = None
    best_weights: dict[str, float] = {}

    def simplex_compositions(parts: int, total: int) -> list[list[int]]:
        if parts == 1:
            return [[total]]
        rows = []
        for value in range(total + 1):
            for suffix in simplex_compositions(parts - 1, total - value):
                rows.append([value, *suffix])
        return rows

    units = 20
    for scaled in simplex_compositions(len(candidates), units):
        weights = {name: value / units for name, value in zip(candidates, scaled)}
        if sum(weight > 0 for weight in weights.values()) < 2:
            continue
        val_mix = sum(weights[name] * val_predictions[name] for name in candidates)
        val_mix_raw = denormalize(val_mix, windows.flow_val, data)
        val_wape = metrics(y_val_raw, val_mix_raw)["WAPE"]
        if val_wape < best_val_wape:
            best_val_wape = val_wape
            best_val_pred = val_mix.astype(np.float32)
            best_test_pred = sum(weights[name] * test_predictions[name] for name in candidates).astype(np.float32)
            best_weights = weights

    active = ", ".join(f"{name}={weight:.2f}" for name, weight in best_weights.items() if weight > 0.0)
    best_name = f"Proposed WRTCN-GB Ensemble ({active})"
    assert best_val_pred is not None and best_test_pred is not None
    return best_name, best_val_pred, best_test_pred, {"validation_WAPE": best_val_wape, **best_weights}


def bootstrap_wape_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ends: np.ndarray,
    seed: int,
    n_bootstrap: int = 500,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_ends = np.unique(ends)
    samples = []
    for _ in range(n_bootstrap):
        sampled_ends = rng.choice(unique_ends, size=len(unique_ends), replace=True)
        mask = np.isin(ends, sampled_ends)
        samples.append(metrics(y_true[mask], y_pred[mask])["WAPE"])
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def make_plots(
    data: SplitData,
    windows: WindowData,
    results_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    predictions_raw: dict[str, np.ndarray],
    cfg: Config,
) -> None:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    sample_points = min(864, len(data.raw))
    for j, col in enumerate(data.columns[:3]):
        ax.plot(data.times.iloc[:sample_points], data.raw[:sample_points, j], linewidth=1.1, label=col)
    ax.set_title("Sample Abilene OD traffic traces")
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic volume")
    ax.legend(loc="upper right", frameon=True)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "traffic_sample.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ordered_models = results_df.sort_values("WAPE")["model"].tolist()
    for model_name in ordered_models:
        subset = horizon_df[horizon_df["model"] == model_name]
        ax.plot(subset["horizon"], subset["WAPE"], marker="o", linewidth=1.6, label=model_name)
    ax.set_title("Forecast error by 5-minute horizon")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("WAPE (%)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "wape_by_horizon.png", dpi=220)
    plt.close(fig)

    best_model = results_df.iloc[0]["model"]
    target_flow = 0
    mask = windows.flow_test == target_flow
    chosen = np.where(mask)[0][:96]
    y_true = denormalize(windows.y_test[chosen], windows.flow_test[chosen], data)[:, 0]
    y_pred = predictions_raw[best_model][chosen, 0]
    end_indices = windows.end_test[chosen]
    plot_times = data.times.iloc[end_indices].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(plot_times, y_true, label="Actual", linewidth=1.8)
    ax.plot(plot_times, y_pred, label=f"{best_model} one-step forecast", linewidth=1.5)
    ax.set_title(f"One-step forecast trace for {data.columns[target_flow]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Traffic volume")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "prediction_trace.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axis("off")
    boxes = [
        ("Raw OD flow", 0.04, 0.62),
        ("DWT trend", 0.25, 0.78),
        ("DWT residual", 0.25, 0.46),
        ("Temporal features", 0.25, 0.14),
        ("Causal TCN blocks", 0.52, 0.54),
        ("Residual horizon head", 0.76, 0.54),
    ]
    for text, x, y in boxes:
        rect = plt.Rectangle((x, y), 0.18, 0.18, fill=False, linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.09, text, ha="center", va="center", fontsize=9)
    arrows = [
        ((0.22, 0.71), (0.25, 0.87)),
        ((0.22, 0.71), (0.25, 0.55)),
        ((0.22, 0.71), (0.52, 0.63)),
        ((0.43, 0.87), (0.52, 0.63)),
        ((0.43, 0.55), (0.52, 0.63)),
        ((0.43, 0.23), (0.52, 0.63)),
        ((0.70, 0.63), (0.76, 0.63)),
    ]
    for xy1, xy2 in arrows:
        ax.annotate("", xy=xy2, xytext=xy1, arrowprops=dict(arrowstyle="->", linewidth=1.2))
    ax.set_title("Wavelet-Residual TCN pipeline")
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "method_pipeline.png", dpi=220)
    plt.close(fig)


def run(cfg: Config) -> None:
    start_time = time.time()
    set_seed(cfg.seed)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()

    data = load_data(cfg)
    windows = build_windows(data, cfg)
    split_summary = {
        "train_windows": int(len(windows.y_train)),
        "val_windows": int(len(windows.y_val)),
        "test_windows": int(len(windows.y_test)),
        "selected_flows": data.columns,
        "train_cut_row": int(data.train_cut),
        "val_cut_row": int(data.val_cut),
        "wavelet_features": "computed causally inside each lookback window",
    }

    normalized_val_predictions: dict[str, np.ndarray] = {}
    normalized_predictions: dict[str, np.ndarray] = {}
    training_notes: dict[str, dict[str, float] | str] = {}

    normalized_val_predictions["Persistence"] = persistence_predict(windows.x_basic_val, cfg.horizon)
    normalized_predictions["Persistence"] = persistence_predict(windows.x_basic_test, cfg.horizon)
    normalized_val_predictions["Seasonal naive"] = seasonal_naive_predict(
        data, windows.flow_val, windows.end_val, windows.y_val.shape, cfg
    )
    normalized_predictions["Seasonal naive"] = seasonal_naive_predict(
        data, windows.flow_test, windows.end_test, windows.y_test.shape, cfg
    )
    damped_alphas = fit_damped_alphas(data)
    normalized_val_predictions["Damped persistence"] = damped_persistence_predict(
        data, windows.flow_val, windows.end_val, windows.y_val.shape, damped_alphas
    )
    normalized_predictions["Damped persistence"] = damped_persistence_predict(
        data, windows.flow_test, windows.end_test, windows.y_test.shape, damped_alphas
    )
    training_notes["Damped persistence"] = {
        "alpha_mean": float(np.mean(damped_alphas)),
        "alpha_min": float(np.min(damped_alphas)),
        "alpha_max": float(np.max(damped_alphas)),
    }
    ridge_val, ridge_test = train_ridge(windows)
    normalized_val_predictions["Ridge lags"] = ridge_val
    normalized_predictions["Ridge lags"] = ridge_test
    hgb_val, hgb_test = train_hgb(windows)
    normalized_val_predictions["HistGradientBoosting"] = hgb_val
    normalized_predictions["HistGradientBoosting"] = hgb_test

    lstm_val, lstm_preds, lstm_notes = train_torch_model(
        LSTMForecaster(input_dim=windows.x_basic_train.shape[2], hidden=48, horizon=cfg.horizon),
        windows.x_basic_train,
        windows.y_train,
        windows.x_basic_val,
        windows.y_val,
        windows.x_basic_test,
        cfg,
        device,
    )
    normalized_val_predictions["Compact LSTM"] = lstm_val
    normalized_predictions["Compact LSTM"] = lstm_preds
    training_notes["Compact LSTM"] = lstm_notes

    basic_tcn_val, basic_tcn_preds, basic_tcn_notes = train_torch_model(
        WaveletResidualTCN(input_dim=windows.x_basic_train.shape[2], channels=64, horizon=cfg.horizon),
        windows.x_basic_train,
        windows.y_train,
        windows.x_basic_val,
        windows.y_val,
        windows.x_basic_test,
        cfg,
        device,
    )
    normalized_val_predictions["Residual TCN (no wavelet)"] = basic_tcn_val
    normalized_predictions["Residual TCN (no wavelet)"] = basic_tcn_preds
    training_notes["Residual TCN (no wavelet)"] = basic_tcn_notes

    tcn_val, tcn_preds, tcn_notes = train_torch_model(
        WaveletResidualTCN(input_dim=windows.x_wave_train.shape[2], channels=64, horizon=cfg.horizon),
        windows.x_wave_train,
        windows.y_train,
        windows.x_wave_val,
        windows.y_val,
        windows.x_wave_test,
        cfg,
        device,
    )
    normalized_val_predictions["Wavelet-Residual TCN"] = tcn_val
    normalized_predictions["Wavelet-Residual TCN"] = tcn_preds
    training_notes["Wavelet-Residual TCN"] = tcn_notes

    ensemble_name, ensemble_val, ensemble_test, ensemble_notes = search_ensemble(
        normalized_val_predictions, normalized_predictions, data, windows
    )
    normalized_val_predictions[ensemble_name] = ensemble_val
    normalized_predictions[ensemble_name] = ensemble_test
    training_notes[ensemble_name] = ensemble_notes

    raw_predictions: dict[str, np.ndarray] = {}
    y_true_raw = denormalize(windows.y_test, windows.flow_test, data)
    rows = []
    horizon_rows = []
    for model_name, pred_norm in normalized_predictions.items():
        pred_raw = denormalize(pred_norm, windows.flow_test, data)
        raw_predictions[model_name] = pred_raw
        row = metrics(y_true_raw, pred_raw)
        ci_low, ci_high = bootstrap_wape_ci(y_true_raw, pred_raw, windows.end_test, cfg.seed)
        row["WAPE_CI95_low"] = ci_low
        row["WAPE_CI95_high"] = ci_high
        row["model"] = model_name
        rows.append(row)
        horizon_rows.extend(horizon_metrics(y_true_raw, pred_raw, model_name))

    results_df = pd.DataFrame(rows).sort_values("WAPE").reset_index(drop=True)
    horizon_df = pd.DataFrame(horizon_rows)
    results_df.to_csv(cfg.results_dir / "summary_metrics.csv", index=False)
    horizon_df.to_csv(cfg.results_dir / "horizon_metrics.csv", index=False)

    make_plots(data, windows, results_df, horizon_df, raw_predictions, cfg)

    best = results_df.iloc[0].to_dict()
    baseline = results_df[results_df["model"] == "Persistence"].iloc[0].to_dict()
    best_gain = (baseline["WAPE"] - best["WAPE"]) / baseline["WAPE"] * 100.0
    metadata = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "device": str(device),
        "split_summary": split_summary,
        "training_notes": training_notes,
        "best_model": best,
        "persistence_wape_reduction_percent": best_gain,
        "runtime_seconds": time.time() - start_time,
    }
    with (cfg.results_dir / "experiment_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(results_df.to_string(index=False))
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sampled Internet traffic forecasting experiments.")
    parser.add_argument("--sample-rows", type=int, default=Config.sample_rows)
    parser.add_argument("--top-flows", type=int, default=Config.top_flows)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--stride", type=int, default=Config.stride)
    args = parser.parse_args()
    cfg = Config(
        sample_rows=args.sample_rows,
        top_flows=args.top_flows,
        epochs=args.epochs,
        stride=args.stride,
    )
    run(cfg)


if __name__ == "__main__":
    main()

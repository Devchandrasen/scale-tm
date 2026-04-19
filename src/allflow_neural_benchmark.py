from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.multidataset_benchmark import (
    DATASETS,
    FOLDS,
    Config as MultiConfig,
    damped_predict,
    denormalize,
    fit_damped_alphas,
    fit_ridge,
    load_dataset,
    make_windows,
    metrics,
    moving_average,
    patch_features,
    persistence_predict,
    seasonal_naive_predict,
    time_features,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    sample_rows: int = 2880
    datasets: tuple[str, ...] = ("Abilene", "CERNET")
    lookback_hours: float = 6.0
    horizon_hours: float = 1.0
    stride_hours: float = 1.0
    train_end: float = 0.70
    val_end: float = 0.80
    test_end: float = 1.00
    batch_size: int = 512
    epochs: int = 5
    patience: int = 2
    learning_rate: float = 1.0e-3
    hidden: int = 48
    seed: int = 42
    bootstrap_samples: int = 250
    ensemble_step: float = 0.10
    blocked_folds: bool = False
    output_prefix: str = "allflow_neural_benchmark"
    results_dir: Path = ROOT / "results"
    figures_dir: Path = ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="All-flow neural benchmark for public OD traffic matrices.")
    parser.add_argument("--sample-rows", type=int, default=Config.sample_rows)
    parser.add_argument("--datasets", type=str, default=",".join(Config.datasets))
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--bootstrap-samples", type=int, default=Config.bootstrap_samples)
    parser.add_argument("--ensemble-step", type=float, default=Config.ensemble_step)
    parser.add_argument("--output-prefix", type=str, default=Config.output_prefix)
    parser.add_argument(
        "--blocked-folds",
        action="store_true",
        help="Use the three blocked chronological folds from the all-flow stress test.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def select_device() -> torch.device:
    # The local GTX 1050 has compute capability 6.1, while the installed
    # PyTorch build supports sm_70+. Keep this protocol CPU-stable.
    return torch.device("cpu")


def wavelet_smooth_row(row: np.ndarray, wavelet: str = "db2", level: int = 2) -> np.ndarray:
    row = np.nan_to_num(row.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.std(row)) < 1.0e-8:
        return row
    level = min(level, pywt.dwt_max_level(len(row), pywt.Wavelet(wavelet).dec_len))
    if level < 1:
        return row.astype(np.float32)
    coeffs = pywt.wavedec(row, wavelet=wavelet, mode="periodization", level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    threshold = sigma * math.sqrt(2.0 * math.log(max(len(row), 2)))
    if not np.isfinite(threshold) or threshold <= 0.0:
        smooth_coeffs = coeffs
    else:
        smooth_coeffs = [coeffs[0]]
        smooth_coeffs.extend(np.nan_to_num(pywt.threshold(c, threshold, mode="soft")) for c in coeffs[1:])
    smoothed = pywt.waverec(smooth_coeffs, wavelet=wavelet, mode="periodization")
    return np.nan_to_num(smoothed[: len(row)], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def wavelet_smooth_batch(x: np.ndarray) -> np.ndarray:
    trend = np.empty_like(x, dtype=np.float32)
    for idx in range(len(x)):
        trend[idx] = wavelet_smooth_row(x[idx])
    return trend


def wave_inputs(x: np.ndarray, ends: np.ndarray, tf: np.ndarray) -> np.ndarray:
    trend = wavelet_smooth_batch(x)
    residual = x - trend
    lookback = x.shape[1]
    calendar = np.stack([tf[end - lookback : end] for end in ends], axis=0).astype(np.float32)
    return np.concatenate([x[:, :, None], trend[:, :, None], residual[:, :, None], calendar], axis=2)


def sequence_inputs(x: np.ndarray) -> np.ndarray:
    return x[:, :, None].astype(np.float32)


def apply_constant_flow_guard(pred: np.ndarray, flows: np.ndarray, constant_mask: np.ndarray) -> np.ndarray:
    guarded = pred.copy()
    mask = constant_mask[flows]
    if mask.any():
        guarded[mask] = 0.0
    return guarded


def bootstrap_wape_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ends: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    unique_ends = np.unique(ends)
    abs_error = np.zeros(len(unique_ends), dtype=np.float64)
    abs_true = np.zeros(len(unique_ends), dtype=np.float64)
    for idx, end in enumerate(unique_ends):
        mask = ends == end
        abs_error[idx] = np.sum(np.abs(y_pred[mask] - y_true[mask]))
        abs_true[idx] = np.sum(np.abs(y_true[mask]))
    values = np.zeros(samples, dtype=np.float64)
    for i in range(samples):
        chosen = rng.integers(0, len(unique_ends), len(unique_ends))
        values[i] = np.sum(abs_error[chosen]) / max(np.sum(abs_true[chosen]), 1.0e-8) * 100.0
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def paired_wape_delta_ci(
    y_true: np.ndarray,
    candidate_pred: np.ndarray,
    baseline_pred: np.ndarray,
    ends: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float, float]:
    """Return baseline minus candidate WAPE in percentage points with a paired CI."""
    unique_ends = np.unique(ends)
    candidate_error = np.zeros(len(unique_ends), dtype=np.float64)
    baseline_error = np.zeros(len(unique_ends), dtype=np.float64)
    abs_true = np.zeros(len(unique_ends), dtype=np.float64)
    for idx, end in enumerate(unique_ends):
        mask = ends == end
        candidate_error[idx] = np.sum(np.abs(candidate_pred[mask] - y_true[mask]))
        baseline_error[idx] = np.sum(np.abs(baseline_pred[mask] - y_true[mask]))
        abs_true[idx] = np.sum(np.abs(y_true[mask]))
    denominator = max(float(np.sum(abs_true)), 1.0e-8)
    point = (float(np.sum(baseline_error)) - float(np.sum(candidate_error))) / denominator * 100.0
    values = np.zeros(samples, dtype=np.float64)
    for i in range(samples):
        chosen = rng.integers(0, len(unique_ends), len(unique_ends))
        sampled_denominator = max(float(np.sum(abs_true[chosen])), 1.0e-8)
        values[i] = (
            float(np.sum(baseline_error[chosen])) - float(np.sum(candidate_error[chosen]))
        ) / sampled_denominator * 100.0
    return point, float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


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
        groups = 4 if channels % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AllFlowWRTCN(nn.Module):
    def __init__(self, input_dim: int, channels: int, horizon: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            TCNBlock(channels, 1, 3, 0.08),
            TCNBlock(channels, 2, 3, 0.08),
            TCNBlock(channels, 4, 3, 0.08),
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
        return last_raw + self.head(z[:, -1])


class PatchTSTSmall(nn.Module):
    def __init__(self, lookback: int, horizon: int, patch_len: int, d_model: int) -> None:
        super().__init__()
        self.patch_len = max(2, min(patch_len, lookback))
        self.stride = self.patch_len
        self.n_patches = max(1, (lookback - self.patch_len) // self.stride + 1)
        self.proj = nn.Linear(self.patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        series = x[:, :, 0]
        last = series[:, -1:]
        patches = series.unfold(dimension=1, size=self.patch_len, step=self.stride)
        if patches.shape[1] > self.n_patches:
            patches = patches[:, -self.n_patches :]
        z = self.proj(patches) + self.pos[:, : patches.shape[1]]
        z = self.encoder(z)
        return last + self.head(z.mean(dim=1))


class NBeatsBlock(nn.Module):
    def __init__(self, lookback: int, horizon: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.backcast = nn.Linear(hidden, lookback)
        self.forecast = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.backcast(h), self.forecast(h)


class NBeatsSmall(nn.Module):
    def __init__(self, lookback: int, horizon: int, hidden: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([NBeatsBlock(lookback, horizon, hidden) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        series = x[:, :, 0]
        last = series[:, -1:]
        residual = series
        forecast = torch.zeros(series.shape[0], self.blocks[0].forecast.out_features, device=series.device)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return last + forecast


class TimesNetSmall(nn.Module):
    def __init__(self, lookback: int, horizon: int, hidden: int) -> None:
        super().__init__()
        candidate_periods = [max(2, horizon // 2), horizon, min(lookback, horizon * 2)]
        self.periods = sorted({p for p in candidate_periods if 2 <= p <= lookback})
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, hidden, kernel_size=(1, 3), padding=(0, 1)),
                    nn.GELU(),
                    nn.Conv2d(hidden, hidden, kernel_size=(3, 3), padding=(1, 1)),
                    nn.GELU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                for _ in self.periods
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * len(self.periods)),
            nn.Linear(hidden * len(self.periods), hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        series = x[:, :, 0]
        last = series[:, -1:]
        features = []
        for period, conv in zip(self.periods, self.convs):
            pad = (period - series.shape[1] % period) % period
            if pad:
                padded = torch.cat([series, series[:, -1:].expand(-1, pad)], dim=1)
            else:
                padded = series
            z = padded.reshape(series.shape[0], -1, period).unsqueeze(1)
            features.append(conv(z).flatten(1))
        return last + self.head(torch.cat(features, dim=1))


class NHiTSBlock(nn.Module):
    def __init__(self, lookback: int, horizon: int, hidden: int, pool_size: int, knots: int) -> None:
        super().__init__()
        self.pool_size = max(1, pool_size)
        pooled_len = math.ceil(lookback / self.pool_size)
        self.net = nn.Sequential(
            nn.Linear(pooled_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.backcast = nn.Linear(hidden, lookback)
        self.forecast_knots = nn.Linear(hidden, knots)
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pool_size > 1:
            pad = (self.pool_size - x.shape[1] % self.pool_size) % self.pool_size
            pooled_input = torch.cat([x, x[:, -1:].expand(-1, pad)], dim=1) if pad else x
            pooled = torch.nn.functional.avg_pool1d(
                pooled_input.unsqueeze(1), kernel_size=self.pool_size, stride=self.pool_size
            ).squeeze(1)
        else:
            pooled = x
        h = self.net(pooled)
        knots = self.forecast_knots(h).unsqueeze(1)
        forecast = torch.nn.functional.interpolate(
            knots, size=self.horizon, mode="linear", align_corners=True
        ).squeeze(1)
        return self.backcast(h), forecast


class NHiTSSmall(nn.Module):
    def __init__(self, lookback: int, horizon: int, hidden: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                NHiTSBlock(lookback, horizon, hidden, pool_size=1, knots=horizon),
                NHiTSBlock(lookback, horizon, hidden, pool_size=2, knots=max(2, horizon // 2)),
                NHiTSBlock(lookback, horizon, hidden, pool_size=4, knots=max(2, horizon // 4)),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        series = x[:, :, 0]
        last = series[:, -1:]
        residual = series
        forecast = torch.zeros(series.shape[0], self.blocks[0].horizon, device=series.device)
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return last + forecast


class GraphAttentionSmall(nn.Module):
    def __init__(self, n_flows: int, lookback: int, horizon: int, hidden: int) -> None:
        super().__init__()
        self.temporal = nn.Linear(lookback, hidden)
        self.flow_embedding = nn.Embedding(n_flows, hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, dropout=0.10, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden * 2, hidden),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, :, -1:]
        flow_ids = torch.arange(x.shape[1], device=x.device)
        z = self.temporal(x) + self.flow_embedding(flow_ids).unsqueeze(0)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        z = z + attn_out
        z = z + self.ffn(z)
        return last + self.head(z)


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
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())
        history.append({"epoch": epoch, "train_huber": train_loss / max(n_seen, 1), "val_huber": val_loss})
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

    def predict(x: np.ndarray) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(x), cfg.batch_size):
                xb = torch.from_numpy(x[start : start + cfg.batch_size]).to(device)
                preds.append(model(xb).cpu().numpy())
        return np.concatenate(preds, axis=0).astype(np.float32)

    return predict(x_val), predict(x_test), {
        "best_val_huber": best_val,
        "epochs_run": float(len(history)),
        "final_train_huber": history[-1]["train_huber"] if history else float("nan"),
    }


def matrix_windows(
    bundle,
    windows,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    norm = ((bundle.raw - windows.mean) / windows.std).astype(np.float32)

    def build(ends: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        unique_ends = np.unique(ends)
        x_matrix = np.stack([norm[end - bundle.lookback : end].T for end in unique_ends], axis=0)
        y_matrix = np.stack([norm[end : end + bundle.horizon].T for end in unique_ends], axis=0)
        return x_matrix.astype(np.float32), y_matrix.astype(np.float32), unique_ends.astype(np.int32)

    x_train, y_train, train_ends = build(windows.end_train)
    x_val, y_val, val_ends = build(windows.end_val)
    x_test, y_test, test_ends = build(windows.end_test)
    return x_train, y_train, val_ends, x_val, y_val, test_ends, x_test, y_test


def flatten_matrix_predictions(
    matrix_pred: np.ndarray,
    unique_ends: np.ndarray,
    flat_ends: np.ndarray,
    flat_flows: np.ndarray,
) -> np.ndarray:
    end_to_idx = {int(end): idx for idx, end in enumerate(unique_ends)}
    rows = np.asarray([end_to_idx[int(end)] for end in flat_ends], dtype=np.int64)
    return matrix_pred[rows, flat_flows].astype(np.float32)


def train_graph_model(
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
    loader = DataLoader(train_ds, batch_size=min(32, cfg.batch_size), shuffle=True, drop_last=False)
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
            n_seen += len(xb)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())
        history.append({"epoch": epoch, "train_huber": train_loss / max(n_seen, 1), "val_huber": val_loss})
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

    def predict(x: np.ndarray) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(x), min(32, cfg.batch_size)):
                xb = torch.from_numpy(x[start : start + min(32, cfg.batch_size)]).to(device)
                preds.append(model(xb).cpu().numpy())
        return np.concatenate(preds, axis=0).astype(np.float32)

    return predict(x_val), predict(x_test), {
        "best_val_huber": best_val,
        "epochs_run": float(len(history)),
        "final_train_huber": history[-1]["train_huber"] if history else float("nan"),
    }


def simplex_weights(n_models: int, step: float) -> list[np.ndarray]:
    units = int(round(1.0 / step))
    rows: list[np.ndarray] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append(np.asarray(prefix + [remaining], dtype=np.float32) / units)
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots - 1)

    rec([], units, n_models)
    return rows


def choose_ensemble(
    names: list[str],
    val_raw: dict[str, np.ndarray],
    test_norm: dict[str, np.ndarray],
    y_val_raw: np.ndarray,
    step: float,
) -> tuple[np.ndarray, dict[str, float]]:
    best_score = float("inf")
    best_weights = np.zeros(len(names), dtype=np.float32)
    for weights in simplex_weights(len(names), step):
        pred = np.zeros_like(y_val_raw, dtype=np.float32)
        for weight, name in zip(weights, names):
            if weight > 0:
                pred += weight * val_raw[name]
        score = metrics(y_val_raw, pred)["WAPE"]
        if score < best_score:
            best_score = score
            best_weights = weights
    mixed = np.zeros_like(test_norm[names[0]], dtype=np.float32)
    for weight, name in zip(best_weights, names):
        if weight > 0:
            mixed += weight * test_norm[name]
    return mixed, {name: float(weight) for name, weight in zip(names, best_weights) if weight > 0}


def train_linear_controls(bundle, windows, cfg: Config) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    norm = ((bundle.raw - windows.mean) / windows.std).astype(np.float32)
    alphas = fit_damped_alphas(norm, windows.train_cut, 81)

    last_train = windows.x_train[:, -1:]
    nlinear = fit_ridge(windows.x_train - last_train, windows.y_train - last_train, 1.0)
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
    dlinear = Ridge(alpha=1.0, fit_intercept=False)
    dlinear.fit(dlinear_train, windows.y_train)

    patch_train = patch_features(windows.x_train, bundle.horizon)
    patch_val = patch_features(windows.x_val, bundle.horizon)
    patch_test = patch_features(windows.x_test, bundle.horizon)
    patch_linear = Ridge(alpha=1.0, fit_intercept=False)
    patch_linear.fit(patch_train, windows.y_train)

    val = {
        "NLinear": nlinear_val,
        "DLinear": dlinear.predict(dlinear_val).astype(np.float32),
        "PatchLinear": patch_linear.predict(patch_val).astype(np.float32),
        "Damped persistence": damped_predict(windows.x_val, windows.flow_val, alphas, bundle.horizon),
        "Persistence": persistence_predict(windows.x_val, bundle.horizon),
        "Seasonal naive": seasonal_naive_predict(
            norm, windows.x_val, windows.flow_val, windows.end_val, bundle.horizon, bundle.daily_period
        ),
    }
    test = {
        "NLinear": nlinear_test,
        "DLinear": dlinear.predict(dlinear_test).astype(np.float32),
        "PatchLinear": patch_linear.predict(patch_test).astype(np.float32),
        "Damped persistence": damped_predict(windows.x_test, windows.flow_test, alphas, bundle.horizon),
        "Persistence": persistence_predict(windows.x_test, bundle.horizon),
        "Seasonal naive": seasonal_naive_predict(
            norm, windows.x_test, windows.flow_test, windows.end_test, bundle.horizon, bundle.daily_period
        ),
    }
    return val, test


def evaluate_dataset(
    name: str,
    cfg: Config,
    device: torch.device,
    rng: np.random.Generator,
    fold: dict[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    multi_cfg = MultiConfig(
        sample_rows=cfg.sample_rows,
        lookback_hours=cfg.lookback_hours,
        horizon_hours=cfg.horizon_hours,
        stride_hours=cfg.stride_hours,
    )
    bundle = load_dataset(name, DATASETS[name], multi_cfg)
    windows = make_windows(bundle, fold)
    constant_mask = bundle.raw[: windows.train_cut].std(axis=0) < 1.0e-6
    tf = time_features(bundle.times)

    print(
        f"{name} fold {int(fold['fold'])}: rows={bundle.raw.shape[0]}, flows={bundle.raw.shape[1]}, "
        f"L={bundle.lookback}, H={bundle.horizon}, train={len(windows.x_train)}"
    )

    val_preds, test_preds = train_linear_controls(bundle, windows, cfg)

    seq_train = sequence_inputs(windows.x_train)
    seq_val = sequence_inputs(windows.x_val)
    seq_test = sequence_inputs(windows.x_test)

    print(f"  training PatchTST-Small on {name}")
    patch_model = PatchTSTSmall(bundle.lookback, bundle.horizon, patch_len=bundle.horizon, d_model=cfg.hidden)
    val_preds["PatchTST-Small"], test_preds["PatchTST-Small"], patch_summary = train_torch_model(
        patch_model, seq_train, windows.y_train, seq_val, windows.y_val, seq_test, cfg, device
    )

    print(f"  training N-BEATS-Small on {name}")
    nbeats_model = NBeatsSmall(bundle.lookback, bundle.horizon, cfg.hidden * 2)
    val_preds["N-BEATS-Small"], test_preds["N-BEATS-Small"], nbeats_summary = train_torch_model(
        nbeats_model, seq_train, windows.y_train, seq_val, windows.y_val, seq_test, cfg, device
    )

    print(f"  training TimesNet-Small on {name}")
    times_model = TimesNetSmall(bundle.lookback, bundle.horizon, cfg.hidden)
    val_preds["TimesNet-Small"], test_preds["TimesNet-Small"], times_summary = train_torch_model(
        times_model, seq_train, windows.y_train, seq_val, windows.y_val, seq_test, cfg, device
    )

    print(f"  training N-HiTS-Small on {name}")
    nhits_model = NHiTSSmall(bundle.lookback, bundle.horizon, cfg.hidden * 2)
    val_preds["N-HiTS-Small"], test_preds["N-HiTS-Small"], nhits_summary = train_torch_model(
        nhits_model, seq_train, windows.y_train, seq_val, windows.y_val, seq_test, cfg, device
    )

    print(f"  building causal wavelet inputs for {name}")
    wave_train = wave_inputs(windows.x_train, windows.end_train, tf)
    wave_val = wave_inputs(windows.x_val, windows.end_val, tf)
    wave_test = wave_inputs(windows.x_test, windows.end_test, tf)

    print(f"  training All-flow WRTCN on {name}")
    wrtcn_model = AllFlowWRTCN(wave_train.shape[2], cfg.hidden, bundle.horizon)
    val_preds["All-flow WRTCN"], test_preds["All-flow WRTCN"], wrtcn_summary = train_torch_model(
        wrtcn_model, wave_train, windows.y_train, wave_val, windows.y_val, wave_test, cfg, device
    )

    print(f"  training GraphAttention-Small on {name}")
    (
        graph_x_train,
        graph_y_train,
        graph_val_ends,
        graph_x_val,
        graph_y_val,
        graph_test_ends,
        graph_x_test,
        graph_y_test,
    ) = matrix_windows(bundle, windows)
    graph_model = GraphAttentionSmall(bundle.raw.shape[1], bundle.lookback, bundle.horizon, cfg.hidden)
    graph_val_matrix, graph_test_matrix, graph_summary = train_graph_model(
        graph_model, graph_x_train, graph_y_train, graph_x_val, graph_y_val, graph_x_test, cfg, device
    )
    val_preds["GraphAttention-Small"] = flatten_matrix_predictions(
        graph_val_matrix, graph_val_ends, windows.end_val, windows.flow_val
    )
    test_preds["GraphAttention-Small"] = flatten_matrix_predictions(
        graph_test_matrix, graph_test_ends, windows.end_test, windows.flow_test
    )

    for pred_map, flows in [(val_preds, windows.flow_val), (test_preds, windows.flow_test)]:
        for model_name in list(pred_map):
            pred_map[model_name] = apply_constant_flow_guard(pred_map[model_name], flows, constant_mask)

    y_val_raw = denormalize(windows.y_val, windows.flow_val, windows)
    y_test_raw = denormalize(windows.y_test, windows.flow_test, windows)
    val_raw = {model: denormalize(pred, windows.flow_val, windows) for model, pred in val_preds.items()}

    ensemble_candidates = [
        "DLinear",
        "NLinear",
        "PatchLinear",
        "Damped persistence",
        "PatchTST-Small",
        "N-BEATS-Small",
        "TimesNet-Small",
        "N-HiTS-Small",
        "All-flow WRTCN",
        "GraphAttention-Small",
    ]
    ensemble_norm, ensemble_weights = choose_ensemble(
        ensemble_candidates, val_raw, test_preds, y_val_raw, cfg.ensemble_step
    )
    test_preds["SCALE-TM neural ensemble"] = apply_constant_flow_guard(
        ensemble_norm, windows.flow_test, constant_mask
    )

    no_wrtcn_candidates = [name for name in ensemble_candidates if name != "All-flow WRTCN"]
    no_wrtcn_norm, no_wrtcn_weights = choose_ensemble(
        no_wrtcn_candidates, val_raw, test_preds, y_val_raw, cfg.ensemble_step
    )
    test_preds["SCALE-TM no-WRTCN ensemble"] = apply_constant_flow_guard(
        no_wrtcn_norm, windows.flow_test, constant_mask
    )

    uniform_norm = np.zeros_like(test_preds[ensemble_candidates[0]], dtype=np.float32)
    for candidate_name in ensemble_candidates:
        uniform_norm += test_preds[candidate_name] / float(len(ensemble_candidates))
    test_preds["Uniform candidate ensemble"] = apply_constant_flow_guard(
        uniform_norm, windows.flow_test, constant_mask
    )

    train_summaries = {
        "PatchTST-Small": patch_summary,
        "N-BEATS-Small": nbeats_summary,
        "TimesNet-Small": times_summary,
        "N-HiTS-Small": nhits_summary,
        "All-flow WRTCN": wrtcn_summary,
        "GraphAttention-Small": graph_summary,
    }
    ensemble_weight_maps = {
        "SCALE-TM neural ensemble": ensemble_weights,
        "SCALE-TM no-WRTCN ensemble": no_wrtcn_weights,
        "Uniform candidate ensemble": {name: 1.0 / len(ensemble_candidates) for name in ensemble_candidates},
    }
    rows: list[dict[str, object]] = []
    test_raw_by_model: dict[str, np.ndarray] = {}
    metrics_by_model: dict[str, dict[str, float]] = {}
    for model_name, pred_norm in test_preds.items():
        pred_raw = denormalize(pred_norm, windows.flow_test, windows)
        test_raw_by_model[model_name] = pred_raw
        row = metrics(y_test_raw, pred_raw)
        metrics_by_model[model_name] = row
        ci_low, ci_high = bootstrap_wape_ci(
            y_test_raw, pred_raw, windows.end_test, rng, cfg.bootstrap_samples
        )
        row.update(
            {
                "dataset": name,
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
                "constant_flows": int(constant_mask.sum()),
                "ensemble_weights": json.dumps(ensemble_weight_maps.get(model_name, {}), sort_keys=True),
                "train_summary": json.dumps(train_summaries.get(model_name, {}), sort_keys=True),
            }
        )
        rows.append(row)

    primary = "SCALE-TM neural ensemble"
    single_model_names = [
        model_name
        for model_name in test_raw_by_model
        if "ensemble" not in model_name.lower()
        and model_name != "Uniform candidate ensemble"
    ]
    best_single = min(single_model_names, key=lambda model_name: metrics_by_model[model_name]["WAPE"])
    requested_baselines = [
        ("Best single non-ensemble", best_single),
        ("Persistence", "Persistence"),
        ("N-BEATS-Small", "N-BEATS-Small"),
        ("N-HiTS-Small", "N-HiTS-Small"),
        ("All-flow WRTCN", "All-flow WRTCN"),
        ("SCALE-TM no-WRTCN ensemble", "SCALE-TM no-WRTCN ensemble"),
        ("Uniform candidate ensemble", "Uniform candidate ensemble"),
    ]
    delta_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for baseline_label, baseline_model in requested_baselines:
        if baseline_model not in test_raw_by_model or (baseline_label, baseline_model) in seen:
            continue
        seen.add((baseline_label, baseline_model))
        delta, ci_low, ci_high = paired_wape_delta_ci(
            y_test_raw,
            test_raw_by_model[primary],
            test_raw_by_model[baseline_model],
            windows.end_test,
            rng,
            cfg.bootstrap_samples,
        )
        baseline_wape = metrics_by_model[baseline_model]["WAPE"]
        candidate_wape = metrics_by_model[primary]["WAPE"]
        delta_rows.append(
            {
                "dataset": name,
                "fold": int(fold["fold"]),
                "candidate": primary,
                "baseline": baseline_label,
                "baseline_model": baseline_model,
                "candidate_wape": candidate_wape,
                "baseline_wape": baseline_wape,
                "wape_point_improvement": delta,
                "wape_point_CI_low": ci_low,
                "wape_point_CI_high": ci_high,
                "relative_improvement_percent": delta / max(baseline_wape, 1.0e-8) * 100.0,
            }
        )
    return rows, delta_rows


def plot_neural_results(summary: pd.DataFrame, output_path: Path) -> None:
    preferred = [
        "SCALE-TM neural ensemble",
        "SCALE-TM no-WRTCN ensemble",
        "Uniform candidate ensemble",
        "All-flow WRTCN",
        "GraphAttention-Small",
        "PatchTST-Small",
        "N-BEATS-Small",
        "TimesNet-Small",
        "N-HiTS-Small",
        "DLinear",
        "NLinear",
        "PatchLinear",
        "Damped persistence",
        "Persistence",
        "Seasonal naive",
    ]
    datasets = summary["dataset"].unique().tolist()
    models = [m for m in preferred if m in set(summary["model"])]
    x = np.arange(len(datasets))
    width = min(0.08, 0.82 / max(len(models), 1))
    plt.figure(figsize=(12, 5.8))
    for idx, model in enumerate(models):
        vals = []
        for dataset in datasets:
            match = summary[(summary["dataset"] == dataset) & (summary["model"] == model)]
            vals.append(float(match["WAPE"].iloc[0]) if len(match) else np.nan)
        plt.bar(x + (idx - (len(models) - 1) / 2.0) * width, vals, width=width, label=model)
    plt.xticks(x, datasets)
    plt.ylabel("WAPE (%)")
    plt.xlabel("Dataset")
    plt.title("All-flow neural and modern baseline benchmark")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    cfg = Config(
        sample_rows=args.sample_rows,
        datasets=datasets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        bootstrap_samples=args.bootstrap_samples,
        ensemble_step=args.ensemble_step,
        blocked_folds=args.blocked_folds,
        output_prefix=args.output_prefix,
    )
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = select_device()
    started = time.time()
    print(f"device={device}")

    if cfg.blocked_folds:
        folds = FOLDS
    else:
        folds = ({"fold": 1, "train_end": cfg.train_end, "val_end": cfg.val_end, "test_end": cfg.test_end},)

    rows: list[dict[str, object]] = []
    paired_delta_rows: list[dict[str, object]] = []
    for dataset in cfg.datasets:
        for fold in folds:
            fold_rows, fold_delta_rows = evaluate_dataset(dataset, cfg, device, rng, fold)
            rows.extend(fold_rows)
            paired_delta_rows.extend(fold_delta_rows)

    results = pd.DataFrame(rows)
    results_path = cfg.results_dir / f"{cfg.output_prefix}.csv"
    results.to_csv(results_path, index=False)
    if cfg.blocked_folds:
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
                constant_flows=("constant_flows", "max"),
            )
            .sort_values(["dataset", "WAPE"])
            .reset_index(drop=True)
        )
    else:
        summary = results.sort_values(["dataset", "WAPE"]).reset_index(drop=True)
    summary_path = cfg.results_dir / f"{cfg.output_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    paired = pd.DataFrame(paired_delta_rows)
    paired_path = cfg.results_dir / f"{cfg.output_prefix}_paired_deltas.csv"
    paired.to_csv(paired_path, index=False)
    if cfg.blocked_folds:
        paired_summary = (
            paired.groupby(["dataset", "candidate", "baseline"], as_index=False)
            .agg(
                candidate_wape=("candidate_wape", "mean"),
                baseline_wape=("baseline_wape", "mean"),
                wape_point_improvement=("wape_point_improvement", "mean"),
                wape_point_CI_low=("wape_point_CI_low", "mean"),
                wape_point_CI_high=("wape_point_CI_high", "mean"),
                relative_improvement_percent=("relative_improvement_percent", "mean"),
                folds=("fold", "nunique"),
                baseline_models=("baseline_model", lambda values: "; ".join(sorted(set(values)))),
            )
            .sort_values(["dataset", "baseline"])
            .reset_index(drop=True)
        )
    else:
        paired_summary = paired.sort_values(["dataset", "baseline"]).reset_index(drop=True)
    paired_summary_path = cfg.results_dir / f"{cfg.output_prefix}_paired_deltas_summary.csv"
    paired_summary.to_csv(paired_summary_path, index=False)

    figure_path = cfg.figures_dir / f"{cfg.output_prefix}_wape.png"
    plot_neural_results(summary, figure_path)

    config_dict = asdict(cfg)
    config_dict["results_dir"] = str(config_dict["results_dir"])
    config_dict["figures_dir"] = str(config_dict["figures_dir"])
    metadata = {
        "config": config_dict,
        "device": str(device),
        "runtime_seconds": time.time() - started,
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "paired_deltas": str(paired_path),
            "paired_delta_summary": str(paired_summary_path),
            "figure": str(figure_path),
        },
        "folds": folds,
        "notes": [
            "All OD columns are included for every requested dataset.",
            "Use --blocked-folds for the three blocked chronological folds used in the strong-review Protocol D.",
            "PatchTST-Small uses patch tokenization plus a Transformer encoder, not a linear patch surrogate.",
            "N-BEATS-Small uses stacked backcast/forecast residual blocks.",
            "TimesNet-Small uses periodized 2D convolution blocks over fixed horizon-derived periods.",
            "N-HiTS-Small uses hierarchical pooling plus interpolated forecast heads.",
            "GraphAttention-Small uses matrix-level self-attention across all OD-flow nodes.",
            "All-flow WRTCN computes wavelet trend and residual channels causally inside each lookback window.",
            "SCALE-TM no-WRTCN ensemble and Uniform candidate ensemble are reported as ensemble ablations.",
            "Paired deltas bootstrap shared target end timestamps and report baseline minus SCALE-TM WAPE.",
            "Constant training flows are guarded by forcing normalized predictions to zero for every model.",
        ],
    }
    metadata_path = cfg.results_dir / f"{cfg.output_prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary[["dataset", "model", "WAPE", "WAPE_CI_low", "WAPE_CI_high"]].to_string(index=False))
    print(paired_summary.to_string(index=False))
    print(f"Saved {results_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {paired_path}")
    print(f"Saved {paired_summary_path}")
    print(f"Saved {figure_path}")
    print(f"Runtime seconds: {metadata['runtime_seconds']:.2f}")


if __name__ == "__main__":
    main()

"""Phase 0 EM surrogate models (latent MLP, pooled-mask MLP, optional mask CNN)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from nano_inv.latent import LATENT_DIM, LATENT_SHAPE, flatten_latent, pad_latent_to_standard
from nano_inv.manifest import filter_by_source
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Architecture = Literal["latent_mlp", "mask_mlp", "mask_cnn"]

MASK_SHAPE = (180, 180)
DEFAULT_MASK_POOL = 6


def normalize_mask_to_standard(mask: np.ndarray) -> np.ndarray:
    """Center-pad or crop binary masks to MASK_SHAPE (perturb 180×180, perlin 160×160)."""
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim > 2:
        m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {m.shape}")
    h, w = m.shape
    th, tw = MASK_SHAPE
    if (h, w) == (th, tw):
        return m
    if h > th or w > tw:
        ih = (h - th) // 2
        iw = (w - tw) // 2
        return m[ih : ih + th, iw : iw + tw].astype(np.float32)
    out = np.zeros(MASK_SHAPE, dtype=np.float32)
    dh = (th - h) // 2
    dw = (tw - w) // 2
    out[dh : dh + h, dw : dw + w] = m
    return out


@dataclass(frozen=True)
class SurrogateMetrics:
    n_ok_total: int
    n_train: int
    n_val: int
    target: str
    val_mae: float
    val_rmse: float
    val_r2: float
    holdout_fraction: float
    seed: int
    architecture: str = "latent_mlp"
    source_filter: str = "all"
    val_spearman_abs_err: float | None = None
    n_train_before_near_filter: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pool_mask(mask: np.ndarray, pool: int = DEFAULT_MASK_POOL) -> np.ndarray:
    m = normalize_mask_to_standard(mask)
    h, w = m.shape
    h2, w2 = h // pool * pool, w // pool * pool
    m = m[:h2, :w2]
    return m.reshape(h2 // pool, pool, w2 // pool, pool).max(axis=(1, 3))


def mask_to_features(mask: np.ndarray, pool: int = DEFAULT_MASK_POOL) -> np.ndarray:
    return pool_mask(mask, pool).ravel().astype(np.float32)


def load_latent_matrix(
    repo_root: Path,
    table: pd.DataFrame,
    *,
    latent_col: str = "latent_path",
) -> np.ndarray:
    rows = [flatten_latent(np.load(repo_root / p)) for p in table[latent_col]]
    return np.stack(rows, axis=0)


def load_mask_feature_matrix(
    repo_root: Path,
    table: pd.DataFrame,
    *,
    mask_col: str = "mask_path",
    pool: int = DEFAULT_MASK_POOL,
    manifold: Any | None = None,
    decode_from_latent: bool = False,
) -> np.ndarray:
    if decode_from_latent:
        if manifold is None:
            raise ValueError("decode_from_latent requires manifold")
        if "latent_path" not in table.columns:
            raise ValueError("decode_from_latent requires latent_path column")
        rows = [
            mask_to_features(manifold.decode_numpy(np.load(repo_root / p)), pool=pool)
            for p in table["latent_path"]
        ]
    else:
        rows = [mask_to_features(np.load(repo_root / p), pool=pool) for p in table[mask_col]]
    return np.stack(rows, axis=0)


def load_mask_tensor(
    repo_root: Path,
    table: pd.DataFrame,
    *,
    mask_col: str = "mask_path",
) -> np.ndarray:
    """(N, 1, H, W) float32 for CNN."""
    rows = []
    for p in table[mask_col]:
        m = normalize_mask_to_standard(np.load(repo_root / p))
        rows.append(m.reshape(1, *m.shape))
    return np.stack(rows, axis=0)


def _coalesce_merge_column(merged: pd.DataFrame, base: str) -> None:
    """Prefer corpus (_sim) over manifest (_man) when both exist after inner merge."""
    sim_col, man_col = f"{base}_sim", f"{base}_man"
    if sim_col not in merged.columns and man_col not in merged.columns:
        return
    sim_s = merged[sim_col] if sim_col in merged.columns else None
    man_s = merged[man_col] if man_col in merged.columns else None
    if sim_s is not None and man_s is not None:
        merged[base] = sim_s.combine_first(man_s)
    elif sim_s is not None:
        merged[base] = sim_s
    else:
        merged[base] = man_s


def build_labeled_table(
    repo_root: Path,
    manifest_path: Path,
    sim_results_path: Path,
    *,
    recipe_version: str | None = None,
    source_filter: str | list[str] | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    sims = pd.read_csv(sim_results_path)
    merged = manifest.merge(sims, on="sample_id", how="inner", suffixes=("_man", "_sim"))
    for col in ("mask_path", "latent_path", "source"):
        _coalesce_merge_column(merged, col)
    if "sigma_sim" in merged.columns:
        merged["sigma"] = merged["sigma_sim"].combine_first(
            merged.get("sigma_man", pd.Series(dtype=float))
        )
    elif "sigma_man" in merged.columns and "sigma" not in merged.columns:
        merged["sigma"] = merged["sigma_man"]
    ok = merged["status"] == "ok"
    labeled = merged.loc[ok].copy()
    if recipe_version is not None and "recipe_version" in labeled.columns:
        labeled = labeled.loc[labeled["recipe_version"] == recipe_version]
    labeled = filter_by_source(labeled, source_filter)
    return labeled.reset_index(drop=True)


def drop_invalid_targets(table: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in table.columns:
        return table
    valid = table[target].notna() & np.isfinite(table[target].to_numpy(dtype=float))
    return table.loc[valid].reset_index(drop=True)


def filter_near_target(
    table: pd.DataFrame,
    *,
    target_split_ratio: float = 0.5,
    max_abs_err: float | None = None,
) -> pd.DataFrame:
    """Keep rows with |split - target| <= max_abs_err (R²-focused training subset)."""
    if max_abs_err is None or "split_ratio_upper" not in table.columns:
        return table
    err = np.abs(table["split_ratio_upper"].astype(float) - target_split_ratio)
    return table.loc[err <= max_abs_err].reset_index(drop=True)


def append_sigma_feature(
    X: np.ndarray,
    table: pd.DataFrame,
    *,
    default_sigma: float = 0.02,
) -> np.ndarray:
    if "sigma" not in table.columns:
        return X
    sig = table["sigma"].astype(float).fillna(default_sigma).to_numpy().reshape(-1, 1)
    return np.hstack([X, sig.astype(np.float64)])


def val_ranking_spearman(
    y_val: np.ndarray,
    pred_val: np.ndarray,
    *,
    target_split_ratio: float = 0.5,
    target: str = "split_ratio_upper",
) -> float:
    from scipy.stats import spearmanr

    err_true = np.abs(y_val - target_split_ratio)
    err_pred = surrogate_ranking_scores(
        pred_val, target=target, target_split_ratio=target_split_ratio
    )
    if len(err_true) < 3:
        return float("nan")
    rho, _ = spearmanr(err_true, err_pred)
    return float(rho)


DERIVED_TARGETS = frozenset({"abs_split_error"})


def apply_training_target(
    table: pd.DataFrame,
    target: str,
    *,
    target_split_ratio: float = 0.5,
) -> tuple[pd.DataFrame, str]:
    """Return labeled table with target column ready for training."""
    if target == "split_ratio_upper":
        return table, target
    if target == "abs_split_error":
        if "split_ratio_upper" not in table.columns:
            raise ValueError("abs_split_error requires split_ratio_upper column")
        out = table.copy()
        out["abs_split_error"] = np.abs(out["split_ratio_upper"].astype(float) - target_split_ratio)
        return out, target
    raise ValueError(f"unsupported training target {target!r}")


def surrogate_ranking_scores(
    predictions: np.ndarray,
    *,
    target: str,
    target_split_ratio: float = 0.5,
) -> np.ndarray:
    """Lower score = better candidate for inverse design at target_split_ratio."""
    pred = np.asarray(predictions, dtype=float)
    if target in DERIVED_TARGETS:
        return pred
    return np.abs(pred - target_split_ratio)


def make_mlp_pipeline(
    *,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    max_iter: int = 500,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _metrics_from_predictions(
    y_val: np.ndarray,
    pred_val: np.ndarray,
    *,
    target: str,
    n_total: int,
    n_train: int,
    n_val: int,
    holdout_fraction: float,
    seed: int,
    architecture: str,
    source_filter: str,
    target_split_ratio: float = 0.5,
    n_train_before_near_filter: int | None = None,
) -> SurrogateMetrics:
    mae = float(mean_absolute_error(y_val, pred_val))
    rmse = float(np.sqrt(np.mean((y_val - pred_val) ** 2)))
    r2 = float(r2_score(y_val, pred_val)) if len(y_val) > 1 else float("nan")
    rho = val_ranking_spearman(
        y_val, pred_val, target_split_ratio=target_split_ratio, target=target
    )
    return SurrogateMetrics(
        n_ok_total=n_total,
        n_train=n_train,
        n_val=n_val,
        target=target,
        val_mae=mae,
        val_rmse=rmse,
        val_r2=r2,
        holdout_fraction=holdout_fraction,
        seed=seed,
        architecture=architecture,
        source_filter=source_filter,
        val_spearman_abs_err=rho,
        n_train_before_near_filter=n_train_before_near_filter,
    )


def compute_sample_weights(
    split_ratio_upper: np.ndarray,
    *,
    target_split_ratio: float = 0.5,
    mode: str | None = None,
    in_spec_tol: float = 0.05,
    latent_paths: np.ndarray | None = None,
    champion_latent_paths: list[str] | tuple[str, ...] | None = None,
    champion_weight: float = 2.0,
) -> np.ndarray | None:
    """
    Optional per-row weights for MLPRegressor (inverse-design emphasis).

    Modes: ``in_spec_boost`` (3× in-spec), ``soft_target`` (1/(0.08+|err|)),
    ``bimodal`` (2× if |err|<=0.08 else 1). Champion latents get ``champion_weight``
    when ``champion_latent_paths`` is set (multiplicative with mode weights).
    """
    if mode is None and not champion_latent_paths:
        return None
    split = np.asarray(split_ratio_upper, dtype=float)
    err = np.abs(split - target_split_ratio)
    w = np.ones(len(split), dtype=np.float64)
    if mode == "in_spec_boost":
        w[err <= in_spec_tol] = 3.0
    elif mode == "soft_target":
        w = 1.0 / (0.08 + err)
    elif mode == "bimodal":
        w[err <= 0.08] = 2.0
    elif mode == "rank_boost":
        from nano_inv.rank_loss import rank_weighted_regression_weights

        w = rank_weighted_regression_weights(
            split, target_split_ratio=target_split_ratio, power=1.0
        )
    elif mode is not None:
        raise ValueError(f"unknown sample_weight_mode {mode!r}")
    if champion_latent_paths and latent_paths is not None:
        champ_names = {Path(p).name for p in champion_latent_paths}
        for i, lp in enumerate(latent_paths):
            if Path(str(lp)).name in champ_names:
                w[i] *= champion_weight
    return w


def train_sklearn_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    target: str = "split_ratio_upper",
    holdout_fraction: float = 0.2,
    seed: int = 42,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    max_iter: int = 500,
    architecture: str = "latent_mlp",
    source_filter: str = "all",
    target_split_ratio: float = 0.5,
    n_train_before_near_filter: int | None = None,
    sample_weight: np.ndarray | None = None,
    split_for_weights: np.ndarray | None = None,
) -> tuple[Pipeline, SurrogateMetrics]:
    if len(X) < 10:
        raise ValueError(f"need at least 10 ok samples to train, got {len(X)}")
    if not (0.0 < holdout_fraction < 0.5):
        raise ValueError("holdout_fraction should be in (0, 0.5)")

    if sample_weight is not None and len(sample_weight) != len(X):
        raise ValueError("sample_weight length must match X")

    if sample_weight is not None:
        X_train, X_val, y_train, y_val, w_train, _w_val = train_test_split(
            X,
            y,
            sample_weight,
            test_size=holdout_fraction,
            random_state=seed,
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=holdout_fraction, random_state=seed
        )
        w_train = None

    pipe = make_mlp_pipeline(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=seed,
    )
    if w_train is not None:
        pipe.fit(X_train, y_train, mlp__sample_weight=w_train)
    else:
        pipe.fit(X_train, y_train)
    pred_val = pipe.predict(X_val)
    metrics = _metrics_from_predictions(
        y_val,
        pred_val,
        target=target,
        n_total=len(X),
        n_train=len(X_train),
        n_val=len(X_val),
        holdout_fraction=holdout_fraction,
        seed=seed,
        architecture=architecture,
        source_filter=source_filter,
        target_split_ratio=target_split_ratio,
        n_train_before_near_filter=n_train_before_near_filter,
    )
    return pipe, metrics


def train_rank_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    target: str = "split_ratio_upper",
    holdout_fraction: float = 0.2,
    seed: int = 42,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    max_iter: int = 500,
    architecture: str = "mask_mlp",
    source_filter: str = "all",
    target_split_ratio: float = 0.5,
    n_train_before_near_filter: int | None = None,
    sample_weight: np.ndarray | None = None,
    mse_weight: float = 0.15,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> tuple[Any, SurrogateMetrics]:
    """
    MLP with pairwise rank loss (RankNet + LambdaRank-lite pair weights).

    Requires PyTorch: ``uv pip install torch``.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise ImportError(
            "pairwise_rank loss requires PyTorch: uv pip install torch --python .venv/bin/python"
        ) from e

    from nano_inv.rank_loss import pairwise_rank_loss_torch

    if len(X) < 10:
        raise ValueError(f"need at least 10 ok samples to train, got {len(X)}")
    if not (0.0 < holdout_fraction < 0.5):
        raise ValueError("holdout_fraction should be in (0, 0.5)")

    if sample_weight is not None and len(sample_weight) != len(X):
        raise ValueError("sample_weight length must match X")

    if sample_weight is not None:
        X_train, X_val, y_train, y_val, w_train, _w_val = train_test_split(
            X,
            y,
            sample_weight,
            test_size=holdout_fraction,
            random_state=seed,
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=holdout_fraction, random_state=seed
        )
        w_train = None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    device = torch.device("cpu")
    in_dim = X_train_s.shape[1]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_layer_sizes:
        layers.extend([nn.Linear(prev, h), nn.ReLU()])
        prev = h
    layers.append(nn.Linear(prev, 1))
    model = nn.Sequential(*layers).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X_t = torch.from_numpy(X_train_s.astype(np.float32))
    y_t = torch.from_numpy(y_train.astype(np.float32))
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    best_rho = -2.0
    best_state: dict[str, Any] | None = None
    patience = 25
    stale = 0
    epochs = max(max_iter, 100)

    for _epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = pairwise_rank_loss_torch(
                pred,
                yb,
                target_split_ratio=target_split_ratio,
                mse_weight=mse_weight,
            )
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = (
                model(torch.from_numpy(X_val_s.astype(np.float32)).to(device))
                .cpu()
                .numpy()
                .ravel()
            )
        rho = val_ranking_spearman(
            y_val, pred_val, target_split_ratio=target_split_ratio, target=target
        )
        if rho > best_rho + 1e-5:
            best_rho = rho
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    wrapper = TorchRankMLP(
        model=model,
        scaler_mean_=scaler.mean_.astype(np.float64),
        scaler_scale_=scaler.scale_.astype(np.float64),
        device=str(device),
        target_split_ratio=target_split_ratio,
        target=target,
        loss_mode="pairwise_rank",
    )
    pred_val = wrapper.predict(X_val)
    metrics = _metrics_from_predictions(
        y_val,
        pred_val,
        target=target,
        n_total=len(X),
        n_train=len(X_train),
        n_val=len(X_val),
        holdout_fraction=holdout_fraction,
        seed=seed,
        architecture=architecture,
        source_filter=source_filter,
        target_split_ratio=target_split_ratio,
        n_train_before_near_filter=n_train_before_near_filter,
    )
    return wrapper, metrics


@dataclass
class TorchRankMLP:
    """Sklearn-compatible predictor: StandardScaler + torch MLP."""

    model: Any
    scaler_mean_: np.ndarray
    scaler_scale_: np.ndarray
    device: str = "cpu"
    target_split_ratio: float = 0.5
    target: str = "split_ratio_upper"
    loss_mode: str = "pairwise_rank"

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X, dtype=np.float64)
        scale = np.where(self.scaler_scale_ != 0, self.scaler_scale_, 1.0)
        Xs = (X - self.scaler_mean_) / scale
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
            return self.model(t).cpu().numpy().ravel()


def train_mask_cnn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    target: str = "split_ratio_upper",
    holdout_fraction: float = 0.2,
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 32,
    source_filter: str = "all",
) -> tuple[Any, SurrogateMetrics]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise ImportError(
            "mask_cnn requires PyTorch: uv pip install torch --python .venv/bin/python"
        ) from e

    if len(X) < 10:
        raise ValueError(f"need at least 10 ok samples to train, got {len(X)}")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = max(1, int(len(X) * holdout_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    device = torch.device("cpu")
    X_t = torch.from_numpy(X[train_idx])
    y_t = torch.from_numpy(y[train_idx].astype(np.float32).reshape(-1, 1))
    X_v = torch.from_numpy(X[val_idx])
    y_v = y[val_idx].astype(np.float32)

    class SmallCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(8),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_val = model(torch.from_numpy(X[val_idx]).to(device)).cpu().numpy().ravel()

    wrapper = TorchMaskCNN(model=model, device=str(device))
    metrics = _metrics_from_predictions(
        y_v,
        pred_val,
        target=target,
        n_total=len(X),
        n_train=len(train_idx),
        n_val=len(val_idx),
        holdout_fraction=holdout_fraction,
        seed=seed,
        architecture="mask_cnn",
        source_filter=source_filter,
    )
    return wrapper, metrics


@dataclass
class TorchMaskCNN:
    model: Any
    device: str = "cpu"

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(X.astype(np.float32)).to(self.device)
            out = self.model(t).cpu().numpy().ravel()
        return out


@dataclass
class SurrogateArtifact:
    pipeline: Any
    target: str
    recipe_version: str | None
    metrics: SurrogateMetrics
    architecture: str = "latent_mlp"
    input_dim: int = LATENT_DIM
    mask_pool: int = DEFAULT_MASK_POOL
    latent_dim: int | None = None  # backward compat
    target_split_ratio: float = 0.5
    sigma_feature: bool = False
    loss_mode: str = "regression"

    def __post_init__(self) -> None:
        if self.latent_dim is not None and self.architecture == "latent_mlp":
            self.input_dim = self.latent_dim

    def _architecture(self) -> str:
        return getattr(self, "architecture", "latent_mlp")

    def predict_latent(self, latent: np.ndarray) -> float:
        if self._architecture().startswith("mask"):
            raise ValueError(
                f"{self._architecture()} requires decode; use predict_from_latent(manifold=...)"
            )
        x = flatten_latent(latent).reshape(1, -1)
        return float(self.pipeline.predict(x)[0])

    def predict_mask(self, mask: np.ndarray, *, sigma: float | None = None) -> float:
        arch = self._architecture()
        if arch == "mask_cnn":
            m = np.asarray(mask, dtype=np.float32)
            if m.ndim == 2:
                m = m.reshape(1, 1, *m.shape)
            elif m.ndim == 3:
                m = m.reshape(1, *m.shape)
            return float(self.pipeline.predict(m)[0])
        x = mask_to_features(mask, pool=self.mask_pool).reshape(1, -1)
        if self.sigma_feature:
            s = 0.02 if sigma is None else float(sigma)
            x = np.hstack([x, np.array([[s]], dtype=np.float64)])
        return float(self.pipeline.predict(x)[0])

    def predict_from_latent(
        self,
        latent: np.ndarray,
        *,
        manifold: Any | None = None,
        sigma: float | None = None,
    ) -> float:
        if self._architecture().startswith("mask"):
            if manifold is None:
                raise ValueError("mask surrogate needs manifold to decode latent")
            mask = manifold.decode_numpy(latent)
            return self.predict_mask(mask, sigma=sigma)
        return self.predict_latent(latent)

    def predict_batch(self, latents: np.ndarray) -> np.ndarray:
        if self._architecture().startswith("mask"):
            raise ValueError("use predict_from_latent per sample for mask models")
        if latents.ndim > 2:
            latents = np.stack([flatten_latent(z) for z in latents], axis=0)
        return self.pipeline.predict(latents)


def train_surrogate_bundle(
    repo_root: Path,
    labeled: pd.DataFrame,
    *,
    architecture: Architecture,
    target: str,
    holdout_fraction: float,
    seed: int,
    hidden_layer_sizes: tuple[int, ...],
    max_iter: int,
    source_filter: str,
    mask_pool: int = DEFAULT_MASK_POOL,
    cnn_epochs: int = 40,
    target_split_ratio: float = 0.5,
    near_target_max_abs_err: float | None = None,
    sigma_feature: bool = False,
    sample_weight_mode: str | None = None,
    sample_weight_in_spec_tol: float = 0.05,
    champion_latent_paths: list[str] | tuple[str, ...] | None = None,
    champion_weight: float = 2.0,
    decode_masks_from_latent: bool = False,
    manifold: Any | None = None,
    loss_mode: str = "regression",
    rank_mse_weight: float = 0.15,
) -> SurrogateArtifact:
    n_before = len(labeled)
    labeled = filter_near_target(
        labeled, target_split_ratio=target_split_ratio, max_abs_err=near_target_max_abs_err
    )
    n_filtered = len(labeled)
    labeled, train_target = apply_training_target(
        labeled, target, target_split_ratio=target_split_ratio
    )
    y = labeled[train_target].astype(np.float64).to_numpy()
    near_note = n_before if near_target_max_abs_err is not None else None
    split_col = labeled["split_ratio_upper"].astype(np.float64).to_numpy()
    latent_col = (
        labeled["latent_path"].astype(str).to_numpy()
        if "latent_path" in labeled.columns
        else None
    )
    sw_mode = sample_weight_mode
    if loss_mode == "rank_weighted" and sw_mode is None:
        sw_mode = "rank_boost"
    weights = compute_sample_weights(
        split_col,
        target_split_ratio=target_split_ratio,
        mode=sw_mode,
        in_spec_tol=sample_weight_in_spec_tol,
        latent_paths=latent_col,
        champion_latent_paths=champion_latent_paths,
        champion_weight=champion_weight,
    )

    def _train_mlp(
        X: np.ndarray, arch: str, *, sigma: bool = False
    ) -> tuple[Any, SurrogateMetrics]:
        if loss_mode == "pairwise_rank":
            return train_rank_mlp(
                X,
                y,
                target=train_target,
                holdout_fraction=holdout_fraction,
                seed=seed,
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                architecture=arch,
                source_filter=source_filter,
                target_split_ratio=target_split_ratio,
                n_train_before_near_filter=near_note,
                mse_weight=rank_mse_weight,
            )
        return train_sklearn_surrogate(
            X,
            y,
            target=train_target,
            holdout_fraction=holdout_fraction,
            seed=seed,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            architecture=arch,
            source_filter=source_filter,
            target_split_ratio=target_split_ratio,
            n_train_before_near_filter=near_note,
            sample_weight=weights,
        )

    if architecture == "latent_mlp":
        X = load_latent_matrix(repo_root, labeled)
        pipe, metrics = _train_mlp(X, architecture)
        return SurrogateArtifact(
            pipeline=pipe,
            target=train_target,
            recipe_version=None,
            metrics=metrics,
            architecture=architecture,
            input_dim=X.shape[1],
            mask_pool=mask_pool,
            target_split_ratio=target_split_ratio,
            loss_mode=loss_mode,
        )

    if architecture == "mask_mlp":
        X = load_mask_feature_matrix(
            repo_root,
            labeled,
            pool=mask_pool,
            manifold=manifold,
            decode_from_latent=decode_masks_from_latent,
        )
        if sigma_feature:
            X = append_sigma_feature(X, labeled)
        pipe, metrics = _train_mlp(X, architecture, sigma=sigma_feature)
        return SurrogateArtifact(
            pipeline=pipe,
            target=train_target,
            recipe_version=None,
            metrics=metrics,
            architecture=architecture,
            input_dim=X.shape[1],
            mask_pool=mask_pool,
            target_split_ratio=target_split_ratio,
            sigma_feature=sigma_feature,
            loss_mode=loss_mode,
        )

    if architecture == "mask_cnn":
        X = load_mask_tensor(repo_root, labeled)
        pipe, metrics = train_mask_cnn(
            X,
            y,
            target=train_target,
            holdout_fraction=holdout_fraction,
            seed=seed,
            epochs=cnn_epochs,
            source_filter=source_filter,
        )
        return SurrogateArtifact(
            pipeline=pipe,
            target=train_target,
            recipe_version=None,
            metrics=metrics,
            architecture=architecture,
            input_dim=int(np.prod(MASK_SHAPE)),
            mask_pool=mask_pool,
            target_split_ratio=target_split_ratio,
        )

    raise ValueError(f"unknown architecture {architecture!r}")


# backward-compatible alias
train_surrogate = train_sklearn_surrogate


def save_artifact(artifact: SurrogateArtifact, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_dir / "surrogate.joblib")
    meta = artifact.metrics.to_dict()
    meta["architecture"] = artifact.architecture
    meta["input_dim"] = artifact.input_dim
    meta["mask_pool"] = artifact.mask_pool
    meta["target_split_ratio"] = getattr(artifact, "target_split_ratio", 0.5)
    meta["sigma_feature"] = getattr(artifact, "sigma_feature", False)
    meta["loss_mode"] = getattr(artifact, "loss_mode", "regression")
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2))


def load_artifact(path: Path) -> SurrogateArtifact:
    art: SurrogateArtifact = joblib.load(path)
    if not hasattr(art, "architecture"):
        art.architecture = "latent_mlp"  # type: ignore[attr-defined]
    if not hasattr(art, "input_dim"):
        art.input_dim = getattr(art, "latent_dim", LATENT_DIM)  # type: ignore[attr-defined]
    if not hasattr(art, "target_split_ratio"):
        art.target_split_ratio = 0.5  # type: ignore[attr-defined]
    return art

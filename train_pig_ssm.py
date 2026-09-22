"""Canonical PiG-SSM training script.

This script is the cleaned training version of
``cross-materials/cross_materials_embedding2.py``. It keeps the final
manuscript model architecture in ``pig_ssm_model.py`` and preserves the
physics-informed gate regularization used in the original training code.

Revision-v2 (controlled protocol; the only mode used by paper commands):
    python train_pig_ssm.py \
        --split_manifest revision_v2/manifests/alloy-1_seed42.json \
        --seed 42 --output_root revision_v2/runs/<run> --evaluate_test_once

The revision-v2 path loads exactly the sequences named in the manifest,
fits scalers on the train split only, and evaluates the test split exactly
once after loading the best checkpoint.

Legacy (historical random 80/20 split; never used by paper commands):
    python train_pig_ssm.py --data_dirs /path/to/data --legacy_random_split

Distributed (legacy):
    torchrun --nproc_per_node=4 train_pig_ssm.py --data_dirs /path/to/data --legacy_random_split
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pig_ssm_model import StressPredictorMamba, count_parameters

try:
    from revision_pipeline.paths import resolve_project_root
    from revision_pipeline.json_utils import dump_json
    from revision_pipeline.provenance import runtime_provenance, utc_now_iso
except ImportError:  # imported as organized_pig_ssm.train_pig_ssm from the project root
    from organized_pig_ssm.revision_pipeline.paths import resolve_project_root
    from organized_pig_ssm.revision_pipeline.json_utils import dump_json
    from organized_pig_ssm.revision_pipeline.provenance import runtime_provenance, utc_now_iso


class StrainStressDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, material_ids: np.ndarray):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)
        self.material_ids = torch.LongTensor(material_ids)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.material_ids[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return distributed, rank, local_rank, world_size, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def natural_key(path: str | os.PathLike) -> list[int]:
    numbers = re.findall(r"\d+", os.path.basename(str(path)))
    return [int(token) for token in numbers] if numbers else [0]


def material_key_from_filename(path: str | os.PathLike) -> str | None:
    parts = os.path.basename(str(path)).split("-")
    if len(parts) < 2:
        return None
    return f"{parts[0]}-{parts[1]}"


def list_csv_files(data_dirs: list[str | os.PathLike]) -> list[str]:
    files: list[str] = []
    for data_dir in data_dirs:
        files.extend(glob.glob(os.path.join(str(data_dir), "*.csv")))
    return sorted(files, key=natural_key)


def load_json_mapping(mapping_path: str | os.PathLike) -> dict[str, int]:
    with open(mapping_path, "r") as f:
        return json.load(f)


def save_json_mapping(mapping: dict[str, int], mapping_path: str | os.PathLike) -> None:
    mapping_path = Path(mapping_path)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(mapping_path, mapping)


def create_or_update_material_mapping(
    data_dirs: list[str | os.PathLike],
    mapping_path: str | os.PathLike,
    reset_mapping: bool = False,
) -> dict[str, int]:
    """Create or extend a material mapping from CSV file names.

    Material ids follow the original code convention: the first two filename
    fields separated by '-' form the material key, e.g. ``DP980-001-...``.
    """

    mapping_path = Path(mapping_path)
    if mapping_path.exists() and not reset_mapping:
        mapping = load_json_mapping(mapping_path)
    else:
        mapping = {}

    next_idx = len(mapping)
    for file_path in list_csv_files(data_dirs):
        material_key = material_key_from_filename(file_path)
        if material_key is None:
            continue
        if material_key not in mapping:
            mapping[material_key] = next_idx
            next_idx += 1

    save_json_mapping(mapping, mapping_path)
    return mapping


def load_csv_array(file_path: str | os.PathLike, skip_header: bool | None = None):
    try:
        return np.loadtxt(file_path, delimiter=",", skiprows=1 if skip_header else 0)
    except ValueError:
        if skip_header is False:
            raise
        return np.loadtxt(file_path, delimiter=",", skiprows=1)


def load_raw_sequences(
    data_dirs: list[str | os.PathLike],
    mapping_path: str | os.PathLike,
    seq_len: int = 101,
    strain_zero_threshold: float = 0.01,
    stress_zero_threshold: float = 50.0,
    skip_header: bool | None = None,
    max_sequences: int | None = None,
):
    mapping = load_json_mapping(mapping_path)
    x_list, y_list, material_ids = [], [], []

    for file_path in list_csv_files(data_dirs):
        material_key = material_key_from_filename(file_path)
        if material_key not in mapping:
            continue

        try:
            data = load_csv_array(file_path, skip_header=skip_header)
        except Exception:
            continue

        if data.ndim != 2 or data.shape[0] < seq_len or data.shape[1] < 12:
            continue

        x_data = data[:seq_len, :6].copy()
        y_data = data[:seq_len, 6:12].copy()

        for channel in range(6):
            if np.ptp(x_data[:, channel]) < strain_zero_threshold:
                x_data[:, channel] = 0.0
            if np.ptp(y_data[:, channel]) < stress_zero_threshold:
                y_data[:, channel] = 0.0

        x_list.append(x_data)
        y_list.append(y_data)
        material_ids.append(mapping[material_key])
        if max_sequences is not None and len(x_list) >= max_sequences:
            break

    if not x_list:
        raise ValueError(f"No usable CSV sequences found in {data_dirs}")

    return np.array(x_list), np.array(y_list), np.array(material_ids)


def build_criterion(loss_name: str):
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    if loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def gate_regularization_loss(
    all_z: list[torch.Tensor],
    batch_x: torch.Tensor,
    gate_threshold: float = 1e-4,
) -> torch.Tensor:
    eps_prev = torch.cat([batch_x[:, :1], batch_x[:, :-1]], dim=1)
    delta_eps = batch_x - eps_prev
    delta_norm = torch.norm(delta_eps, dim=-1, keepdim=True)
    load_mask = (delta_norm > gate_threshold).float()

    gate_loss = batch_x.new_tensor(0.0)
    for z_gate in all_z:
        z_mean = z_gate.mean(dim=-1, keepdim=True)
        gate_loss = gate_loss + F.mse_loss(z_mean, load_mask)
    return gate_loss / max(len(all_z), 1)


def train_one_epoch(
    model: nn.Module,
    criterion,
    optimizer,
    dataloader: DataLoader,
    device: torch.device,
    lambda_gate: float,
    gate_threshold: float,
    grad_clip: float,
):
    model.train()
    total_task_loss = 0.0
    total_gate_loss = 0.0
    total_combined_loss = 0.0

    for batch_x, batch_y, batch_mat in dataloader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_mat = batch_mat.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred, all_z, _ = model(batch_x, material_id=batch_mat, return_gate=True)
        task_loss = criterion(pred, batch_y)
        gate_loss = gate_regularization_loss(all_z, batch_x, gate_threshold=gate_threshold)
        combined_loss = task_loss + lambda_gate * gate_loss
        combined_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = batch_x.size(0)
        total_task_loss += task_loss.item() * batch_size
        total_gate_loss += gate_loss.item() * batch_size
        total_combined_loss += combined_loss.item() * batch_size

    dataset_size = max(len(dataloader.dataset), 1)
    return {
        "train_loss": total_task_loss / dataset_size,
        "gate_loss": total_gate_loss / dataset_size,
        "combined_loss": total_combined_loss / dataset_size,
    }


def validate(model: nn.Module, criterion, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y, batch_mat in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_mat = batch_mat.to(device, non_blocking=True)
            pred = model(batch_x, material_id=batch_mat, return_gate=False)
            loss = criterion(pred, batch_y)
            total_loss += loss.item() * batch_x.size(0)
    return total_loss / max(len(dataloader.dataset), 1)


def make_split_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    mat_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    mat_val: np.ndarray,
    args,
    distributed: bool,
    world_size: int,
    rank: int,
):
    """Build train/val DataLoaders from already-split, already-normalized arrays.

    This helper performs no splitting and no scaler fitting: callers are
    responsible for where the data came from (legacy random split or a
    revision-v2 manifest).
    """
    train_dataset = StrainStressDataset(x_train, y_train, mat_train)
    val_dataset = StrainStressDataset(x_val, y_val, mat_val)

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_sampler


def make_data_loaders(
    x: np.ndarray,
    y: np.ndarray,
    material_ids: np.ndarray,
    args,
    distributed: bool,
    world_size: int,
    rank: int,
):
    """Legacy random 80/20 split + train-fit scalers.

    Retained solely for historical runs behind ``--legacy_random_split``;
    revision-v2 runs must use :func:`run_manifest_core` instead.
    """
    x_train, x_val, y_train, y_val, mat_train, mat_val = train_test_split(
        x,
        y,
        material_ids,
        test_size=args.val_split,
        random_state=args.seed,
    )

    scaler_x, scaler_y = fit_scalers_from_train_only(x_train, y_train)

    x_train = scaler_x.transform(x_train.reshape(-1, args.input_dim)).reshape(x_train.shape)
    x_val = scaler_x.transform(x_val.reshape(-1, args.input_dim)).reshape(x_val.shape)
    y_train = scaler_y.transform(y_train.reshape(-1, args.output_dim)).reshape(y_train.shape)
    y_val = scaler_y.transform(y_val.reshape(-1, args.output_dim)).reshape(y_val.shape)

    train_loader, val_loader, train_sampler = make_split_loaders(
        x_train,
        y_train,
        mat_train,
        x_val,
        y_val,
        mat_val,
        args,
        distributed,
        world_size,
        rank,
    )
    return train_loader, val_loader, train_sampler, scaler_x, scaler_y


# ---------------------------------------------------------------------------
# revision-v2 manifest-driven training (plan Task 4)
# ---------------------------------------------------------------------------


def fit_scalers_from_train_only(train_x: np.ndarray, train_y: np.ndarray):
    """Fit StandardScalers using the train split only.

    The signature intentionally has no validation/test arguments: the
    revision-v2 protocol forbids val/test data from influencing scaler
    statistics (mean/std).
    """
    scaler_x = StandardScaler().fit(train_x.reshape(-1, train_x.shape[-1]))
    scaler_y = StandardScaler().fit(train_y.reshape(-1, train_y.shape[-1]))
    return scaler_x, scaler_y


def load_manifest(manifest_path: str | os.PathLike) -> dict:
    return json.loads(Path(manifest_path).read_text())


def manifest_sha256(manifest_path: str | os.PathLike) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(manifest_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_mapping_from_manifest(project_root, manifest: dict, split: str = "train") -> dict[str, int]:
    """Deterministic material mapping from the TRAIN split file names.

    Keys are sorted lexicographically so index assignment is reproducible and
    identical across seeds (every seed's train split contains every material).
    Only the train split is consulted, per the protocol: validation/test must
    not influence the material mapping.
    """
    keys = set()
    for entry in manifest[split]:
        key = material_key_from_filename(entry["path"])
        if key is not None:
            keys.add(key)
    if not keys:
        raise ValueError(f"no material keys found in manifest split {split!r}")
    return {key: idx for idx, key in enumerate(sorted(keys))}


def load_manifest_split(
    project_root,
    manifest: dict,
    mapping: dict[str, int],
    split: str,
    seq_len: int = 101,
    strain_zero_threshold: float = 0.01,
    stress_zero_threshold: float = 50.0,
):
    """Load exactly the sequences listed in one manifest split.

    Strict by design: a missing file or a material key absent from the
    train-derived mapping raises instead of being skipped silently (the
    leakage audit is supposed to have caught both before training starts).
    """
    x_list, y_list, material_ids = [], [], []
    for entry in manifest[split]:
        rel_path = entry["path"]
        path = project_root / rel_path
        material_key = material_key_from_filename(path)
        if material_key not in mapping:
            raise ValueError(
                f"material key {material_key!r} ({split} file {rel_path!r}) is absent from "
                "the train-derived mapping; refusing to continue (protocol violation)"
            )
        data = load_csv_array(path, skip_header=None)
        if data.ndim != 2 or data.shape[0] < seq_len or data.shape[1] < 12:
            raise ValueError(
                f"manifest file {rel_path!r} has shape {data.shape}; "
                f"expected at least ({seq_len}, 12)"
            )
        x_data = data[:seq_len, :6].copy()
        y_data = data[:seq_len, 6:12].copy()
        for channel in range(6):
            if np.ptp(x_data[:, channel]) < strain_zero_threshold:
                x_data[:, channel] = 0.0
            if np.ptp(y_data[:, channel]) < stress_zero_threshold:
                y_data[:, channel] = 0.0
        x_list.append(x_data)
        y_list.append(y_data)
        material_ids.append(mapping[material_key])
    if not x_list:
        raise ValueError(f"manifest split {split!r} is empty")
    return np.array(x_list), np.array(y_list), np.array(material_ids)


class StrainStressEvalDataset(Dataset):
    """One-shot evaluation batches: normalized inputs/targets plus raw physical arrays."""

    def __init__(self, x_norm: np.ndarray, y_norm: np.ndarray, x_raw: np.ndarray, y_raw: np.ndarray, material_ids: np.ndarray):
        self.x_norm = torch.FloatTensor(x_norm)
        self.y_norm = torch.FloatTensor(y_norm)
        self.x_raw = torch.FloatTensor(x_raw)
        self.y_raw = torch.FloatTensor(y_raw)
        self.material_ids = torch.LongTensor(material_ids)

    def __len__(self) -> int:
        return len(self.x_norm)

    def __getitem__(self, idx: int):
        return (
            self.x_norm[idx],
            self.y_norm[idx],
            self.x_raw[idx],
            self.y_raw[idx],
            self.material_ids[idx],
        )


def active_component_mask(y_true_raw: np.ndarray, active_threshold: float = 1e-3) -> np.ndarray:
    """Per-sample, per-component activity: raw stress ptp above threshold.

    Same definition as the historical E1 evaluation (active_threshold=1e-3 MPa).
    """
    return np.ptp(y_true_raw, axis=1) > active_threshold


def compute_physical_metrics(
    pred_norm: np.ndarray,
    y_raw: np.ndarray,
    scaler_y,
    output_dim: int,
    active_threshold: float = 1e-3,
) -> dict:
    """Physical-unit (MPa) metrics after inverse-transforming with scaler_y.

    Includes ``per_sequence_active_mae_mpa`` (aligned to input sequence order)
    so that the Task 5 bootstrap CI can be computed without re-running the
    checkpoint.
    """
    pred_raw = scaler_y.inverse_transform(pred_norm.reshape(-1, output_dim)).reshape(pred_norm.shape)
    abs_error = np.abs(pred_raw - y_raw)

    active_mask = active_component_mask(y_raw, active_threshold)  # (n, 6)
    # Expand to full (n, T, 6): every timestep of an active component is an
    # error slot — the historical E1 evaluator's definition
    # (abs_error[sample, :, active_comps]).
    slot_mask = np.broadcast_to(active_mask[:, None, :], abs_error.shape)
    if slot_mask.any():
        active_err = abs_error[slot_mask]
        active_true = y_raw[slot_mask]
        active_pred = pred_raw[slot_mask]
        active_mae = float(np.mean(active_err))
        active_rmse = float(np.sqrt(np.mean((active_pred - active_true) ** 2)))
        active_max_error = float(np.max(active_err))
        denom = np.maximum(np.abs(active_true), 1e-8)
        active_mape = float(np.mean(np.abs((active_true - active_pred) / denom)) * 100.0)
    else:
        active_mae = active_rmse = active_max_error = active_mape = float("nan")

    slot_counts = slot_mask.sum(axis=(1, 2))
    per_seq = np.where(
        slot_counts > 0,
        (abs_error * slot_mask).sum(axis=(1, 2)) / np.maximum(slot_counts, 1),
        np.nan,
    )

    per_component = {}
    for comp in range(output_dim):
        comp_err = abs_error[:, :, comp].reshape(-1)
        comp_diff = (pred_raw[:, :, comp] - y_raw[:, :, comp]).reshape(-1)
        per_component[f"mae_s{comp}_mpa"] = float(np.mean(comp_err))
        per_component[f"rmse_s{comp}_mpa"] = float(np.sqrt(np.mean(comp_diff ** 2)))

    return {
        "active_component_mae_mpa": active_mae,
        "active_component_rmse_mpa": active_rmse,
        "max_error_active_mpa": active_max_error,
        "mape_active_percent": active_mape,
        "all_component_mae_mpa": float(np.mean(abs_error)),
        "all_component_rmse_mpa": float(np.sqrt(np.mean((pred_raw - y_raw) ** 2))),
        "max_error_all_mpa": float(np.max(abs_error)),
        "num_eval_sequences": int(y_raw.shape[0]),
        "num_active_component_slots": int(active_mask.sum()),
        "active_threshold_mpa": float(active_threshold),
        "per_component": per_component,
        "per_sequence_active_mae_mpa": [float(v) for v in per_seq],
    }


def gate_auroc(z_gate: np.ndarray, x_raw: np.ndarray, active_delta_threshold: float = 1e-6) -> float:
    """AUROC of the last-layer gate as a classifier of deforming timesteps.

    Identical definition to the archived E3 metric
    ``gate_auc_active_delta_norm`` (E3/code/evaluate_e3_physical_consistency.py,
    ``compute_gate_latent_metrics``): for every (sample, timestep) slot the
    score is the channel mean of the gate, and the label is 1 when the
    2-norm of the strain increment delta_eps[t] = eps[t] - eps[t-1]
    (delta_eps[0] = 0) exceeds ``active_delta_threshold``. Measures whether
    the gate opens when the material is actually deforming. The gate has
    ``d_model`` channels (not one per stress component), so the score must be
    reduced over channels; the archived E3 comparison uses the last layer, so
    callers must pass the last layer's gate. Returns nan for a single-class
    or constant-score set (e.g. the ``no_gate`` variant, z identically 1; the
    ``static_gate`` variant, z constant over (sample, timestep)).
    """
    from sklearn.metrics import roc_auc_score

    scores = z_gate.mean(axis=2).reshape(-1)  # (n, T) -> flat
    delta_eps = np.empty_like(x_raw)
    delta_eps[:, 0] = 0.0
    delta_eps[:, 1:] = x_raw[:, 1:] - x_raw[:, :-1]
    delta_norm = np.linalg.norm(delta_eps, axis=-1)
    y_true = (delta_norm > active_delta_threshold).astype(float).reshape(-1)
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        return float("nan")
    if not np.any(scores != scores[0]):
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def cyclic_energy_relative_error(pred_raw: np.ndarray, y_raw: np.ndarray, x_raw: np.ndarray, min_abs_energy: float = 1e-6) -> float:
    """Mean |E_pred - E_true| / |E_true| over sequences with nonzero true energy.

    E = sum_t sum_c sigma[t, c] * delta_eps[t, c], with delta_eps[0] = 0 and
    delta_eps[t] = eps[t] - eps[t-1] (the same increment convention as the
    gate regularization). A dissipation-style energy proxy for path-dependent
    response, computed on the zero-threshold-processed strain/stress arrays.
    Sequences with |E_true| < min_abs_energy are excluded.
    """
    delta_eps = np.empty_like(x_raw)
    delta_eps[:, 0] = 0.0
    delta_eps[:, 1:] = x_raw[:, 1:] - x_raw[:, :-1]
    e_true = (y_raw * delta_eps).sum(axis=(1, 2))
    e_pred = (pred_raw * delta_eps).sum(axis=(1, 2))
    valid = np.abs(e_true) >= min_abs_energy
    if not valid.any():
        return float("nan")
    rel = np.abs(e_pred[valid] - e_true[valid]) / np.abs(e_true[valid])
    return float(np.mean(rel))


def evaluate_split_once(
    model: nn.Module,
    dataloader: DataLoader,
    scaler_y,
    device: torch.device,
    output_dim: int,
    active_threshold: float = 1e-3,
    collect_gates: bool = True,
) -> dict:
    """Evaluate a model over a dataloader exactly once and return metrics.

    The dataloader must yield (x_norm, y_norm, x_raw, y_raw, material_ids)
    (:class:`StrainStressEvalDataset`). This is the only permitted
    evaluation entry point for revision-v2 splits: it runs once after the
    best checkpoint is loaded, never inside the training loop.
    """
    model.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    smooth_l1 = nn.SmoothL1Loss(reduction="sum")

    total_mse = 0.0
    total_smooth_l1 = 0.0
    total_elements = 0
    pred_norm_batches, y_raw_batches, x_raw_batches = [], [], []
    z_last_batches = []

    with torch.no_grad():
        for batch_x, batch_y_norm, batch_x_raw, batch_y_raw, batch_mat in dataloader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y_norm = batch_y_norm.to(device, non_blocking=True)
            batch_x_raw = batch_x_raw.to(device, non_blocking=True)
            batch_y_raw = batch_y_raw.to(device, non_blocking=True)
            batch_mat = batch_mat.to(device, non_blocking=True)

            if collect_gates:
                pred, all_z, _ = model(batch_x, material_id=batch_mat, return_gate=True)
                z_last_batches.append(all_z[-1].detach().cpu().numpy())
            else:
                pred = model(batch_x, material_id=batch_mat, return_gate=False)

            total_mse += mse_loss(pred, batch_y_norm).item()
            total_smooth_l1 += smooth_l1(pred, batch_y_norm).item()
            total_elements += batch_y_norm.numel()

            pred_norm_batches.append(pred.detach().cpu().numpy())
            y_raw_batches.append(batch_y_raw.cpu().numpy())
            x_raw_batches.append(batch_x_raw.cpu().numpy())

    pred_norm = np.concatenate(pred_norm_batches, axis=0)
    y_raw = np.concatenate(y_raw_batches, axis=0)
    x_raw = np.concatenate(x_raw_batches, axis=0)

    metrics = compute_physical_metrics(pred_norm, y_raw, scaler_y, output_dim, active_threshold)
    metrics["normalized_mse"] = float(total_mse / max(total_elements, 1))
    metrics["normalized_smooth_l1"] = float(total_smooth_l1 / max(total_elements, 1))

    if collect_gates and z_last_batches:
        z_last = np.concatenate(z_last_batches, axis=0)
        metrics["gate_auroc"] = gate_auroc(z_last, x_raw)
        pred_raw = scaler_y.inverse_transform(pred_norm.reshape(-1, output_dim)).reshape(pred_norm.shape)
        metrics["cyclic_energy_relative_error"] = cyclic_energy_relative_error(pred_raw, y_raw, x_raw)
    return metrics


def load_checkpoint_state(path: str | os.PathLike, map_location="cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint.get("model_state_dict", checkpoint)


def _best_epoch_from_history(history: list[dict]) -> int:
    best_idx = min(range(len(history)), key=lambda i: history[i]["val_loss"])
    return int(history[best_idx]["epoch"])


def _run_config_base(
    args,
    manifest: dict,
    digest: str,
    variant: str,
    extra: dict | None = None,
    runtime: dict | None = None,
) -> dict:
    config = {
        "protocol_version": manifest.get("protocol_version"),
        "manifest_path": str(getattr(args, "split_manifest", "")),
        "manifest_sha256": digest,
        "seed": int(args.seed),
        "variant": variant,
        "lambda_gate": float(args.lambda_gate),
        "gate_threshold": float(args.gate_threshold),
        "training": {
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.epochs),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "grad_clip": float(args.grad_clip),
            "loss": args.loss,
            "input_dim": int(args.input_dim),
            "output_dim": int(args.output_dim),
            "d_model": int(args.d_model),
            "num_layers": int(args.num_layers),
            "mat_embed_dim": int(args.mat_embed_dim),
            "d_state": int(args.d_state),
            "dt_rank": int(args.dt_rank),
            "seq_len": int(args.seq_len),
        },
        "counts": {
            "train": len(manifest.get("train", [])),
            "validation": len(manifest.get("validation", [])),
            "test": len(manifest.get("test", [])),
        },
        "evaluate_test_once": bool(getattr(args, "evaluate_test_once", False)),
    }
    if runtime is not None:
        config["runtime"] = runtime
    if extra:
        config.update(extra)
    return config


def write_provenance(
    run_dir: Path,
    args,
    manifest: dict,
    digest: str,
    variant: str = "full",
    extra: dict | None = None,
    runtime: dict | None = None,
) -> None:
    """Write the static provenance files before training starts.

    ``run_config.json`` is written here without the result fields (a crashed
    run still records its provenance) and rewritten by
    :func:`finalize_run_config` at the end of the run.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (run_dir / "manifest.sha256").write_text(digest + "\n")

    env_lines = [
        f"python {sys.version.split()[0]} ({sys.executable})",
        f"torch {torch.__version__}",
        f"cuda_available {torch.cuda.is_available()}",
        f"cuda_build {torch.version.cuda}",
        f"cuda_visible_devices {os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
    ]
    if torch.cuda.is_available():
        try:
            env_lines.append("gpu_index 0")
            env_lines.append(f"gpu {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    try:
        import subprocess

        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if freeze.returncode == 0:
            env_lines.append("")
            env_lines.append("--- pip freeze ---")
            env_lines.extend(freeze.stdout.strip().splitlines())
    except Exception as exc:  # environment.txt must never kill a run
        env_lines.append(f"pip freeze unavailable: {exc}")
    (run_dir / "environment.txt").write_text("\n".join(env_lines) + "\n")

    dump_json(run_dir / "run_config.json", _run_config_base(args, manifest, digest, variant, extra, runtime))


def finalize_run_config(
    run_dir: Path,
    args,
    manifest: dict,
    digest: str,
    variant: str,
    best_epoch: int,
    best_val_loss: float,
    test_metrics: dict | None,
    extra: dict | None = None,
    runtime: dict | None = None,
) -> None:
    config = _run_config_base(args, manifest, digest, variant, extra, runtime)
    config["best_epoch"] = int(best_epoch)
    config["best_val_loss"] = float(best_val_loss)
    if test_metrics is not None:
        config["test_metrics"] = test_metrics
    dump_json(Path(run_dir) / "run_config.json", config)


def update_runtime_provenance(run_dir: Path, runtime: dict) -> None:
    """Update a pre-training run config after an exception without hiding it."""
    config_path = Path(run_dir) / "run_config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text())
    config["runtime"] = runtime
    dump_json(config_path, config)


def build_canonical_model(args, n_materials: int, device: torch.device) -> nn.Module:
    return StressPredictorMamba(
        input_dim=args.input_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        n_materials=n_materials,
        mat_embed_dim=args.mat_embed_dim,
        d_state=args.d_state,
        dt_rank=args.dt_rank,
    ).to(device)


def _wrap_ddp(model: nn.Module, device: torch.device, distributed: bool) -> nn.Module:
    if not distributed:
        return model
    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=True,
    )


def _runtime_for_device(
    device: torch.device,
    started_at: str,
    ended_at: str | None,
    exit_code: int | None,
) -> dict:
    gpu_index = int(device.index) if device.type == "cuda" and device.index is not None else None
    gpu_name = None
    if gpu_index is not None and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(gpu_index)
        except Exception:
            gpu_name = None
    return runtime_provenance(
        started_at=started_at,
        ended_at=ended_at,
        exit_code=exit_code,
        gpu_index=gpu_index,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_name=gpu_name,
    )


def run_manifest_core(args, model_builder, variant: str, extra: dict | None = None) -> None:
    """Shared revision-v2 training core (plan Task 4).

    ``model_builder(args, n_materials, device)`` returns an unwrapped module.
    Flow: manifest splits -> train-only scalers -> train with validation-loss
    early stopping -> reload best checkpoint -> one-shot validation metrics
    -> (if ``--evaluate_test_once``) one-shot test metrics -> provenance.
    No runtime random split happens anywhere in this path.
    """
    started_at = utc_now_iso()
    set_seed(args.seed)
    distributed, rank, _local_rank, world_size, device = setup_distributed()

    project_root = resolve_project_root()
    manifest = load_manifest(args.split_manifest)
    digest = manifest_sha256(args.split_manifest)
    run_dir = Path(args.output_root)

    args.model_save_path = str(run_dir / "best_model.pth")
    args.scaler_x_path = str(run_dir / "scaler_x.joblib")
    args.scaler_y_path = str(run_dir / "scaler_y.joblib")
    args.history_path = str(run_dir / "training_history.csv")

    try:
        if is_main_process(rank):
            mapping = build_mapping_from_manifest(project_root, manifest)
            save_json_mapping(mapping, run_dir / "material_mapping.json")
            write_provenance(
                run_dir,
                args,
                manifest,
                digest,
                variant=variant,
                extra=extra,
                runtime=_runtime_for_device(device, started_at, None, None),
            )

        if distributed:
            dist.barrier()

        x_train, y_train, mat_train = load_manifest_split(
            project_root,
            manifest,
            mapping,
            "train",
            seq_len=args.seq_len,
            strain_zero_threshold=args.strain_zero_threshold,
            stress_zero_threshold=args.stress_zero_threshold,
        )
        x_val, y_val, mat_val = load_manifest_split(
            project_root,
            manifest,
            mapping,
            "validation",
            seq_len=args.seq_len,
            strain_zero_threshold=args.strain_zero_threshold,
            stress_zero_threshold=args.stress_zero_threshold,
        )

        scaler_x, scaler_y = fit_scalers_from_train_only(x_train, y_train)

        def _fit(arr: np.ndarray, scaler) -> np.ndarray:
            return scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)

        train_loader, val_loader, train_sampler = make_split_loaders(
            _fit(x_train, scaler_x),
            _fit(y_train, scaler_y),
            mat_train,
            _fit(x_val, scaler_x),
            _fit(y_val, scaler_y),
            mat_val,
            args,
            distributed,
            world_size,
            rank,
        )

        model = model_builder(args, n_materials=len(mapping), device=device)
        model = _wrap_ddp(model, device, distributed)

        history, best_val_loss = train_model(
            model,
            train_loader,
            val_loader,
            train_sampler,
            args,
            device,
            rank=rank,
        )
        best_epoch = _best_epoch_from_history(history)

        if is_main_process(rank):
            Path(args.scaler_x_path).parent.mkdir(parents=True, exist_ok=True)
            Path(args.scaler_y_path).parent.mkdir(parents=True, exist_ok=True)
            dump(scaler_x, args.scaler_x_path)
            dump(scaler_y, args.scaler_y_path)
            save_history(history, args.history_path)

            # Reload the best checkpoint, then evaluate validation and test
            # exactly once each (test only when requested).
            best_model = model_builder(args, n_materials=len(mapping), device=device)
            best_model.load_state_dict(load_checkpoint_state(args.model_save_path, map_location=device))

            def _eval_loader(x_raw: np.ndarray, y_raw: np.ndarray, mat: np.ndarray) -> DataLoader:
                dataset = StrainStressEvalDataset(
                    _fit(x_raw, scaler_x),
                    _fit(y_raw, scaler_y),
                    x_raw,
                    y_raw,
                    mat,
                )
                return DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                )

            val_metrics = evaluate_split_once(
                best_model,
                _eval_loader(x_val, y_val, mat_val),
                scaler_y,
                device,
                args.output_dim,
                args.active_threshold,
                collect_gates=True,
            )
            validation_metrics = {
                "split": "validation",
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
                "note": "one-shot evaluation of the best checkpoint on the validation split",
                "metrics": val_metrics,
            }
            dump_json(run_dir / "validation_metrics.json", validation_metrics)

            test_metrics = None
            if args.evaluate_test_once:
                x_test, y_test, mat_test = load_manifest_split(
                    project_root,
                    manifest,
                    mapping,
                    "test",
                    seq_len=args.seq_len,
                    strain_zero_threshold=args.strain_zero_threshold,
                    stress_zero_threshold=args.stress_zero_threshold,
                )
                test_metrics = evaluate_split_once(
                    best_model,
                    _eval_loader(x_test, y_test, mat_test),
                    scaler_y,
                    device,
                    args.output_dim,
                    args.active_threshold,
                    collect_gates=True,
                )
                test_metrics["split"] = "test"
                dump_json(run_dir / "test_metrics.json", test_metrics)

            finalize_run_config(
                run_dir,
                args,
                manifest,
                digest,
                variant=variant,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                test_metrics=test_metrics,
                extra=extra,
                runtime=_runtime_for_device(device, started_at, utc_now_iso(), 0),
            )
            if is_main_process(rank):
                print(
                    json.dumps(
                        {
                            "run_dir": str(run_dir),
                            "variant": variant,
                            "best_epoch": int(best_epoch),
                            "best_val_loss": float(best_val_loss),
                        },
                        indent=2,
                    )
                )
    except Exception:
        if is_main_process(rank):
            update_runtime_provenance(
                run_dir,
                _runtime_for_device(device, started_at, utc_now_iso(), 1),
            )
        raise
    finally:
        cleanup_distributed(distributed)


def run_training_manifest(args) -> None:
    """revision-v2 entry point for the canonical (full-gate) PiG-SSM model."""
    if args.split_manifest is None:
        raise SystemExit(
            "--split_manifest is required in revision-v2 mode "
            "(pass --legacy_random_split to use the historical random-split path)"
        )
    if args.seed is None:
        raise SystemExit("--seed is required in revision-v2 mode")
    if args.output_root is None:
        raise SystemExit("--output_root is required in revision-v2 mode (the run directory)")
    run_manifest_core(args, build_canonical_model, variant=args.variant)


def save_history(history: list[dict], history_path: str | os.PathLike) -> None:
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "gate_loss",
                "combined_loss",
                "val_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def checkpoint_state(model: nn.Module) -> dict:
    model_to_save = model.module if hasattr(model, "module") else model
    return {"model_state_dict": model_to_save.state_dict()}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_sampler,
    args,
    device: torch.device,
    rank: int = 0,
):
    criterion = build_criterion(args.loss)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    early_stopping_counter = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model,
            criterion,
            optimizer,
            train_loader,
            device,
            lambda_gate=args.lambda_gate,
            gate_threshold=args.gate_threshold,
            grad_clip=args.grad_clip,
        )
        val_loss = validate(model, criterion, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "gate_loss": train_metrics["gate_loss"],
            "combined_loss": train_metrics["combined_loss"],
            "val_loss": val_loss,
        }
        history.append(row)

        if is_main_process(rank):
            print(
                f"Epoch {epoch:04d} | "
                f"Train: {row['train_loss']:.4e} | "
                f"Gate: {row['gate_loss']:.4e} | "
                f"Combined: {row['combined_loss']:.4e} | "
                f"Val: {row['val_loss']:.4e}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            if is_main_process(rank):
                Path(args.model_save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(checkpoint_state(model), args.model_save_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                if is_main_process(rank):
                    print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    return history, best_val_loss


def run_training(args):
    """Dispatch: revision-v2 manifest path by default, legacy behind a flag."""
    if getattr(args, "legacy_random_split", False):
        run_training_legacy(args)
    else:
        run_training_manifest(args)


def run_training_legacy(args):
    """Historical random 80/20 split flow; kept behind --legacy_random_split only.

    Never used by any paper command in the revision-v2 protocol.
    """
    if args.seed is None:
        args.seed = 42
    if args.data_dirs is None:
        raise SystemExit("--data_dirs is required for --legacy_random_split")
    set_seed(args.seed)
    distributed, rank, _local_rank, world_size, device = setup_distributed()

    try:
        if is_main_process(rank):
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            create_or_update_material_mapping(
                args.data_dirs,
                args.mapping_path,
                reset_mapping=args.reset_mapping,
            )

        if distributed:
            dist.barrier()

        mapping = load_json_mapping(args.mapping_path)
        x, y, material_ids = load_raw_sequences(
            args.data_dirs,
            args.mapping_path,
            seq_len=args.seq_len,
            strain_zero_threshold=args.strain_zero_threshold,
            stress_zero_threshold=args.stress_zero_threshold,
            skip_header=args.skip_header,
            max_sequences=getattr(args, "max_sequences", None),
        )

        train_loader, val_loader, train_sampler, scaler_x, scaler_y = make_data_loaders(
            x, y, material_ids, args, distributed, world_size, rank
        )

        model = StressPredictorMamba(
            input_dim=args.input_dim,
            d_model=args.d_model,
            num_layers=args.num_layers,
            output_dim=args.output_dim,
            n_materials=len(mapping),
            mat_embed_dim=args.mat_embed_dim,
            d_state=args.d_state,
            dt_rank=args.dt_rank,
        ).to(device)

        if distributed:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                find_unused_parameters=True,
            )

        history, best_val_loss = train_model(
            model,
            train_loader,
            val_loader,
            train_sampler,
            args,
            device,
            rank=rank,
        )

        if is_main_process(rank):
            Path(args.scaler_x_path).parent.mkdir(parents=True, exist_ok=True)
            Path(args.scaler_y_path).parent.mkdir(parents=True, exist_ok=True)
            dump(scaler_x, args.scaler_x_path)
            dump(scaler_y, args.scaler_y_path)
            save_history(history, args.history_path)
            summary = {
                "mode": "pretrain",
                "data_dirs": [str(path) for path in args.data_dirs],
                "mapping_path": str(args.mapping_path),
                "model_save_path": str(args.model_save_path),
                "scaler_x_path": str(args.scaler_x_path),
                "scaler_y_path": str(args.scaler_y_path),
                "num_sequences": int(len(x)),
                "num_materials": int(len(mapping)),
                "best_val_loss": float(best_val_loss),
                "num_parameters": int(count_parameters(model.module if hasattr(model, "module") else model)),
                "lambda_gate": float(args.lambda_gate),
                "gate_threshold": float(args.gate_threshold),
                "d_model": int(args.d_model),
                "num_layers": int(args.num_layers),
                "d_state": int(args.d_state),
                "dt_rank": int(args.dt_rank),
            }
            with open(args.summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(json.dumps(summary, indent=2))
    finally:
        cleanup_distributed(distributed)


def parse_args():
    default_output_dir = SCRIPT_DIR / "outputs" / "pretrain"
    parser = argparse.ArgumentParser(description="Train PiG-SSM with gate regularization.")
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        default=None,
        help="legacy mode only; revision-v2 loads files listed in --split_manifest",
    )
    parser.add_argument("--output_dir", default=str(default_output_dir))
    parser.add_argument("--mapping_path", default=str(default_output_dir / "material_mapping.json"))
    parser.add_argument("--model_save_path", default=str(default_output_dir / "best_model.pth"))
    parser.add_argument("--scaler_x_path", default=str(default_output_dir / "scaler_x.joblib"))
    parser.add_argument("--scaler_y_path", default=str(default_output_dir / "scaler_y.joblib"))
    parser.add_argument("--history_path", default=str(default_output_dir / "training_history.csv"))
    parser.add_argument("--summary_path", default=str(default_output_dir / "training_summary.json"))
    parser.add_argument("--reset_mapping", action="store_true")
    parser.add_argument("--skip_header", action="store_true")

    parser.add_argument("--seq_len", type=int, default=101)
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--output_dim", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--mat_embed_dim", type=int, default=32)
    parser.add_argument("--d_state", type=int, default=7)
    parser.add_argument("--dt_rank", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--loss", choices=["smooth_l1", "mse"], default="smooth_l1")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_gate", type=float, default=0.001)
    parser.add_argument("--gate_threshold", type=float, default=1e-4)
    parser.add_argument("--strain_zero_threshold", type=float, default=0.01)
    parser.add_argument("--stress_zero_threshold", type=float, default=50.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="required in revision-v2 mode; legacy mode defaults to 42",
    )

    # revision-v2 controlled protocol (plan Task 4)
    parser.add_argument(
        "--split_manifest",
        type=Path,
        default=None,
        help="revision-v2 split manifest JSON (required unless --legacy_random_split)",
    )
    parser.add_argument(
        "--evaluate_test_once",
        action="store_true",
        help="evaluate the test split exactly once after loading the best checkpoint",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="revision-v2 run directory (all provenance files are written here)",
    )
    parser.add_argument(
        "--variant",
        default="full",
        help="run label recorded in run_config.json (e.g. full, lambda_1e-3)",
    )
    parser.add_argument(
        "--active_threshold",
        type=float,
        default=1e-3,
        help="MPa peak-to-peak threshold for stress-active components",
    )
    parser.add_argument(
        "--legacy_random_split",
        action="store_true",
        help="use the historical random 80/20 split (excluded from all paper commands)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())

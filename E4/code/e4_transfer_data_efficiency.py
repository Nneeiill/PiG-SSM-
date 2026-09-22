"""E4 transfer-learning data-efficiency experiment.

For each target material dataset, this script:
1. splits complete CSV sequences into a fixed train_pool/test split,
2. samples a specified fraction from train_pool,
3. trains either a scratch model or a pretrained-transfer model,
4. evaluates the best checkpoint on the fixed test split in physical units.

The split unit is always the complete CSV loading history.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib import load
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SCRIPT_PATH = Path(__file__).resolve()
E4_DIR = SCRIPT_PATH.parents[1]
ORGANIZED_DIR = SCRIPT_PATH.parents[2]
PROJECT_ROOT = ORGANIZED_DIR.parent

if str(ORGANIZED_DIR) not in sys.path:
    sys.path.insert(0, str(ORGANIZED_DIR))

from pig_ssm_model import StressPredictorMamba, count_parameters
from train_pig_ssm import (
    gate_regularization_loss,
    load_csv_array,
    material_key_from_filename,
    natural_key,
)


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


def load_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def list_target_files(data_dir: str | Path) -> list[Path]:
    return sorted(Path(data_dir).glob("*.csv"), key=natural_key)


def extend_mapping(base_mapping_path: str | Path, files: list[Path], output_path: str | Path) -> dict[str, int]:
    mapping = load_json(base_mapping_path)
    next_idx = len(mapping)
    for path in files:
        key = material_key_from_filename(path)
        if key is not None and key not in mapping:
            mapping[key] = next_idx
            next_idx += 1
    save_json(mapping, output_path)
    return mapping


def load_sequences_from_files(
    files: list[Path],
    mapping: dict[str, int],
    seq_len: int,
    strain_zero_threshold: float,
    stress_zero_threshold: float,
):
    x_list, y_list, mat_ids, kept_files = [], [], [], []
    for file_path in files:
        key = material_key_from_filename(file_path)
        if key not in mapping:
            continue
        try:
            data = load_csv_array(file_path, skip_header=None)
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
        mat_ids.append(mapping[key])
        kept_files.append(str(file_path))
    if not x_list:
        raise ValueError(f"No usable sequences in {files[:3]} ...")
    return np.array(x_list), np.array(y_list), np.array(mat_ids), kept_files


def migrate_embedding_state(new_model: StressPredictorMamba, old_state: dict) -> dict:
    new_state = new_model.state_dict()
    for name, param in old_state.items():
        if name == "material_embed.weight" and name in new_state:
            rows = min(new_state[name].shape[0], param.shape[0])
            new_state[name][:rows].copy_(param[:rows])
        elif name in new_state and new_state[name].shape == param.shape:
            new_state[name].copy_(param)
    return new_state


def load_checkpoint_state(path: str | Path, map_location="cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint.get("model_state_dict", checkpoint)


def build_model(args, n_materials: int, mode: str, device: torch.device):
    model = StressPredictorMamba(
        input_dim=args.input_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        n_materials=n_materials,
        mat_embed_dim=args.mat_embed_dim,
        d_state=args.d_state,
        dt_rank=args.dt_rank,
    )
    if mode == "transfer":
        old_state = load_checkpoint_state(args.pretrained_path, map_location="cpu")
        model.load_state_dict(migrate_embedding_state(model, old_state))
    return model.to(device)


def normalize_with_source_scalers(x, y, scaler_x, scaler_y, args):
    x_norm = scaler_x.transform(x.reshape(-1, args.input_dim)).reshape(x.shape)
    y_norm = scaler_y.transform(y.reshape(-1, args.output_dim)).reshape(y.shape)
    return x_norm, y_norm


def make_loader(x, y, mat_ids, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        StrainStressDataset(x, y, mat_ids),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(model, loader, criterion, optimizer, device, args):
    model.train()
    total_task, total_gate, total_combined = 0.0, 0.0, 0.0
    for batch_x, batch_y, batch_mat in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_mat = batch_mat.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred, all_z, _ = model(batch_x, material_id=batch_mat, return_gate=True)
        task_loss = criterion(pred, batch_y)
        gate_loss = gate_regularization_loss(all_z, batch_x, gate_threshold=args.gate_threshold)
        combined_loss = task_loss + args.lambda_gate * gate_loss
        combined_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        n = batch_x.size(0)
        total_task += task_loss.item() * n
        total_gate += gate_loss.item() * n
        total_combined += combined_loss.item() * n
    denom = max(len(loader.dataset), 1)
    return total_task / denom, total_gate / denom, total_combined / denom


def evaluate_norm_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch_x, batch_y, batch_mat in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_mat = batch_mat.to(device, non_blocking=True)
            pred = model(batch_x, material_id=batch_mat, return_gate=False)
            total += criterion(pred, batch_y).item() * batch_x.size(0)
    return total / max(len(loader.dataset), 1)


def evaluate_physical_metrics(model, loader, scaler_y, device, args):
    model.eval()
    pred_batches, true_batches = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_mat in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_mat = batch_mat.to(device, non_blocking=True)
            pred = model(batch_x, material_id=batch_mat, return_gate=False).cpu().numpy()
            pred_batches.append(pred)
            true_batches.append(batch_y.numpy())

    pred_norm = np.concatenate(pred_batches, axis=0)
    true_norm = np.concatenate(true_batches, axis=0)
    n, t, d = pred_norm.shape
    pred = scaler_y.inverse_transform(pred_norm.reshape(-1, d)).reshape(n, t, d)
    true = scaler_y.inverse_transform(true_norm.reshape(-1, d)).reshape(n, t, d)

    abs_error = np.abs(pred - true)
    mae_all = float(abs_error.mean())
    rmse_all = float(np.sqrt(np.mean((pred - true) ** 2)))
    max_error_all = float(abs_error.max())

    active_abs_errors = []
    active_sq_errors = []
    component_records = {}
    for j in range(d):
        active_mask = np.ptp(true[:, :, j], axis=1) > args.active_stress_threshold
        if active_mask.any():
            diff = pred[active_mask, :, j] - true[active_mask, :, j]
            active_abs_errors.append(np.abs(diff).reshape(-1))
            active_sq_errors.append((diff ** 2).reshape(-1))
            component_records[f"MAE_active_comp_{j}"] = float(np.mean(np.abs(diff)))
            component_records[f"RMSE_active_comp_{j}"] = float(np.sqrt(mean_squared_error(true[active_mask, :, j].reshape(-1), pred[active_mask, :, j].reshape(-1))))
        else:
            component_records[f"MAE_active_comp_{j}"] = None
            component_records[f"RMSE_active_comp_{j}"] = None

    if active_abs_errors:
        active_abs = np.concatenate(active_abs_errors)
        active_sq = np.concatenate(active_sq_errors)
        mae_active = float(active_abs.mean())
        rmse_active = float(np.sqrt(active_sq.mean()))
        max_error_active = float(active_abs.max())
    else:
        mae_active = rmse_active = max_error_active = float("nan")

    metrics = {
        "MAE_All6": mae_all,
        "RMSE_All6": rmse_all,
        "MaxError_All6": max_error_all,
        "MAE_ActiveOnly": mae_active,
        "RMSE_ActiveOnly": rmse_active,
        "MaxError_ActiveOnly": max_error_active,
    }
    metrics.update(component_records)
    return metrics


def write_history(history, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "gate_loss", "combined_loss", "val_loss"],
        )
        writer.writeheader()
        writer.writerows(history)


def write_split_files(train_files, val_files, test_files, output_dir: Path):
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, files in [("train.txt", train_files), ("val.txt", val_files), ("test.txt", test_files)]:
        (split_dir / name).write_text("\n".join(map(str, files)) + "\n")


def parse_target_name(data_dir: str | Path) -> str:
    return Path(data_dir).name.replace("/", "_")


def resolve_protocol_version(explicit: str | None) -> str:
    """Frozen protocol version for run_config provenance (revision-v2)."""
    if explicit:
        return explicit
    cfg = ORGANIZED_DIR / "configs" / "revision_protocol.json"
    if cfg.is_file():
        return json.load(cfg.open())["protocol_version"]
    raise SystemExit("no --protocol_version given and configs/revision_protocol.json not found")


def run(args):
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    target_name = args.target_name or parse_target_name(args.data_dir)
    ratio_tag = str(args.ratio).replace(".", "p")
    run_name = f"{target_name}_r{ratio_tag}_{args.mode}_seed{args.seed}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_target_files(args.data_dir)
    if len(files) < 10:
        raise ValueError(f"Too few CSV files in {args.data_dir}: {len(files)}")

    mapping_path = output_dir / "material_mapping.json"
    mapping = extend_mapping(args.base_mapping_path, files, mapping_path)

    train_pool_files, test_files = train_test_split(
        files,
        test_size=args.test_split,
        random_state=args.seed,
        shuffle=True,
    )
    train_count = max(1, int(round(len(train_pool_files) * args.ratio)))
    rng = np.random.default_rng(args.seed + int(round(args.ratio * 1000)))
    selected_indices = rng.choice(len(train_pool_files), size=train_count, replace=False)
    trainval_files = [train_pool_files[i] for i in sorted(selected_indices)]

    val_size = max(1, int(round(len(trainval_files) * args.val_split)))
    if len(trainval_files) <= 2:
        train_files, val_files = trainval_files, trainval_files
    else:
        train_files, val_files = train_test_split(
            trainval_files,
            test_size=val_size,
            random_state=args.seed,
            shuffle=True,
        )

    # revision-v2 provenance (plan Task 7): run_config.json + command.txt +
    # environment.txt are written BEFORE training so a crashed run is
    # diagnosable and restart-safety can match the protocol version. Split
    # file-list hashes pin the fixed-test-set contract (Step 3): per
    # (target, seed) the test list must be identical across ratios and modes,
    # and the train/val lists identical between scratch and transfer.
    split_hashes = {
        name: hashlib.sha256(("\n".join(map(str, files)) + "\n").encode()).hexdigest()
        for name, files in (("train", train_files), ("val", val_files), ("test", test_files))
    }
    save_json(
        {
            "protocol_version": resolve_protocol_version(args.protocol_version),
            "run_name": run_name,
            "target": target_name,
            "mode": args.mode,
            "ratio": float(args.ratio),
            "seed": int(args.seed),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "split_file_list_hashes_sha256": split_hashes,
            "n_files": {
                "train_pool": len(train_pool_files),
                "train": len(train_files),
                "val": len(val_files),
                "test": len(test_files),
            },
        },
        output_dir / "run_config.json",
    )
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
    (output_dir / "environment.txt").write_text(
        f"python={platform.python_version()}\n"
        f"torch={torch.__version__}\n"
        f"cuda_available={torch.cuda.is_available()}\n"
        f"cuda_device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}\n"
        f"device={str(device)}\n"
    )

    x_train, y_train, m_train, kept_train = load_sequences_from_files(
        train_files,
        mapping,
        args.seq_len,
        args.strain_zero_threshold,
        args.stress_zero_threshold,
    )
    x_val, y_val, m_val, kept_val = load_sequences_from_files(
        val_files,
        mapping,
        args.seq_len,
        args.strain_zero_threshold,
        args.stress_zero_threshold,
    )
    x_test, y_test, m_test, kept_test = load_sequences_from_files(
        test_files,
        mapping,
        args.seq_len,
        args.strain_zero_threshold,
        args.stress_zero_threshold,
    )

    scaler_x = load(args.scaler_x_path)
    scaler_y = load(args.scaler_y_path)
    x_train, y_train = normalize_with_source_scalers(x_train, y_train, scaler_x, scaler_y, args)
    x_val, y_val = normalize_with_source_scalers(x_val, y_val, scaler_x, scaler_y, args)
    x_test, y_test = normalize_with_source_scalers(x_test, y_test, scaler_x, scaler_y, args)

    train_loader = make_loader(x_train, y_train, m_train, args.batch_size, True, args.num_workers)
    val_loader = make_loader(x_val, y_val, m_val, args.batch_size, False, args.num_workers)
    test_loader = make_loader(x_test, y_test, m_test, args.batch_size, False, args.num_workers)

    model = build_model(args, n_materials=len(mapping), mode=args.mode, device=device)
    criterion = nn.SmoothL1Loss() if args.loss == "smooth_l1" else nn.MSELoss()
    lr = args.lr_transfer if args.mode == "transfer" else args.lr_scratch
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    model_path = output_dir / "best_model.pth"
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, gate_loss, combined_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args
        )
        val_loss = evaluate_norm_loss(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "gate_loss": gate_loss,
            "combined_loss": combined_loss,
            "val_loss": val_loss,
        }
        history.append(row)
        print(
            f"{run_name} | Epoch {epoch:04d} | "
            f"Train {train_loss:.4e} | Gate {gate_loss:.4e} | Val {val_loss:.4e}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict()}, model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"{run_name} | Early stopping at epoch {epoch}.", flush=True)
                break

    model.load_state_dict(load_checkpoint_state(model_path, map_location=device))
    test_norm_loss = evaluate_norm_loss(model, test_loader, criterion, device)
    test_metrics = evaluate_physical_metrics(model, test_loader, scaler_y, device, args)

    write_history(history, output_dir / "history.csv")
    write_split_files(kept_train, kept_val, kept_test, output_dir)

    summary = {
        "run_name": run_name,
        "target": target_name,
        "data_dir": str(args.data_dir),
        "mode": args.mode,
        "ratio": float(args.ratio),
        "seed": int(args.seed),
        "test_split": float(args.test_split),
        "val_split_within_selected_train": float(args.val_split),
        "num_total_files": int(len(files)),
        "num_train_pool_files": int(len(train_pool_files)),
        "num_selected_trainval_files": int(len(trainval_files)),
        "num_train_sequences": int(len(x_train)),
        "num_val_sequences": int(len(x_val)),
        "num_test_sequences": int(len(x_test)),
        "num_materials": int(len(mapping)),
        "mapping_path": str(mapping_path),
        "pretrained_path": str(args.pretrained_path) if args.mode == "transfer" else None,
        "model_path": str(model_path),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history)),
        "test_norm_loss": float(test_norm_loss),
        "lr": float(lr),
        "lambda_gate": float(args.lambda_gate),
        "num_parameters": int(count_parameters(model)),
        "trainable_parameters": int(count_parameters(model, trainable_only=True)),
    }
    summary.update(test_metrics)
    save_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="E4 transfer data-efficiency run.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--target_name", default=None)
    parser.add_argument("--mode", choices=["scratch", "transfer"], required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--output_dir", default=str(E4_DIR / "results" / "runs"))
    parser.add_argument("--base_mapping_path", default=str(PROJECT_ROOT / "cross-materials" / "material_mapping.json"))
    parser.add_argument("--pretrained_path", default=str(PROJECT_ROOT / "cross-materials" / "best_model_mamba_cross-2.pth"))
    parser.add_argument("--scaler_x_path", default=str(PROJECT_ROOT / "cross-materials" / "scaler_x.joblib"))
    parser.add_argument("--scaler_y_path", default=str(PROJECT_ROOT / "cross-materials" / "scaler_y.joblib"))
    parser.add_argument("--device", default=None)

    parser.add_argument("--seq_len", type=int, default=101)
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--output_dim", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--mat_embed_dim", type=int, default=32)
    parser.add_argument("--d_state", type=int, default=7)
    parser.add_argument("--dt_rank", type=int, default=5)

    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_transfer", type=float, default=1e-5)
    parser.add_argument("--lr_scratch", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--loss", choices=["smooth_l1", "mse"], default="smooth_l1")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_gate", type=float, default=0.001)
    parser.add_argument("--gate_threshold", type=float, default=1e-4)
    parser.add_argument("--strain_zero_threshold", type=float, default=0.01)
    parser.add_argument("--stress_zero_threshold", type=float, default=50.0)
    parser.add_argument("--active_stress_threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protocol_version",
        default=None,
        help="frozen protocol version recorded in run_config.json "
        "(default: read from configs/revision_protocol.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

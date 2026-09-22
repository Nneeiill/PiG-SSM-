"""Evaluate full PiG-SSM and E1 ablation checkpoints on a common split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from joblib import load
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
E1_DIR = SCRIPT_DIR.parent
ORGANIZED_DIR = E1_DIR.parent
PROJECT_ROOT = ORGANIZED_DIR.parent

if str(ORGANIZED_DIR) not in sys.path:
    sys.path.insert(0, str(ORGANIZED_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from e1_ablation_train import StressPredictorMambaGateAblation
from pig_ssm_model import StressPredictorMamba
from revision_pipeline.paths import resolve_project_root
from train_pig_ssm import (
    StrainStressEvalDataset,
    dump_json,
    evaluate_split_once,
    load_checkpoint_state,
    load_json_mapping,
    load_manifest,
    load_manifest_split,
    load_raw_sequences,
)


class EvalDataset(Dataset):
    def __init__(self, x_norm: np.ndarray, y_norm: np.ndarray, y_raw: np.ndarray, material_ids: np.ndarray):
        self.x_norm = torch.FloatTensor(x_norm)
        self.y_norm = torch.FloatTensor(y_norm)
        self.y_raw = torch.FloatTensor(y_raw)
        self.material_ids = torch.LongTensor(material_ids)

    def __len__(self) -> int:
        return len(self.x_norm)

    def __getitem__(self, idx: int):
        return self.x_norm[idx], self.y_norm[idx], self.y_raw[idx], self.material_ids[idx]


def load_checkpoint_state(path: str | Path, map_location="cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint.get("model_state_dict", checkpoint)


def build_eval_model(name: str, n_materials: int, args, device: torch.device):
    if name in {"full", "dynamic_no_gate_reg"}:
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
    elif name == "static_gate":
        model = StressPredictorMambaGateAblation(
            input_dim=args.input_dim,
            d_model=args.d_model,
            num_layers=args.num_layers,
            output_dim=args.output_dim,
            n_materials=n_materials,
            mat_embed_dim=args.mat_embed_dim,
            d_state=args.d_state,
            dt_rank=args.dt_rank,
            gate_mode="static",
        )
    elif name == "no_gate":
        model = StressPredictorMambaGateAblation(
            input_dim=args.input_dim,
            d_model=args.d_model,
            num_layers=args.num_layers,
            output_dim=args.output_dim,
            n_materials=n_materials,
            mat_embed_dim=args.mat_embed_dim,
            d_state=args.d_state,
            dt_rank=args.dt_rank,
            gate_mode="none",
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

    model.to(device)
    return model


def active_component_mask(y_true_raw: np.ndarray, active_threshold: float):
    # Per-sample, per-component activity mask based on stress variation.
    return np.ptp(y_true_raw, axis=1) > active_threshold


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8):
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def evaluate_model(name: str, checkpoint_path: Path, loader: DataLoader, scaler_y, n_materials: int, args, device):
    model = build_eval_model(name, n_materials=n_materials, args=args, device=device)
    state = load_checkpoint_state(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    mse_loss = nn.MSELoss(reduction="sum")
    smooth_l1 = nn.SmoothL1Loss(reduction="sum")

    total_mse = 0.0
    total_smooth_l1 = 0.0
    total_elements = 0
    pred_raw_batches = []
    y_raw_batches = []

    with torch.no_grad():
        for x_norm, y_norm, y_raw, material_ids in loader:
            x_norm = x_norm.to(device, non_blocking=True)
            y_norm = y_norm.to(device, non_blocking=True)
            material_ids = material_ids.to(device, non_blocking=True)

            pred_norm = model(x_norm, material_ids)
            total_mse += mse_loss(pred_norm, y_norm).item()
            total_smooth_l1 += smooth_l1(pred_norm, y_norm).item()
            total_elements += y_norm.numel()

            pred_np = pred_norm.detach().cpu().numpy()
            pred_raw = scaler_y.inverse_transform(pred_np.reshape(-1, args.output_dim)).reshape(pred_np.shape)
            pred_raw_batches.append(pred_raw)
            y_raw_batches.append(y_raw.numpy())

    pred_raw = np.concatenate(pred_raw_batches, axis=0)
    y_raw = np.concatenate(y_raw_batches, axis=0)
    abs_error = np.abs(pred_raw - y_raw)

    active_mask = active_component_mask(y_raw, active_threshold=args.active_threshold)
    active_errors = []
    active_true = []
    active_pred = []
    for sample_idx in range(y_raw.shape[0]):
        comps = active_mask[sample_idx]
        if np.any(comps):
            active_errors.append(abs_error[sample_idx, :, comps])
            active_true.append(y_raw[sample_idx, :, comps])
            active_pred.append(pred_raw[sample_idx, :, comps])

    if active_errors:
        active_errors_flat = np.concatenate([x.reshape(-1) for x in active_errors])
        active_true_flat = np.concatenate([x.reshape(-1) for x in active_true])
        active_pred_flat = np.concatenate([x.reshape(-1) for x in active_pred])
        active_mae = float(np.mean(active_errors_flat))
        active_rmse = float(np.sqrt(np.mean((active_pred_flat - active_true_flat) ** 2)))
        active_max_error = float(np.max(active_errors_flat))
        active_mape = float(safe_mape(active_true_flat, active_pred_flat))
    else:
        active_mae = active_rmse = active_max_error = active_mape = float("nan")

    per_component = {}
    for comp in range(args.output_dim):
        comp_true = y_raw[:, :, comp].reshape(-1)
        comp_pred = pred_raw[:, :, comp].reshape(-1)
        comp_err = np.abs(comp_pred - comp_true)
        per_component[f"mae_s{comp}"] = float(np.mean(comp_err))
        per_component[f"rmse_s{comp}"] = float(np.sqrt(np.mean((comp_pred - comp_true) ** 2)))

    row = {
        "model": name,
        "checkpoint": str(checkpoint_path),
        "normalized_mse": float(total_mse / total_elements),
        "normalized_smooth_l1": float(total_smooth_l1 / total_elements),
        "mae_all_mpa": float(np.mean(abs_error)),
        "rmse_all_mpa": float(np.sqrt(np.mean((pred_raw - y_raw) ** 2))),
        "max_error_all_mpa": float(np.max(abs_error)),
        "mae_active_mpa": active_mae,
        "rmse_active_mpa": active_rmse,
        "max_error_active_mpa": active_max_error,
        "mape_active_percent": active_mape,
        "num_eval_sequences": int(y_raw.shape[0]),
        "num_active_component_slots": int(np.sum(active_mask)),
    }
    row.update(per_component)
    return row


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate E1 ablations on a common held-out split.")
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "alloy-1"))
    parser.add_argument("--mapping_path", default=str(PROJECT_ROOT / "cross-materials" / "material_mapping.json"))
    parser.add_argument("--scaler_x", default=str(PROJECT_ROOT / "cross-materials" / "scaler_x.joblib"))
    parser.add_argument("--scaler_y", default=str(PROJECT_ROOT / "cross-materials" / "scaler_y.joblib"))
    parser.add_argument("--full_checkpoint", default=str(PROJECT_ROOT / "cross-materials" / "best_model_mamba_cross-2.pth"))
    parser.add_argument("--e1_results_dir", default=str(E1_DIR / "results"))
    parser.add_argument("--output_csv", default=str(E1_DIR / "results" / "e1_test_metrics.csv"))
    parser.add_argument("--output_json", default=str(E1_DIR / "results" / "e1_test_metrics.json"))
    parser.add_argument("--seq_len", type=int, default=101)
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--output_dim", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--mat_embed_dim", type=int, default=32)
    parser.add_argument("--d_state", type=int, default=7)
    parser.add_argument("--dt_rank", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--active_threshold", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # revision-v2 controlled protocol (plan Task 4)
    parser.add_argument(
        "--split_manifest",
        type=Path,
        default=None,
        help="revision-v2 split manifest JSON (required unless --legacy_random_split)",
    )
    parser.add_argument(
        "--results_root",
        default=None,
        help="revision-v2 E1 run root with layout variant/seed_S/",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["full", "dynamic_no_gate_reg", "static_gate", "no_gate"],
        default=None,
    )
    parser.add_argument(
        "--legacy_random_split",
        action="store_true",
        help="use the historical random 80/20 split (excluded from all paper commands)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.legacy_random_split:
        run_legacy(args)
    else:
        if args.split_manifest is None:
            raise SystemExit(
                "--split_manifest is required in revision-v2 mode (or pass --legacy_random_split)"
            )
        if args.results_root is None:
            raise SystemExit("--results_root is required in revision-v2 mode")
        if not args.variants:
            raise SystemExit("--variants is required in revision-v2 mode")
        if not args.seeds:
            raise SystemExit("--seeds is required in revision-v2 mode")
        run_v2(args)


def run_v2(args):
    """revision-v2 evaluation (plan Task 4): explicit manifest test list only.

    Each (variant, seed) run directory supplies its own checkpoint, mapping,
    and scalers; the test split comes exclusively from the manifest. This
    path never calls train_test_split.
    """
    device = torch.device(args.device)
    project_root = resolve_project_root()
    manifest = load_manifest(args.split_manifest)
    results_root = Path(args.results_root)

    reference_mapping = None
    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            run_dir = results_root / variant / f"seed_{seed}"
            checkpoint = run_dir / "best_model.pth"
            if not checkpoint.exists():
                raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
            mapping = load_json_mapping(run_dir / "material_mapping.json")
            if reference_mapping is None:
                reference_mapping = mapping
            elif mapping != reference_mapping:
                raise ValueError(
                    "material mapping differs between runs; refusing to compare "
                    "incompatible runs (manifests must share the train material set)"
                )
            scaler_x = load(run_dir / "scaler_x.joblib")
            scaler_y = load(run_dir / "scaler_y.joblib")

            x_test, y_test, mat_test = load_manifest_split(
                project_root,
                manifest,
                mapping,
                "test",
                seq_len=args.seq_len,
            )

            def _fit(arr, scaler):
                return scaler.transform(arr.reshape(-1, arr.shape[-1])).reshape(arr.shape)

            loader = DataLoader(
                StrainStressEvalDataset(
                    _fit(x_test, scaler_x),
                    _fit(y_test, scaler_y),
                    x_test,
                    y_test,
                    mat_test,
                ),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )

            model = build_eval_model(variant, n_materials=len(mapping), args=args, device=device)
            model.load_state_dict(load_checkpoint_state(checkpoint, map_location=device))

            row = evaluate_split_once(
                model,
                loader,
                scaler_y,
                device,
                args.output_dim,
                active_threshold=args.active_threshold,
                collect_gates=True,
            )
            row.update(row.pop("per_component", {}))
            row.update(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "checkpoint": str(checkpoint),
                }
            )
            print(f"Evaluated {variant} seed={seed}: active MAE {row['active_component_mae_mpa']:.4f} MPa")
            rows.append(row)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    # JSON keeps the per-sequence lists (needed for the Task 5 bootstrap CI);
    # the CSV stays flat.
    dump_json(output_json, rows)
    csv_rows = [{k: v for k, v in row.items() if k != "per_sequence_active_mae_mpa"} for row in rows]
    write_csv(csv_rows, output_csv)
    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON: {output_json}")


def run_legacy(args):
    """Historical E1 evaluation on a random 20% split; behind --legacy_random_split."""
    device = torch.device(args.device)
    mapping = load_json_mapping(args.mapping_path)
    scaler_x = load(args.scaler_x)
    scaler_y = load(args.scaler_y)

    print(f"Loading raw data from {args.data_dir}")
    x, y, material_ids = load_raw_sequences(
        [args.data_dir],
        args.mapping_path,
        seq_len=args.seq_len,
    )
    _, x_eval, _, y_eval, _, mat_eval = train_test_split(
        x,
        y,
        material_ids,
        test_size=args.val_split,
        random_state=args.seed,
    )

    x_eval_norm = scaler_x.transform(x_eval.reshape(-1, args.input_dim)).reshape(x_eval.shape)
    y_eval_norm = scaler_y.transform(y_eval.reshape(-1, args.output_dim)).reshape(y_eval.shape)

    loader = DataLoader(
        EvalDataset(x_eval_norm, y_eval_norm, y_eval, mat_eval),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    e1_results_dir = Path(args.e1_results_dir)
    checkpoints = {
        "full": Path(args.full_checkpoint),
        "dynamic_no_gate_reg": e1_results_dir / "dynamic_no_gate_reg" / "best_model.pth",
        "static_gate": e1_results_dir / "static_gate" / "best_model.pth",
        "no_gate": e1_results_dir / "no_gate" / "best_model.pth",
    }

    rows = []
    for name, checkpoint_path in checkpoints.items():
        print(f"Evaluating {name}: {checkpoint_path}")
        rows.append(
            evaluate_model(
                name,
                checkpoint_path,
                loader,
                scaler_y,
                n_materials=len(mapping),
                args=args,
                device=device,
            )
        )

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    write_csv(rows, output_csv)
    dump_json(output_json, rows)

    print(f"Saved CSV: {output_csv}")
    print(f"Saved JSON: {output_json}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

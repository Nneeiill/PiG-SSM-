"""Canonical PiG-SSM transfer-learning script.

This script fine-tunes the final manuscript PiG-SSM model on a target dataset.
It reuses the training utilities from ``train_pig_ssm.py`` and keeps the same
physics-informed gate regularization term as the pretraining script.

Example (run from the project root, the parent of organized_pig_ssm/):
    python organized_pig_ssm/train_pig_ssm_transfer.py \
        --data_dirs <PROJECT_ROOT>/1-0 \
        --pretrained_path <PROJECT_ROOT>/cross-materials/best_model_mamba_cross-2.pth \
        --base_mapping_path <PROJECT_ROOT>/cross-materials/material_mapping.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from joblib import load

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pig_ssm_model import StressPredictorMamba, count_parameters
from train_pig_ssm import (
    create_or_update_material_mapping,
    load_json_mapping,
    load_raw_sequences,
    make_data_loaders,
    save_history,
    set_seed,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    train_model,
)


def load_checkpoint_state(path: str | Path, map_location="cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    return checkpoint.get("model_state_dict", checkpoint)


def copy_base_mapping(base_mapping_path: str | Path, output_mapping_path: str | Path) -> None:
    with open(base_mapping_path, "r") as f:
        mapping = json.load(f)
    output_mapping_path = Path(output_mapping_path)
    output_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)


def prepare_transfer_mapping(args, rank: int, distributed: bool):
    """Copy the source mapping and extend it with target-domain materials."""

    output_mapping = Path(args.mapping_path)
    if is_main_process(rank):
        if args.reset_mapping or not output_mapping.exists():
            copy_base_mapping(args.base_mapping_path, output_mapping)
        create_or_update_material_mapping(
            args.data_dirs,
            output_mapping,
            reset_mapping=False,
        )

    if distributed:
        import torch.distributed as dist

        dist.barrier()

    return load_json_mapping(output_mapping)


def migrate_embedding_state(new_model: StressPredictorMamba, old_state: dict) -> dict:
    """Load a checkpoint into a model whose material embedding may be larger."""

    new_state = new_model.state_dict()
    for name, param in old_state.items():
        if name == "material_embed.weight" and name in new_state:
            rows = min(new_state[name].shape[0], param.shape[0])
            new_state[name][:rows].copy_(param[:rows])
        elif name in new_state and new_state[name].shape == param.shape:
            new_state[name].copy_(param)
    return new_state


def build_transfer_model(args, n_materials: int, device: torch.device):
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
    old_state = load_checkpoint_state(args.pretrained_path, map_location="cpu")
    model.load_state_dict(migrate_embedding_state(model, old_state))
    return model.to(device)


def make_transfer_loaders(x, y, material_ids, args, distributed, world_size, rank):
    if args.fit_target_scalers:
        return make_data_loaders(x, y, material_ids, args, distributed, world_size, rank)

    scaler_x = load(args.scaler_x_path)
    scaler_y = load(args.scaler_y_path)

    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from train_pig_ssm import StrainStressDataset

    x_train, x_val, y_train, y_val, mat_train, mat_val = train_test_split(
        x,
        y,
        material_ids,
        test_size=args.val_split,
        random_state=args.seed,
    )

    x_train = scaler_x.transform(x_train.reshape(-1, args.input_dim)).reshape(x_train.shape)
    x_val = scaler_x.transform(x_val.reshape(-1, args.input_dim)).reshape(x_val.shape)
    y_train = scaler_y.transform(y_train.reshape(-1, args.output_dim)).reshape(y_train.shape)
    y_val = scaler_y.transform(y_val.reshape(-1, args.output_dim)).reshape(y_val.shape)

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
    return train_loader, val_loader, train_sampler, scaler_x, scaler_y


def run_transfer(args):
    set_seed(args.seed)
    distributed, rank, _local_rank, world_size, device = setup_distributed()

    try:
        if is_main_process(rank):
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        mapping = prepare_transfer_mapping(args, rank=rank, distributed=distributed)
        x, y, material_ids = load_raw_sequences(
            args.data_dirs,
            args.mapping_path,
            seq_len=args.seq_len,
            strain_zero_threshold=args.strain_zero_threshold,
            stress_zero_threshold=args.stress_zero_threshold,
            skip_header=args.skip_header,
        )

        train_loader, val_loader, train_sampler, scaler_x, scaler_y = make_transfer_loaders(
            x, y, material_ids, args, distributed, world_size, rank
        )

        model = build_transfer_model(args, n_materials=len(mapping), device=device)

        if distributed:
            import torch.nn as nn

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
            if args.fit_target_scalers:
                from joblib import dump

                Path(args.output_scaler_x_path).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output_scaler_y_path).parent.mkdir(parents=True, exist_ok=True)
                dump(scaler_x, args.output_scaler_x_path)
                dump(scaler_y, args.output_scaler_y_path)
            save_history(history, args.history_path)

            model_for_count = model.module if hasattr(model, "module") else model
            summary = {
                "mode": "transfer",
                "data_dirs": [str(path) for path in args.data_dirs],
                "pretrained_path": str(args.pretrained_path),
                "base_mapping_path": str(args.base_mapping_path),
                "mapping_path": str(args.mapping_path),
                "model_save_path": str(args.model_save_path),
                "num_sequences": int(len(x)),
                "num_materials": int(len(mapping)),
                "best_val_loss": float(best_val_loss),
                "num_parameters": int(count_parameters(model_for_count)),
                "trainable_parameters": int(count_parameters(model_for_count, trainable_only=True)),
                "lambda_gate": float(args.lambda_gate),
                "gate_threshold": float(args.gate_threshold),
                "fit_target_scalers": bool(args.fit_target_scalers),
            }
            with open(args.summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(json.dumps(summary, indent=2))
    finally:
        cleanup_distributed(distributed)


def parse_args():
    default_output_dir = SCRIPT_DIR / "outputs" / "transfer"
    default_pretrained = PROJECT_ROOT / "cross-materials" / "best_model_mamba_cross-2.pth"
    default_base_mapping = PROJECT_ROOT / "cross-materials" / "material_mapping.json"
    default_scaler_x = PROJECT_ROOT / "cross-materials" / "experiments_results" / "scaler_x.joblib"
    default_scaler_y = PROJECT_ROOT / "cross-materials" / "experiments_results" / "scaler_y.joblib"

    parser = argparse.ArgumentParser(description="Fine-tune PiG-SSM with gate regularization.")
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", default=str(default_output_dir))
    parser.add_argument("--pretrained_path", default=str(default_pretrained))
    parser.add_argument("--base_mapping_path", default=str(default_base_mapping))
    parser.add_argument("--mapping_path", default=str(default_output_dir / "material_mapping.json"))
    parser.add_argument("--model_save_path", default=str(default_output_dir / "finetuned_model.pth"))
    parser.add_argument("--history_path", default=str(default_output_dir / "finetuning_history.csv"))
    parser.add_argument("--summary_path", default=str(default_output_dir / "finetuning_summary.json"))
    parser.add_argument("--scaler_x_path", default=str(default_scaler_x))
    parser.add_argument("--scaler_y_path", default=str(default_scaler_y))
    parser.add_argument("--output_scaler_x_path", default=str(default_output_dir / "scaler_x.joblib"))
    parser.add_argument("--output_scaler_y_path", default=str(default_output_dir / "scaler_y.joblib"))
    parser.add_argument("--reset_mapping", action="store_true")
    parser.add_argument("--fit_target_scalers", action="store_true")
    parser.add_argument("--skip_header", action="store_true")

    parser.add_argument("--seq_len", type=int, default=101)
    parser.add_argument("--input_dim", type=int, default=6)
    parser.add_argument("--output_dim", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--mat_embed_dim", type=int, default=32)
    parser.add_argument("--d_state", type=int, default=7)
    parser.add_argument("--dt_rank", type=int, default=5)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_transfer(parse_args())

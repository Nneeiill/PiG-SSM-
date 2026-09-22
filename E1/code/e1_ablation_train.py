"""E1 ablation trainer for the PiG-SSM manuscript model.

E1 focuses on the gate mechanism because it is the central novel component in
the final PiG-SSM architecture. All variants keep the same SSM backbone,
material embedding, data split, optimizer, and preprocessing. Only the gate
mechanism or gate regularization is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import dump

SCRIPT_DIR = Path(__file__).resolve().parent
E1_DIR = SCRIPT_DIR.parent
ORGANIZED_DIR = E1_DIR.parent
PROJECT_ROOT = ORGANIZED_DIR.parent

if str(ORGANIZED_DIR) not in sys.path:
    sys.path.insert(0, str(ORGANIZED_DIR))

from pig_ssm_model import StressPredictorMamba, count_parameters
from train_pig_ssm import (
    cleanup_distributed,
    create_or_update_material_mapping,
    is_main_process,
    load_json_mapping,
    load_raw_sequences,
    make_data_loaders,
    save_history,
    set_seed,
    setup_distributed,
    train_model,
    run_manifest_core,
)


class AblationMambaSSM(nn.Module):
    """PiG-SSM mixer with alternative gate modes.

    gate_mode:
        dynamic: original input-dependent gate z_t = sigmoid(gate_net(...)).
        static: learned channel-wise constant gate, independent of loading path.
        none: no gate; equivalent to z_t = 1 for all time steps and channels.
    """

    def __init__(self, d_model: int, d_state: int = 7, dt_rank: int = 5, gate_mode: str = "dynamic"):
        super().__init__()
        if gate_mode not in {"dynamic", "static", "none"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")

        self.d_model = d_model
        self.gate_mode = gate_mode
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        self.dt_init = nn.Linear(d_model, dt_rank)
        self.proj_x = nn.Linear(d_model, d_model * 2, bias=False)
        self.A = nn.Parameter(
            torch.arange(1, d_state + 1).unsqueeze(0).expand(d_model, -1).float()
        )
        self.D = nn.Parameter(torch.ones(d_model))

        if gate_mode == "dynamic":
            self.gate_net = nn.Sequential(
                nn.Linear(d_model * 2 + 1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        elif gate_mode == "static":
            self.static_gate_logit = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        batch_size, seq_len, d_model = x.shape
        x_proj = self.proj_x(x)
        x_skip, x_ssm = x_proj.split([d_model, d_model], dim=-1)

        dt = self.dt_proj(self.dt_init(x.mean(dim=1)))
        delta = F.softplus(dt).unsqueeze(1)
        a_matrix = -torch.exp(self.A)
        a_discrete = torch.exp(delta.unsqueeze(-1) * a_matrix.unsqueeze(0))
        b_discrete = delta.expand(-1, seq_len, -1).unsqueeze(-1) * x_ssm.unsqueeze(-1)
        hidden_state = torch.cumsum(a_discrete * b_discrete, dim=1)
        ssm_out = (
            torch.einsum("bldn,dn->bld", hidden_state, a_matrix)
            * self.D.view(1, 1, -1)
        )

        if self.gate_mode == "dynamic":
            x_prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
            delta_x = x - x_prev
            delta_norm = torch.norm(delta_x, dim=-1, keepdim=True)
            gate_input = torch.cat([x, delta_x, delta_norm], dim=-1)
            z_gate = torch.sigmoid(self.gate_net(gate_input))
        elif self.gate_mode == "static":
            z_gate = torch.sigmoid(self.static_gate_logit).view(1, 1, -1)
            z_gate = z_gate.expand(batch_size, seq_len, d_model)
        else:
            z_gate = torch.ones_like(ssm_out)

        out = z_gate * (ssm_out + x_skip)
        if return_gate:
            return out, z_gate, hidden_state
        return out


class AblationMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, dt_rank: int, gate_mode: str):
        super().__init__()
        self.mixer = AblationMambaSSM(
            d_model=d_model,
            d_state=d_state,
            dt_rank=dt_rank,
            gate_mode=gate_mode,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        if return_gate:
            out, z_gate, hidden_state = self.mixer(x, return_gate=True)
            return self.norm(x + out), z_gate, hidden_state
        return self.norm(x + self.mixer(x))


class StressPredictorMambaGateAblation(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 256,
        num_layers: int = 5,
        output_dim: int = 6,
        n_materials: int = 26,
        mat_embed_dim: int = 32,
        d_state: int = 7,
        dt_rank: int = 5,
        gate_mode: str = "static",
    ):
        super().__init__()
        self.material_embed = nn.Embedding(n_materials, mat_embed_dim)
        self.embed = nn.Linear(input_dim * 2 + mat_embed_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                AblationMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    dt_rank=dt_rank,
                    gate_mode=gate_mode,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor, material_id: torch.Tensor, return_gate: bool = False):
        _, seq_len, _ = x.shape
        eps_prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        delta_eps = x - eps_prev
        x_seq = torch.cat([x, delta_eps], dim=-1)

        material_embedding = self.material_embed(material_id)
        material_embedding = material_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        hidden = self.embed(torch.cat([x_seq, material_embedding], dim=-1))

        all_z, all_h = [], []
        for block in self.blocks:
            if return_gate:
                hidden, z_gate, hidden_state = block(hidden, return_gate=True)
                all_z.append(z_gate)
                all_h.append(hidden_state)
            else:
                hidden = block(hidden)

        out = self.output_proj(hidden)
        if return_gate:
            return out, all_z, all_h
        return out


ABLATION_PRESETS = {
    "full": {
        "gate_mode": "dynamic",
        "lambda_gate": None,
        "description": "Original PiG-SSM: dynamic gate with gate regularization.",
    },
    "dynamic_no_gate_reg": {
        "gate_mode": "dynamic",
        "lambda_gate": 0.0,
        "description": "Original dynamic gate, but remove the gate regularization term.",
    },
    "static_gate": {
        "gate_mode": "static",
        "lambda_gate": 0.0,
        "description": "Replace input-dependent dynamic gate with a learned static channel gate.",
    },
    "no_gate": {
        "gate_mode": "none",
        "lambda_gate": 0.0,
        "description": "Remove gate modulation by forcing z=1.",
    },
}


def build_model(args, n_materials: int):
    preset = ABLATION_PRESETS[args.ablation]
    if args.ablation in {"full", "dynamic_no_gate_reg"}:
        return StressPredictorMamba(
            input_dim=args.input_dim,
            d_model=args.d_model,
            num_layers=args.num_layers,
            output_dim=args.output_dim,
            n_materials=n_materials,
            mat_embed_dim=args.mat_embed_dim,
            d_state=args.d_state,
            dt_rank=args.dt_rank,
        )

    return StressPredictorMambaGateAblation(
        input_dim=args.input_dim,
        d_model=args.d_model,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        n_materials=n_materials,
        mat_embed_dim=args.mat_embed_dim,
        d_state=args.d_state,
        dt_rank=args.dt_rank,
        gate_mode=preset["gate_mode"],
    )


def apply_ablation_preset(args):
    preset = ABLATION_PRESETS[args.ablation]
    if preset["lambda_gate"] is not None:
        args.lambda_gate = preset["lambda_gate"]
    return args


def run_e1_ablation(args):
    """Dispatch: revision-v2 manifest path by default, legacy behind a flag."""
    if getattr(args, "legacy_random_split", False):
        run_e1_ablation_legacy(args)
    else:
        run_e1_ablation_manifest(args)


def run_e1_ablation_manifest(args):
    """revision-v2 E1 path (plan Task 4/5): manifest splits, train-only scalers,
    one-shot test evaluation. Output layout: <output_root>/<ablation>/seed_<seed>/
    """
    args = apply_ablation_preset(args)
    if args.split_manifest is None:
        raise SystemExit(
            "--split_manifest is required in revision-v2 mode "
            "(pass --legacy_random_split to use the historical random-split path)"
        )
    if args.seed is None:
        raise SystemExit("--seed is required in revision-v2 mode")
    if args.output_root is None:
        raise SystemExit("--output_root is required in revision-v2 mode")

    args.output_root = str(Path(args.output_root) / args.ablation / f"seed_{args.seed}")
    run_manifest_core(
        args,
        # run_manifest_core calls model_builder(args, n_materials=..., device=...)
        # with keywords, so the parameter names must match the contract.
        model_builder=lambda a, n_materials, device: build_model(a, n_materials=n_materials).to(device),
        variant=args.ablation,
        extra={
            "experiment": "E1",
            "description": ABLATION_PRESETS[args.ablation]["description"],
        },
    )


def run_e1_ablation_legacy(args):
    """Historical E1 flow (random 80/20 split); kept behind --legacy_random_split."""
    args = apply_ablation_preset(args)
    set_seed(args.seed)
    distributed, rank, _local_rank, world_size, device = setup_distributed()

    run_dir = Path(args.output_root) / args.ablation
    args.model_save_path = str(run_dir / "best_model.pth")
    args.mapping_path = str(run_dir / "material_mapping.json")
    args.scaler_x_path = str(run_dir / "scaler_x.joblib")
    args.scaler_y_path = str(run_dir / "scaler_y.joblib")
    args.history_path = str(run_dir / "training_history.csv")
    args.summary_path = str(run_dir / "summary.json")

    try:
        if is_main_process(rank):
            run_dir.mkdir(parents=True, exist_ok=True)
            create_or_update_material_mapping(
                args.data_dirs,
                args.mapping_path,
                reset_mapping=args.reset_mapping,
            )

        if distributed:
            import torch.distributed as dist

            dist.barrier()

        mapping = load_json_mapping(args.mapping_path)
        x, y, material_ids = load_raw_sequences(
            args.data_dirs,
            args.mapping_path,
            seq_len=args.seq_len,
            strain_zero_threshold=args.strain_zero_threshold,
            stress_zero_threshold=args.stress_zero_threshold,
            skip_header=args.skip_header,
            max_sequences=args.max_sequences,
        )

        train_loader, val_loader, train_sampler, scaler_x, scaler_y = make_data_loaders(
            x,
            y,
            material_ids,
            args,
            distributed,
            world_size,
            rank,
        )

        model = build_model(args, n_materials=len(mapping)).to(device)
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
            dump(scaler_x, args.scaler_x_path)
            dump(scaler_y, args.scaler_y_path)
            save_history(history, args.history_path)

            model_for_count = model.module if hasattr(model, "module") else model
            summary = {
                "experiment": "E1",
                "ablation": args.ablation,
                "description": ABLATION_PRESETS[args.ablation]["description"],
                "data_dirs": [str(path) for path in args.data_dirs],
                "num_sequences": int(len(x)),
                "num_materials": int(len(mapping)),
                "best_val_loss": float(best_val_loss),
                "lambda_gate": float(args.lambda_gate),
                "gate_threshold": float(args.gate_threshold),
                "d_model": int(args.d_model),
                "num_layers": int(args.num_layers),
                "d_state": int(args.d_state),
                "dt_rank": int(args.dt_rank),
                "num_parameters": int(count_parameters(model_for_count)),
                "trainable_parameters": int(count_parameters(model_for_count, trainable_only=True)),
                "model_save_path": args.model_save_path,
                "history_path": args.history_path,
            }
            with open(args.summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(json.dumps(summary, indent=2))
    finally:
        cleanup_distributed(distributed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run E1 PiG-SSM gate ablations.")
    parser.add_argument(
        "--ablation",
        choices=sorted(ABLATION_PRESETS),
        required=True,
    )
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        default=[str(PROJECT_ROOT / "alloy-1")],
    )
    parser.add_argument(
        "--output_root",
        default=str(E1_DIR / "results"),
    )
    parser.add_argument("--reset_mapping", action="store_true")
    parser.add_argument("--skip_header", action="store_true")
    parser.add_argument("--max_sequences", type=int, default=None)

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
    parser.add_argument("--seed", type=int, default=42)

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
    run_e1_ablation(parse_args())

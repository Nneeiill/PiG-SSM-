"""Main PiG-SSM model architecture extracted from the paper notebook.

The architecture matches the repeated definitions in
cross-materials/experiments_results/experiments_figures.ipynb:
strain and strain-increment inputs are concatenated with a material embedding,
then processed by stacked physics-informed gated state-space blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PiGSSMConfig:
    input_dim: int = 6
    output_dim: int = 6
    d_model: int = 256
    num_layers: int = 5
    n_materials: int = 26
    mat_embed_dim: int = 32
    d_state: int = 7
    dt_rank: int = 5


class MambaSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 7, dt_rank: int = 5):
        super().__init__()
        self.d_model = d_model
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        self.dt_init = nn.Linear(d_model, dt_rank)
        self.proj_x = nn.Linear(d_model, d_model * 2, bias=False)
        self.A = nn.Parameter(
            torch.arange(1, d_state + 1).unsqueeze(0).expand(d_model, -1).float()
        )
        self.D = nn.Parameter(torch.ones(d_model))
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

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

        x_prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        delta_x = x - x_prev
        delta_norm = torch.norm(delta_x, dim=-1, keepdim=True)
        gate_input = torch.cat([x, delta_x, delta_norm], dim=-1)
        z_gate = torch.sigmoid(self.gate_net(gate_input))
        out = z_gate * (ssm_out + x_skip)

        if return_gate:
            return out, z_gate, hidden_state
        return out


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, dt_rank: int):
        super().__init__()
        self.mixer = MambaSSM(d_model, d_state=d_state, dt_rank=dt_rank)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, return_gate: bool = False):
        if return_gate:
            out, z_gate, hidden_state = self.mixer(x, return_gate=True)
            return self.norm(x + out), z_gate, hidden_state
        return self.norm(x + self.mixer(x))


class StressPredictorMamba(nn.Module):
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
    ):
        super().__init__()
        self.material_embed = nn.Embedding(n_materials, mat_embed_dim)
        self.embed = nn.Linear(input_dim * 2 + mat_embed_dim, d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state=d_state, dt_rank=dt_rank) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(d_model, output_dim)

    @classmethod
    def from_config(cls, config: PiGSSMConfig) -> "StressPredictorMamba":
        return cls(**config.__dict__)

    def forward(self, x: torch.Tensor, material_id: torch.Tensor, return_gate: bool = False):
        batch_size, seq_len, _ = x.shape
        eps_prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        delta_eps = x - eps_prev
        x_seq = torch.cat([x, delta_eps], dim=-1)

        material_embedding = self.material_embed(material_id)
        material_embedding = material_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        x_aug = torch.cat([x_seq, material_embedding], dim=-1)
        hidden = self.embed(x_aug)

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


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


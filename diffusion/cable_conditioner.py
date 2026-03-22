import torch
import torch.nn as nn


class CableConditioner(nn.Module):
    """Conditioning network for DDPM: embeds cable physical parameters into a
    fixed-size conditioning vector.
    """

    CABLE_MATERIALS = {"fiber": 0, "coaxial": 1, "hybrid": 2, "armored": 3}

    def __init__(self, condition_dim: int = 32):
        super().__init__()
        self.condition_dim = condition_dim

        self.material_embed = nn.Embedding(4, 8)
        self.depth_proj = nn.Linear(1, 8)
        self.age_proj = nn.Linear(1, 8)
        self.severity_proj = nn.Linear(1, 4)
        self.position_proj = nn.Linear(1, 4)
        # Total concatenated dim = 8+8+8+4+4 = 32
        self.fusion = nn.Linear(32, condition_dim)

    def forward(
        self,
        cable_material: torch.LongTensor,
        depth_m: torch.Tensor,
        age_years: torch.Tensor,
        cut_severity: torch.Tensor,
        position_km: torch.Tensor,
    ) -> torch.Tensor:
        """Encode cable parameters into a (batch, condition_dim) vector."""
        depth_norm = depth_m / 8000.0
        age_norm = age_years / 50.0
        pos_norm = position_km / 10000.0

        e_mat = self.material_embed(cable_material)                       # (B, 8)
        e_dep = self.depth_proj(depth_norm.unsqueeze(-1))                 # (B, 8)
        e_age = self.age_proj(age_norm.unsqueeze(-1))                     # (B, 8)
        e_sev = self.severity_proj(cut_severity.unsqueeze(-1))            # (B, 4)
        e_pos = self.position_proj(pos_norm.unsqueeze(-1))                # (B, 4)

        cond = torch.cat([e_mat, e_dep, e_age, e_sev, e_pos], dim=-1)    # (B, 32)
        return self.fusion(cond)                                          # (B, condition_dim)

from __future__ import annotations

import torch
import torch.nn as nn


class DirectorModel(nn.Module):
    UNTRAINED_STATUS = "UNTRAINED"

    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.is_trained = False

        self.input_proj = nn.Linear(8, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.composition_head = nn.Linear(d_model, 1)
        self.quality_head = nn.Linear(d_model, 1)
        self.hierarchy_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_proj(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return {
            "composition": torch.sigmoid(self.composition_head(pooled)),
            "quality": torch.sigmoid(self.quality_head(pooled)),
            "hierarchy": torch.sigmoid(self.hierarchy_head(pooled)),
        }

    def evaluate(self, assembled: dict[str, Any], task: dict) -> dict[str, Any]:
        batch = torch.zeros(1, 4, 8)
        with torch.no_grad():
            scores = self.forward(batch)
        composition = float(scores["composition"].item())
        quality = float(scores["quality"].item())
        hierarchy = float(scores["hierarchy"].item())
        overall = (composition + quality + hierarchy) / 3.0

        issues = []
        if composition < 0.5:
            issues.append("weak_composition")
        if quality < 0.5:
            issues.append("low_quality")
        if hierarchy < 0.5:
            issues.append("poor_visual_hierarchy")

        return {
            "status": "UNTRAINED" if not self.is_trained else "TRAINED",
            "composition": composition,
            "quality": quality,
            "hierarchy": hierarchy,
            "overall": overall,
            "issues": issues,
        }

    def load_weights(self, path: str) -> None:
        self.is_trained = True

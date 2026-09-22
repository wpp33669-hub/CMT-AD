"""Extract aligned log and KPI representations for cross-modal transfer."""

from __future__ import annotations

import torch
from torch import nn


class MultimodalFeatureEncoder(nn.Module):
    def __init__(
        self, log_model, kpi_model, log_dimension, kpi_dimension, modality_dimension
    ):
        super().__init__()
        self.log_model, self.kpi_model = log_model, kpi_model
        # Symmetric modality projections produce the paper's d_m-dimensional
        # representations without changing either pretrained base encoder.
        self.log_projection = nn.Linear(log_dimension, modality_dimension)
        self.kpi_projection = nn.Linear(kpi_dimension, modality_dimension)
        for model in (log_model, kpi_model):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.log_model.eval()
        self.kpi_model.eval()
        return self

    def forward(self, logs, lengths, sequences, counts, kpis):
        _, log_global = self.log_model.encode(logs, lengths, sequences, counts)
        log_features = self.log_projection(log_global)
        kpi_nodes = torch.stack([self.kpi_model.encode_window(window) for window in kpis])
        kpi_features = self.kpi_projection(kpi_nodes.mean(dim=1))
        return log_features, kpi_features, kpi_nodes

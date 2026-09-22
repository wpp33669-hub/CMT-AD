"""Confidence-gated CMKT, PKT and optional contrastive regularizers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def entropy_confidence(assignments, epsilon=1e-8):
    assignments = assignments.clamp(min=epsilon)
    entropy = -(assignments * assignments.log()).sum(dim=1)
    entropy /= torch.log(assignments.new_tensor(assignments.shape[1])) + epsilon
    return 1.0 - entropy


def dec_soft_assign(features, centers, alpha=1.0, epsilon=1e-12):
    distance = ((features.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=2)
    assignments = (1.0 + distance / alpha).pow(-(alpha + 1.0) / 2.0)
    return assignments / assignments.sum(dim=1, keepdim=True).clamp(min=epsilon)


def pkt_loss(teacher, student, temperature=0.2, epsilon=1e-8):
    teacher, student = F.normalize(teacher, dim=-1), F.normalize(student, dim=-1)
    left = F.softmax(teacher @ teacher.T / temperature, dim=1).clamp(min=epsilon)
    right = F.softmax(student @ student.T / temperature, dim=1).clamp(min=epsilon)
    return (left * (left.log() - right.log())).sum(dim=1).mean()


class CMKTRegularizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj_log = nn.Linear(config.modality_dimension, config.common_dimension)
        self.proj_kpi = nn.Linear(config.modality_dimension, config.common_dimension)
        self.fuse = nn.Sequential(
            nn.Linear(config.common_dimension, config.common_dimension), nn.ReLU(),
            nn.Linear(config.common_dimension, config.common_dimension),
        )
        self.prototypes = nn.Parameter(torch.randn(config.semantic_prototypes, config.common_dimension))

    def _prototype_assignment(self, features):
        return F.softmax(
            F.normalize(features, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
            / self.config.semantic_temperature,
            dim=1,
        )

    @staticmethod
    def _kl(left, right, epsilon=1e-8):
        left, right = left.clamp(min=epsilon), right.clamp(min=epsilon)
        return (left * (left.log() - right.log())).sum(dim=1).mean()

    def forward_no_centers(self, log_features, kpi_features):
        projected_log, projected_kpi = self.proj_log(log_features), self.proj_kpi(kpi_features)
        batch = log_features.shape[0]
        weights = torch.full((batch,), 0.5, device=log_features.device, dtype=log_features.dtype)
        if self.config.gate_without_centers == "equal":
            confidence = weights.clone()
        else:
            confidence = torch.exp(-torch.norm(projected_log - projected_kpi, dim=1))
        middle = self.fuse(weights.unsqueeze(1) * (projected_log + projected_kpi))
        return middle, weights, weights.clone(), confidence, confidence.clone()

    def center_free_alignment_loss(self, log_features, kpi_features):
        projected_log, projected_kpi = self.proj_log(log_features), self.proj_kpi(kpi_features)
        middle, *_ = self.forward_no_centers(log_features, kpi_features)
        loss = F.mse_loss(projected_log, projected_kpi)
        loss += self.config.structural_weight * pkt_loss(
            projected_log, projected_kpi, self.config.pkt_temperature
        )
        loss += 0.5 * (
            F.mse_loss(middle, projected_log) + F.mse_loss(middle, projected_kpi)
        )
        return loss

    def forward(self, log_features, kpi_features, centers):
        projected_log, projected_kpi = self.proj_log(log_features), self.proj_kpi(kpi_features)
        log_confidence = entropy_confidence(
            dec_soft_assign(projected_log, centers, self.config.dec_alpha)
        )
        kpi_confidence = entropy_confidence(
            dec_soft_assign(projected_kpi, centers, self.config.dec_alpha)
        )
        denominator = log_confidence + kpi_confidence + 1e-8
        log_weight, kpi_weight = log_confidence / denominator, kpi_confidence / denominator
        if self.config.detach_gate:
            log_weight, kpi_weight = log_weight.detach(), kpi_weight.detach()
        middle = self.fuse(
            log_weight.unsqueeze(1) * projected_log + kpi_weight.unsqueeze(1) * projected_kpi
        )
        loss = (1 - log_weight).mean() * (
            F.mse_loss(projected_log, middle)
            + self.config.structural_weight
            * pkt_loss(middle, projected_log, self.config.pkt_temperature)
        )
        loss += (1 - kpi_weight).mean() * (
            F.mse_loss(projected_kpi, middle)
            + self.config.structural_weight
            * pkt_loss(middle, projected_kpi, self.config.pkt_temperature)
        )
        if self.config.semantic_weight > 0:
            middle_assignment = self._prototype_assignment(middle)
            semantic = (1 - log_weight).mean() * self._kl(
                middle_assignment, self._prototype_assignment(projected_log)
            )
            semantic += (1 - kpi_weight).mean() * self._kl(
                middle_assignment, self._prototype_assignment(projected_kpi)
            )
            loss += self.config.semantic_weight * semantic
        return loss, middle, log_weight, kpi_weight, log_confidence, kpi_confidence


def intra_modal_contrastive_loss(features, labels, temperature=0.07):
    values = F.normalize(features.mean(dim=1), dim=1)
    similarity = values @ values.T / temperature
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    log_probability = similarity - torch.logsumexp(
        similarity.masked_fill(diagonal, float("-inf")), dim=1, keepdim=True
    )
    return -((log_probability * positives).sum(1) / positives.sum(1).clamp(min=1)).mean()


def cross_modal_contrastive_loss(log_features, kpi_features, temperature=0.07):
    log_features = F.normalize(log_features, dim=1)
    kpi_features = F.normalize(kpi_features, dim=1)
    logits = log_features @ kpi_features.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)
    )

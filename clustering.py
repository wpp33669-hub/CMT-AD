"""K-means initialization and DEC optimization over the CMKT middle modality."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch import nn
from tqdm import tqdm

from regularization import (
    cross_modal_contrastive_loss,
    dec_soft_assign,
    intra_modal_contrastive_loss,
)
from utils import save_checkpoint


@torch.no_grad()
def initialize_cluster_centers(log_loader, kpi_loader, encoder, regularizer, clusters, device):
    encoder.eval()
    regularizer.eval()
    features = []
    for (logs, lengths, sequences, counts, _), kpis in tqdm(
        zip(log_loader, kpi_loader), desc="K-means middle modality"
    ):
        logs, lengths, counts, kpis = (
            logs.to(device), lengths.to(device), counts.to(device), kpis.to(device)
        )
        log_features, kpi_features, _ = encoder(
            logs, lengths, sequences, counts, kpis
        )
        middle, *_ = regularizer.forward_no_centers(log_features, kpi_features)
        features.append(middle.cpu())
    if not features:
        raise ValueError("Empty dataloader")
    kmeans = KMeans(n_clusters=clusters, n_init=10, init="k-means++", random_state=42)
    kmeans.fit(torch.cat(features).numpy())
    return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)


class DECModel(nn.Module):
    def __init__(self, encoder, regularizer, centers, alpha=1.0):
        super().__init__()
        self.encoder, self.regularizer, self.alpha = encoder, regularizer, alpha
        self.cluster_centers = nn.Parameter(centers.detach().clone())

    def forward(self, logs, lengths, sequences, counts, kpis):
        log_features, kpi_features, kpi_nodes = self.encoder(
            logs, lengths, sequences, counts, kpis
        )
        cmkt = self.regularizer(log_features, kpi_features, self.cluster_centers)
        cmkt_loss, middle, log_weight, kpi_weight, log_confidence, kpi_confidence = cmkt
        assignments = dec_soft_assign(middle, self.cluster_centers, self.alpha)
        weight = assignments.pow(2) / assignments.sum(0).clamp(min=1e-12)
        target = weight / weight.sum(1, keepdim=True).clamp(min=1e-12)
        return assignments, target.detach(), kpi_nodes, log_features, kpi_features, cmkt_loss, middle, log_weight, kpi_weight, log_confidence, kpi_confidence


def pretrain_cross_modal(log_loader, kpi_loader, encoder, regularizer, config, device):
    parameters = [
        parameter
        for module in (encoder.log_projection, encoder.kpi_projection, regularizer)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(parameters, lr=config.alignment_learning_rate)
    for epoch in range(config.alignment_epochs):
        encoder.train()
        regularizer.train()
        total, batches = 0.0, 0
        for (logs, lengths, sequences, counts, _), kpis in tqdm(
            zip(log_loader, kpi_loader), desc="Cross-modal alignment"
        ):
            logs, lengths, counts, kpis = (
                logs.to(device), lengths.to(device), counts.to(device), kpis.to(device)
            )
            log_features, kpi_features, _ = encoder(
                logs, lengths, sequences, counts, kpis
            )
            loss = regularizer.center_free_alignment_loss(log_features, kpi_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, batches = total + loss.item(), batches + 1
        print("Alignment %d/%d: loss=%.6f" % (
            epoch + 1, config.alignment_epochs, total / max(batches, 1)
        ))


def derive_binary_mapping(labels):
    clusters, counts = np.unique(labels, return_counts=True)
    if len(clusters) <= 2:
        normal = {clusters[np.argmax(counts)]}
        return {int(cluster): (0 if cluster in normal else 1) for cluster in clusters}
    ordered = clusters[np.argsort(counts)]
    normal = {ordered[0], ordered[-1]}
    return {int(cluster): (0 if cluster in normal else 1) for cluster in clusters}


def cluster_labels_to_binary(labels, mapping=None):
    mapping = mapping or derive_binary_mapping(labels)
    return np.asarray([mapping.get(int(label), 1) for label in labels], dtype=np.int64)


def new_cluster_labels_to_binary(
    assignments, middle_features, cluster_centers, state_mapping=None, epsilon=1e-12
):
    """Map three DEC clusters to normal/anomalous labels via state semantics.

    The mapping follows the proposed normal/transitional/abnormal strategy:

    * R_j: proportion of samples assigned to cluster j.
    * D_j: mean Euclidean distance to cluster center j.
    * H_j: mean normalized entropy of the DEC soft assignments.
    * normal score: R~_j + (1 - D~_j) + (1 - H_j).
    * transitional score: D~_j + H_j.

    The cluster with the largest normal score is the normal state.  Of the two
    remaining clusters, the one with the largest transitional score is the
    transitional state and the last one is the abnormal state.  A sample in
    the transitional cluster is anomalous only when it is closer to the
    abnormal center than to the normal center.

    ``state_mapping`` must be reused for the final test set so test data do not
    participate in deciding the semantic role of each cluster.
    """

    def as_numpy(values):
        if torch.is_tensor(values):
            return values.detach().cpu().numpy()
        return np.asarray(values)

    probabilities = as_numpy(assignments).astype(np.float64, copy=False)
    features = as_numpy(middle_features).astype(np.float64, copy=False)
    centers = as_numpy(cluster_centers).astype(np.float64, copy=False)
    if probabilities.ndim != 2 or features.ndim != 2 or centers.ndim != 2:
        raise ValueError("assignments, middle_features and cluster_centers must be 2-D")
    if len(probabilities) != len(features):
        raise ValueError("assignments and middle_features must contain the same samples")
    cluster_count = probabilities.shape[1]
    if cluster_count != 3 or centers.shape[0] != cluster_count:
        raise ValueError("The new cluster mapping strategy requires exactly three clusters")

    hard_clusters = probabilities.argmax(axis=1)
    if state_mapping is None:
        counts = np.bincount(hard_clusters, minlength=cluster_count).astype(np.float64)
        if np.any(counts == 0):
            raise ValueError("Cannot derive state mapping because at least one cluster is empty")

        proportions = counts / len(hard_clusters)
        dispersions = np.empty(cluster_count, dtype=np.float64)
        entropies = np.empty(cluster_count, dtype=np.float64)
        sample_entropy = -(
            np.clip(probabilities, epsilon, 1.0)
            * np.log(np.clip(probabilities, epsilon, 1.0))
        ).sum(axis=1) / np.log(cluster_count)
        for cluster in range(cluster_count):
            members = hard_clusters == cluster
            dispersions[cluster] = np.linalg.norm(
                features[members] - centers[cluster], axis=1
            ).mean()
            entropies[cluster] = sample_entropy[members].mean()

        def minmax(values):
            span = values.max() - values.min()
            return np.zeros_like(values) if span <= epsilon else (values - values.min()) / span

        normalized_r = minmax(proportions)
        normalized_d = minmax(dispersions)
        # H_j is already normalized to [0, 1] by division by log(C), so the
        # paper only applies cross-cluster min-max normalization to R_j and D_j.
        normal_scores = normalized_r + (1.0 - normalized_d) + (1.0 - entropies)
        transitional_scores = normalized_d + entropies

        normal_cluster = int(np.argmax(normal_scores))
        remaining = [cluster for cluster in range(cluster_count) if cluster != normal_cluster]
        transitional_cluster = int(max(remaining, key=lambda cluster: transitional_scores[cluster]))
        abnormal_cluster = int(next(
            cluster for cluster in remaining if cluster != transitional_cluster
        ))
        state_mapping = {
            "normal": normal_cluster,
            "transitional": transitional_cluster,
            "abnormal": abnormal_cluster,
        }
    else:
        required = {"normal", "transitional", "abnormal"}
        if set(state_mapping) != required:
            raise ValueError("state_mapping must contain normal, transitional and abnormal")

    normal_cluster = int(state_mapping["normal"])
    transitional_cluster = int(state_mapping["transitional"])
    abnormal_cluster = int(state_mapping["abnormal"])
    binary = np.ones(len(hard_clusters), dtype=np.int64)
    binary[hard_clusters == normal_cluster] = 0

    transitional = hard_clusters == transitional_cluster
    if np.any(transitional):
        transitional_features = features[transitional]
        distance_to_normal = np.linalg.norm(
            transitional_features - centers[normal_cluster], axis=1
        )
        distance_to_abnormal = np.linalg.norm(
            transitional_features - centers[abnormal_cluster], axis=1
        )
        binary[transitional] = (distance_to_normal > distance_to_abnormal).astype(np.int64)
    return binary, state_mapping


def train_dec(log_loader, kpi_loader, model, config, device):
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    criterion = nn.KLDivLoss(reduction="batchmean")
    state_mapping = getattr(model, "cluster_state_mapping", None)
    for epoch in range(config.epochs):
        model.train()
        total, batches, predictions, truths = 0.0, 0, [], []
        soft_assignments, middle_features = [], []
        for (logs, lengths, sequences, counts, labels), kpis in tqdm(zip(log_loader, kpi_loader), desc="DEC train"):
            logs, lengths, counts, kpis = (
                logs.to(device), lengths.to(device), counts.to(device), kpis.to(device)
            )
            assignments, target, kpi_nodes, log_features, kpi_features, cmkt_loss, middle, *_ = model(
                logs, lengths, sequences, counts, kpis
            )
            pseudo_labels = target.argmax(dim=1)
            contrastive = intra_modal_contrastive_loss(
                log_features.unsqueeze(1), pseudo_labels
            )
            contrastive += intra_modal_contrastive_loss(kpi_nodes, pseudo_labels)
            contrastive += cross_modal_contrastive_loss(log_features, kpi_features)
            loss = criterion(assignments.clamp(min=1e-12).log(), target)
            loss += config.regularization_weight * cmkt_loss
            loss += config.contrastive_weight * contrastive
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, batches = total + loss.item(), batches + 1
            predictions.extend(assignments.argmax(1).detach().cpu().tolist())
            soft_assignments.append(assignments.detach().cpu())
            middle_features.append(middle.detach().cpu())
            truths.extend(labels.tolist())

        # Legacy cluster-size mapping retained for comparison; it is no longer used.
        # mapping = derive_binary_mapping(np.asarray(predictions))
        # model.binary_mapping = mapping
        # binary = cluster_labels_to_binary(np.asarray(predictions), mapping)

        epoch_assignments = torch.cat(soft_assignments)
        epoch_middle = torch.cat(middle_features)
        cluster_counts = np.bincount(
            epoch_assignments.argmax(dim=1).numpy(), minlength=config.clusters
        )
        mapping_reused = bool(np.any(cluster_counts == 0))
        if mapping_reused:
            if state_mapping is None:
                raise RuntimeError(
                    "DEC produced an empty cluster before a valid three-state mapping "
                    "was available; adjust the clustering optimization and retry"
                )
            # R_j, D_j and H_j are undefined for an empty cluster.  Keep the
            # most recent valid role assignment until all three clusters are
            # populated again instead of aborting a transient DEC epoch.
            binary, _ = new_cluster_labels_to_binary(
                epoch_assignments,
                epoch_middle,
                model.cluster_centers,
                state_mapping=state_mapping,
            )
        else:
            binary, state_mapping = new_cluster_labels_to_binary(
                epoch_assignments,
                epoch_middle,
                model.cluster_centers,
            )
        model.cluster_state_mapping = state_mapping
        truths = np.asarray(truths)
        accuracy = accuracy_score(truths, binary)
        precision = precision_score(truths, binary, pos_label=0, zero_division=0)
        recall = recall_score(truths, binary, pos_label=0, zero_division=0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        print("DEC %d/%d: loss=%.6f accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f" % (
            epoch + 1, config.epochs, total / max(batches, 1), accuracy, precision, recall, f1
        ))
        print("DEC cluster counts: %s states=%s%s" % (
            cluster_counts.tolist(), state_mapping,
            " (reused: empty cluster)" if mapping_reused else "",
        ))
        save_checkpoint(model, optimizer, epoch, config.checkpoint_dir)
    return model


@torch.no_grad()
def evaluate_dec(log_loader, kpi_loader, model, device):
    model.eval()
    predictions, truths = [], []
    soft_assignments, middle_features = [], []
    for (logs, lengths, sequences, counts, labels), kpis in tqdm(
        zip(log_loader, kpi_loader), desc="Final test"
    ):
        logs, lengths, counts, kpis = (
            logs.to(device), lengths.to(device), counts.to(device), kpis.to(device)
        )
        outputs = model(logs, lengths, sequences, counts, kpis)
        assignments, middle = outputs[0], outputs[6]
        predictions.extend(assignments.argmax(1).cpu().tolist())
        soft_assignments.append(assignments.cpu())
        middle_features.append(middle.cpu())
        truths.extend(labels.tolist())

    # Legacy cluster-size mapping retained for comparison; it is no longer used.
    # mapping = getattr(model, "binary_mapping", None)  # old s
    # if mapping is None:
    #     raise RuntimeError(
    #         "DEC model has no cluster-to-label mapping; train it before evaluation"
    #     )
    # binary = cluster_labels_to_binary(np.asarray(predictions), mapping)  old e

    state_mapping = getattr(model, "cluster_state_mapping", None) #new s
    if state_mapping is None:
        raise RuntimeError("DEC model has no cluster-state mapping; train it before evaluation")
    binary, _ = new_cluster_labels_to_binary(
        torch.cat(soft_assignments),
        torch.cat(middle_features),
        model.cluster_centers,
        state_mapping=state_mapping,
    )   #new e
    truths = np.asarray(truths)
    accuracy = accuracy_score(truths, binary)
    precision = precision_score(truths, binary, pos_label=0, zero_division=0)
    recall = recall_score(truths, binary, pos_label=0, zero_division=0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    print("Final test:", {name: round(value, 4) for name, value in metrics.items()})
    return metrics

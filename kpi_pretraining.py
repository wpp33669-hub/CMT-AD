"""FFT-enhanced LSTM-GCN-VGAE KPI pretraining."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from utils import save_checkpoint


def build_kpi_adjacency(window, top_k):
    features = window.detach().cpu().numpy().T
    # Constant KPI series have zero standard deviation.  Their correlations
    # are undefined and are intentionally converted to zero below.
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(features)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    nodes = correlation.shape[0]
    neighbors = min(top_k, nodes - 1)
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    for node in range(nodes):
        selected = np.argpartition(correlation[node], -(neighbors + 1))[-(neighbors + 1):]
        adjacency[node, selected[selected != node]] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T) + np.eye(nodes, dtype=np.float32)
    return adjacency / adjacency.sum(axis=1, keepdims=True).clip(min=1.0)


def _sparse_tensor(matrix):
    matrix = sp.coo_matrix(matrix).astype(np.float32)
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64))
    return torch.sparse_coo_tensor(indices, torch.from_numpy(matrix.data), matrix.shape).coalesce()


def normalize_adjacency(adjacency):
    sparse = sp.coo_matrix(adjacency)
    with_loops = sparse + sp.eye(sparse.shape[0])
    rows = np.asarray(with_loops.sum(1)).ravel().clip(min=1e-12)
    degree = sp.diags(np.power(rows, -0.5))
    normalized = degree @ with_loops @ degree
    labels = torch.tensor((sparse + sp.eye(sparse.shape[0])).toarray(), dtype=torch.float32)
    edges, total = float(sparse.sum()), sparse.shape[0] ** 2
    return _sparse_tensor(normalized), labels, (total - edges) / max(edges, 1.0), total / max(2 * (total - edges), 1.0)


class NodeLSTMEncoder(nn.Module):
    def __init__(self, hidden_dimension=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dimension, batch_first=True)

    def forward(self, values):
        _, (hidden, _) = self.lstm(values.unsqueeze(-1).float())
        return hidden[-1]


class FFTEncoder(nn.Module):
    def forward(self, values):
        magnitude = torch.abs(torch.fft.rfft(values, dim=1))
        return magnitude / (torch.norm(magnitude, dim=1, keepdim=True) + 1e-8)


class GraphConvolution(nn.Module):
    def __init__(self, input_dimension, output_dimension, dropout=0.0, activation=F.relu):
        super().__init__()
        self.dropout, self.activation = dropout, activation
        self.weight = nn.Parameter(torch.empty(input_dimension, output_dimension))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adjacency):
        support = F.dropout(features, self.dropout, self.training) @ self.weight
        return self.activation(torch.sparse.mm(adjacency, support))


class StrongKPIEncoder(nn.Module):
    def __init__(self, sequence_length, lstm_hidden, graph_hidden, latent_dimension, dropout=0.0):
        super().__init__()
        self.node_lstm = NodeLSTMEncoder(lstm_hidden)
        self.fft_encoder = FFTEncoder()
        self.freq_proj = nn.Linear(sequence_length // 2 + 1, lstm_hidden)
        self.gc1 = GraphConvolution(2 * lstm_hidden, graph_hidden, dropout, F.relu)
        self.gc2 = GraphConvolution(graph_hidden, latent_dimension, dropout, lambda value: value)
        self.gc3 = GraphConvolution(graph_hidden, latent_dimension, dropout, lambda value: value)

    def encode_hidden(self, values, adjacency):
        time_features = self.node_lstm(values)
        frequency_features = self.freq_proj(self.fft_encoder(values))
        return self.gc1(torch.cat([time_features, frequency_features], dim=1), adjacency)

    def forward(self, values, adjacency):
        hidden = self.encode_hidden(values, adjacency)
        mean, log_variance = self.gc2(hidden, adjacency), self.gc3(hidden, adjacency)
        latent = mean + torch.randn_like(log_variance) * torch.exp(log_variance) if self.training else mean
        return latent @ latent.T, mean, log_variance


def vgae_loss(prediction, labels, mean, log_variance, nodes, norm, weight):
    reconstruction = norm * F.binary_cross_entropy_with_logits(
        prediction, labels, pos_weight=prediction.new_tensor(weight)
    )
    divergence = -0.5 / nodes * torch.mean(
        torch.sum(1 + 2 * log_variance - mean.pow(2) - log_variance.exp().pow(2), dim=1)
    )
    return reconstruction + divergence


class KPIAutoencoder(nn.Module):
    def __init__(self, data_config, config):
        super().__init__()
        self.data_config, self.config = data_config, config
        graph_hidden = data_config.window_length // 2
        self.GAE_kpi = StrongKPIEncoder(
            data_config.window_length, config.lstm_hidden, graph_hidden, graph_hidden // 2
        )

    def _graph(self, window):
        return normalize_adjacency(build_kpi_adjacency(window, self.config.top_k))

    def encode_window(self, window):
        adjacency, _, _, _ = self._graph(window)
        return self.GAE_kpi.encode_hidden(window.T.contiguous(), adjacency.to(window.device))

    def forward(self, batch):
        losses = []
        for window in batch:
            adjacency, labels, weight, norm = self._graph(window)
            prediction, mean, log_variance = self.GAE_kpi(
                window.T.contiguous(), adjacency.to(window.device)
            )
            losses.append(vgae_loss(
                prediction, labels.to(window.device), mean, log_variance,
                window.shape[1], norm, weight,
            ))
        return torch.stack(losses).mean()


def _epoch(loader, model, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    for batch in tqdm(loader, desc="KPI train" if training else "KPI eval"):
        batch = batch[:, :, : model.data_config.kpi_feature_count].to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            loss = model(batch)
            if training:
                loss.backward()
                optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def pretrain_kpi(train_loader, test_loader, data_config, config, device):
    model = KPIAutoencoder(data_config, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.epochs):
        train_loss = _epoch(train_loader, model, device, optimizer)
        test_loss = _epoch(test_loader, model, device)
        print("KPI %d/%d: train=%.4f test=%.4f" % (epoch + 1, config.epochs, train_loss, test_loss))
        save_checkpoint(model, optimizer, epoch, config.checkpoint_dir)
    return model

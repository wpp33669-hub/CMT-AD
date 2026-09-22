"""CNN-Transformer log pretraining.

The former event-graph GCN branch is intentionally disabled.  ``sequences``
and ``counts`` are still accepted by the public methods so the data loaders and
the downstream cross-modal pipeline do not need a separate interface.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from utils import save_checkpoint


def _flatten(sequence):
    result = []
    for value in sequence:
        result.extend(_flatten(value) if isinstance(value, list) else [value])
    return result


def build_event_adjacency(sequence_batch, nodes, device):
    adjacency = torch.zeros(len(sequence_batch), nodes, nodes, device=device)
    for batch_index, nested in enumerate(sequence_batch):
        sequence = [int(value) for value in _flatten(nested) if value is not None]
        for left, right in zip(sequence, sequence[1:]):
            if 0 <= left < nodes and 0 <= right < nodes:
                adjacency[batch_index, left, right] += 1
        valid = {value for value in sequence if 0 <= value < nodes}
        for left in valid:
            for right in valid:
                if left != right:
                    adjacency[batch_index, left, right] += 1
    return adjacency / (adjacency.sum(dim=2, keepdim=True) + 1e-8)


class PositionalEncoding(nn.Module):
    def __init__(self, dimension, maximum, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        encoding = torch.zeros(maximum, dimension)
        positions = torch.arange(maximum).float().unsqueeze(1)
        frequency = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000.0) / dimension))
        encoding[:, 0::2] = torch.sin(positions * frequency)
        encoding[:, 1::2] = torch.cos(positions * frequency)
        self.register_buffer("pe", encoding)

    def forward(self, values):
        return self.dropout(values + self.pe[: values.shape[1]].unsqueeze(0))


def padding_mask(lengths, maximum):
    return torch.arange(maximum, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)


class LogGraphGCN(nn.Module):
    def __init__(self, input_dimension, hidden_dimension):
        super().__init__()
        self.fc1 = nn.Linear(input_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, hidden_dimension)

    def forward(self, features, adjacency):
        hidden = F.relu(self.fc1(torch.bmm(adjacency, features)))
        return self.fc2(torch.bmm(adjacency, hidden))


class CNNTransformerAE(nn.Module):
    def __init__(self, input_dimension, config):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(input_dimension, config.d_model)
        padding = config.cnn_kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(config.d_model, config.cnn_channels, config.cnn_kernel_size, padding=padding),
            nn.ReLU(), nn.Dropout(config.dropout),
            nn.Conv1d(config.cnn_channels, config.d_model, config.cnn_kernel_size, padding=padding), nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(config.d_model, config.max_sequence_length, config.dropout)
        encoder = nn.TransformerEncoderLayer(
            config.d_model, config.nhead, config.feedforward_dimension, config.dropout
        )
        decoder = nn.TransformerEncoderLayer(
            config.d_model, config.nhead, config.feedforward_dimension, config.dropout
        )
        self.transformer_enc = nn.TransformerEncoder(encoder, config.encoder_layers)
        self.transformer_dec = nn.TransformerEncoder(decoder, config.decoder_layers)
        self.out_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dimension), nn.ReLU(),
            nn.Dropout(config.dropout), nn.Linear(config.feedforward_dimension, input_dimension),
        )

    def encode(self, inputs, lengths=None):
        hidden = self.in_proj(inputs)
        hidden = self.pos_enc(hidden + self.cnn(hidden.transpose(1, 2)).transpose(1, 2))
        mask = padding_mask(lengths, inputs.shape[1]) if lengths is not None else None
        sequence = self.transformer_enc(hidden.transpose(0, 1), src_key_padding_mask=mask).transpose(0, 1)
        if mask is None:
            pooled = sequence.mean(dim=1)
        else:
            valid = ~mask
            pooled = (sequence * valid.unsqueeze(2)).sum(1) / valid.sum(1).clamp(min=1).unsqueeze(1)
        return sequence, pooled

    def forward(self, inputs, lengths=None):
        sequence, _ = self.encode(inputs, lengths)
        return self.decode(sequence, lengths)

    def decode(self, sequence, lengths=None):
        mask = padding_mask(lengths, sequence.shape[1]) if lengths is not None else None
        decoded = self.transformer_dec(sequence.transpose(0, 1), src_key_padding_mask=mask).transpose(0, 1)
        return self.out_mlp(decoded)


class StrongLogAutoencoder(nn.Module):
    """Reconstruct log semantic sequences with a CNN-Transformer encoder."""

    def __init__(self, input_dimension, config):
        super().__init__()
        self.config = config
        self.log_model = CNNTransformerAE(input_dimension, config)

        # Event-graph branch disabled: the log encoder now consists only of
        # CNNTransformerAE.  Keep the legacy LogGraphGCN implementation above
        # for reference, but do not instantiate or optimize it here.

    def encode(self, inputs, lengths=None, sequences=None, counts=None):
        semantic_sequence, semantic_global = self.log_model.encode(inputs, lengths)
        # ``sequences`` and ``counts`` remain in the signature for compatibility
        # with MultimodalFeatureEncoder, but the CNN-Transformer ignores them.
        return semantic_sequence, semantic_global

    def forward(self, inputs, lengths, sequences, counts):
        semantic_sequence, semantic_global = self.log_model.encode(inputs, lengths)
        reconstructed = self.log_model.decode(
            semantic_sequence + semantic_global.unsqueeze(1), lengths
        )
        return reconstructed, semantic_global


def masked_reconstruction_loss(prediction, target, lengths):
    mask = ~padding_mask(lengths, target.shape[1])
    squared_error = (prediction - target).pow(2) * mask.unsqueeze(2)
    denominator = mask.sum().clamp(min=1) * target.shape[2]
    return squared_error.sum() / denominator


def _epoch(loader, model, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    for inputs, lengths, sequences, counts, _ in tqdm(
        loader, desc="Log train" if training else "Log eval"
    ):
        inputs, lengths = inputs.to(device), lengths.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            semantic_reconstruction, _ = model(inputs, lengths, sequences, counts)
            loss = masked_reconstruction_loss(semantic_reconstruction, inputs, lengths)
            if training:
                loss.backward()
                optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def pretrain_logs(train_loader, test_loader, input_dimension, config, device):
    model = StrongLogAutoencoder(input_dimension, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    for epoch in range(config.epochs):
        train_loss = _epoch(train_loader, model, device, optimizer)
        test_loss = _epoch(test_loader, model, device)
        print("Log %d/%d: train=%.4f test=%.4f" % (epoch + 1, config.epochs, train_loss, test_loss))
        save_checkpoint(model, optimizer, epoch, config.checkpoint_dir)
    return model

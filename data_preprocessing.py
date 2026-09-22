"""KPI/log preprocessing, semantic encoding and dataloaders."""

from __future__ import annotations

import ast
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DataConfig


LOG_TEMPLATES = [
    "<*> token generate success <*>",
    "the list of all available services are redisservice1: http://0.0.0.1:9386, redisservice2: http://0.0.0.2:9387",
    "now call <*> <*> as a downstream service",
    "query = select passwards from username_table where <*>",
    "password in database of {user} is null, the user is not registered",
    "Table cloudwise_micross.system_metric_datas does not exist",
    "Query was empty",
    "Unknown column system_data_time in field list",
    "Try to get <*> service`s inst, retry for for <*> time",
    "a position error occurred while obtaining the inst of the <*>",
    "all downstream services redisservice1 and redisservice2 are unavailable",
    "dbservice1 access redis service denied",
    "service refuse",
    "Can't connect to MySQL server: connection refused",
    "Can't connect to MySQL server: timed out",
    "request <*> and param={'keys': <*>",
    "request <*> and param={'uuid': <*> 'user_id': <*>",
    "uuid expired, invalid phone request",
    "get information failed, fail code: 400",
    "uuid: <*> write redis successfully",
    "complete information: {'uuid': <*> 'user_id': <*>",
    "an error occurred in the downstream service",
    "unknown error occurred, status_code == 200",
    "No such file or directory: resources/source_file/source_file.csv",
    "call url with uuid and user_id failed: connection refused",
    "call url with uuid failed: connection refused",
]


def _window(values, length):
    if len(values) < length:
        raise ValueError("Data length is smaller than window length")
    index = np.arange(length)[None, :] + np.arange(len(values) - length + 1)[:, None]
    return values[index]


def preprocess_kpi(frame, config, scaler=None):
    selected = frame.iloc[:, list(range(config.kpi_feature_count + 1)) + [config.kpi_feature_count + 2]].copy()
    if config.timestamp_column in selected.columns:
        selected = selected.drop(columns=config.timestamp_column)
    features = selected.iloc[:, : config.kpi_feature_count].to_numpy()
    labels = selected.iloc[:, config.kpi_feature_count].to_numpy()
    if scaler is None:
        scaler = MinMaxScaler().fit(features)
    features = scaler.transform(features)
    feature_windows = _window(features, config.window_length)
    label_windows = _window(labels, config.window_length)
    window_labels = np.any(label_windows.astype(bool), axis=1).astype(np.int64)
    return pd.DataFrame({"kpi_column": feature_windows.tolist(), "label": window_labels.tolist()}), scaler


def _parse_events(value):
    if isinstance(value, list):
        events = value.copy()
    elif isinstance(value, str):
        parsed = ast.literal_eval(value)
        events = parsed if isinstance(parsed, list) else []
    else:
        events = []
    # The notebook maps IDs outside E0..E25 to E3.
    return [event if len(str(event)) <= 3 and str(event)[1:].isdigit() and int(str(event)[1:]) < 26 else "E3" for event in events]


def _least_frequent(events, count):
    frequencies = Counter(events)
    selected = [event for event, _ in frequencies.most_common()[:-count - 1:-1]]
    return sorted(selected, key=events.index) if events else []


def preprocess_log_events(frame, config, scaler=None):
    log_data = frame[[config.timestamp_column, config.event_column]].copy()
    log_data["events"] = log_data[config.event_column].map(_parse_events)
    counts = np.zeros((len(log_data), config.log_template_count), dtype=np.float32)
    raw_sequences = []
    semantic_sequences = []
    for row, events in enumerate(log_data["events"]):
        numeric = [int(event[1:]) for event in events]
        raw_sequences.append(numeric)
        semantic_sequences.append([int(event[1:]) for event in _least_frequent(events, config.least_log_events)])
        total = max(len(events), 1)
        frequencies = Counter(numeric)
        for event_id in range(config.log_template_count):
            counts[row, event_id] = frequencies[event_id] / total
    if scaler is None:
        scaler = MinMaxScaler().fit(counts)
    counts = scaler.transform(counts)
    semantic_windows, raw_windows = [], []
    for start in range(len(log_data) - config.window_length + 1):
        semantic_windows.append([
            event for row in semantic_sequences[start : start + config.window_length] for event in row
        ])
        raw_windows.append(raw_sequences[start : start + config.window_length])
    return semantic_windows, raw_windows, _window(counts, config.window_length), scaler


def encode_templates(config, device):
    try:
        from transformers import BertModel, BertTokenizer, logging
    except ImportError as exc:
        raise RuntimeError("transformers is required for log preprocessing") from exc
    logging.set_verbosity_warning()
    tokenizer = BertTokenizer.from_pretrained(str(config.bert_vocab_file))
    model = BertModel.from_pretrained(str(config.bert_model_dir)).to(device).eval()
    vectors = []
    with torch.no_grad():
        for template in tqdm(LOG_TEMPLATES, desc="BERT templates"):
            inputs = tokenizer(template, return_tensors="pt", padding=True, truncation=True, max_length=64)
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            states = model(**inputs, output_hidden_states=True).hidden_states
            vectors.append(((states[-1] + states[1]).mean(dim=1))[0].cpu().numpy())
    vectors = np.asarray(vectors)
    mean = vectors.mean(axis=0, keepdims=True)
    covariance = np.cov(vectors.T)
    left, singular, _ = np.linalg.svd(covariance)
    whitening = np.linalg.pinv((left @ np.diag(singular ** 0.5)).T)
    encoded = (vectors - mean) @ whitening[:, : config.window_length]
    return encoded / np.linalg.norm(encoded, axis=1, keepdims=True).clip(min=1e-12)


def build_processed_dataset(config, device):
    frame = pd.read_csv(config.source_csv, memory_map=True, nrows=config.max_rows)
    embeddings = encode_templates(config, device)
    train_end = config.train_rows
    cluster_end = train_end + config.cluster_rows
    if train_end < config.window_length or config.cluster_rows < config.window_length:
        raise ValueError("Train and cluster partitions must each contain at least one full window")
    if len(frame) - cluster_end < config.window_length:
        raise ValueError("Test partition must contain at least one full window")
    raw_partitions = {
        "train": frame.iloc[:train_end].copy(),
        "cluster": frame.iloc[train_end:cluster_end].copy(),
        "test": frame.iloc[cluster_end:].copy(),
    }
    kpi_scaler = None
    log_scaler = None
    partitions = []
    for split_name in ("train", "cluster", "test"):
        raw = raw_partitions[split_name]
        kpi, kpi_scaler = preprocess_kpi(raw, config, kpi_scaler)
        semantic_ids, raw_sequences, count_windows, log_scaler = preprocess_log_events(
            raw, config, log_scaler
        )
        semantic = [[embeddings[event].tolist() for event in window] for window in semantic_ids]
        logs = pd.DataFrame({
            "Column": semantic,
            "seq": raw_sequences,
            "count": [window.T.tolist() for window in count_windows],
        })
        merged = pd.concat([logs, kpi], axis=1)
        merged = merged[merged["Column"].map(bool)].dropna(subset=["Column"]).reset_index(drop=True)
        merged["logs_len"] = merged["Column"].map(len)
        merged["split"] = split_name
        partitions.append(merged)
    merged = pd.concat(partitions, ignore_index=True)
    merged.to_pickle(config.processed_file)
    merged.to_csv(config.processed_file.with_suffix(".csv"), index=False)
    return merged


def load_processed_dataset(config):
    if not config.processed_file.exists():
        raise FileNotFoundError("Run the preprocess stage first: %s" % config.processed_file)
    return pd.read_pickle(config.processed_file)


class StrongLogDataset(Dataset):
    def __init__(self, frame):
        self.rows = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        return row["Column"], int(row["logs_len"]), int(row["label"]), row["seq"], row["count"]


def collate_logs(batch):
    maximum = max(item[1] for item in batch)
    dimension = len(batch[0][0][0])
    padded, lengths, labels, sequences, counts = [], [], [], [], []
    for semantic, length, label, sequence, count in batch:
        padded.append(list(semantic) + [[0.0] * dimension for _ in range(maximum - len(semantic))])
        lengths.append(length)
        labels.append(label)
        sequences.append(sequence)
        counts.append(count)
    return (
        torch.tensor(padded, dtype=torch.float32),
        torch.tensor(lengths, dtype=torch.long),
        sequences,
        torch.tensor(counts, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def make_dataloaders(frame, config, batch_size):
    if "split" not in frame.columns:
        raise ValueError("Processed data is stale: rerun the preprocess stage to create disjoint splits")
    loaders = {}
    for split_name in ("train", "cluster", "test"):
        partition = frame[frame["split"] == split_name].reset_index(drop=True)
        kpi_values = torch.tensor(np.asarray(partition["kpi_column"].tolist()), dtype=torch.float32)
        loaders["kpi_" + split_name] = DataLoader(
            kpi_values, batch_size=batch_size, shuffle=False, drop_last=False
        )
        loaders["log_" + split_name] = DataLoader(
            StrongLogDataset(partition), batch_size=batch_size, shuffle=False,
            drop_last=False, collate_fn=collate_logs,
        )
    return loaders

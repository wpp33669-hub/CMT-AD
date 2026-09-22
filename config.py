"""Central configuration for the stronger CMKT pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    source_csv: Path = Path(r"G:\wpp\dataset and paper\gaia\pre-process\three_node_kpi_log_aligned.csv")
    bert_model_dir: Path = Path(r"D:\wpp\AD_CMKT\Bert\bert-base-uncased")
    bert_vocab_file: Path = Path(r"D:\wpp\AD_CMKT\Bert\bert-base-uncased-vocab.txt")
    processed_file: Path = Path("df_kpi_log.pkl")
    max_rows: int = 36_000
    train_rows: int = 20_000
    cluster_rows: int = 8_000
    window_length: int = 20
    kpi_feature_count: int = 83
    log_template_count: int = 26
    least_log_events: int = 5
    timestamp_column: str = "timestamp"
    event_column: str = "merged_logs"


@dataclass
class KPIConfig:
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 1e-2
    lstm_hidden: int = 32
    top_k: int = 25
    checkpoint_dir: Path = Path("checkpoints_VGAE_KPI")


@dataclass
class LogConfig:
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    d_model: int = 128
    cnn_channels: int = 128
    cnn_kernel_size: int = 3
    nhead: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 2
    feedforward_dimension: int = 256
    dropout: float = 0.1
    max_sequence_length: int = 200
    graph_hidden: int = 128
    # CNN-Transformer-only checkpoints.  Keep them separate from the older
    # CNN-Transformer + event-graph GCN checkpoints, which are incompatible.
    checkpoint_dir: Path = Path("checkpoints_AE_Log_CNN_Transformer")


@dataclass
class CMKTConfig:
    # Paper notation: modality hidden dimension d_m and common dimension d_c.
    modality_dimension: int = 32
    common_dimension: int = 32
    # Student-t parameter used by DEC; this is not the paper's regularization alpha.
    dec_alpha: float = 1.0
    pkt_temperature: float = 0.2
    # Paper notation: structural-consistency balance beta.
    structural_weight: float = 0.2
    semantic_prototypes: int = 16
    semantic_temperature: float = 0.2
    # Paper notation: semantic-consistency balance lambda.
    semantic_weight: float = 0.05
    detach_gate: bool = True
    gate_without_centers: str = "equal"
    alignment_epochs: int = 5
    alignment_learning_rate: float = 1e-3


@dataclass
class ClusterConfig:
    clusters: int = 3
    epochs: int = 10
    learning_rate: float = 1e-2
    weight_decay: float = 1e-2
    # Paper notation: overall regularization balance alpha.
    regularization_weight: float = 0.06
    contrastive_weight: float = 0.05
    checkpoint_dir: Path = Path("checkpoints_DEC_paper_aligned")


@dataclass
class ExperimentConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    kpi: KPIConfig = field(default_factory=KPIConfig)
    log: LogConfig = field(default_factory=LogConfig)
    cmkt: CMKTConfig = field(default_factory=CMKTConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

"""Entry point for the stronger CMKT pipeline; safe to run directly in PyCharm."""

from __future__ import annotations

import argparse
from pathlib import Path

from clustering import (
    DECModel,
    evaluate_dec,
    initialize_cluster_centers,
    pretrain_cross_modal,
    train_dec,
)
from config import ExperimentConfig
from cross_modal import MultimodalFeatureEncoder
from data_preprocessing import build_processed_dataset, load_processed_dataset, make_dataloaders
from kpi_pretraining import KPIAutoencoder, pretrain_kpi
from log_pretraining import StrongLogAutoencoder, pretrain_logs
from regularization import CMKTRegularizer
from utils import get_device, latest_weights, load_weights, seed_everything


def load_pretrained(config, device, log_checkpoint=None, kpi_checkpoint=None):
    log_model = StrongLogAutoencoder(config.data.window_length, config.log).to(device)
    kpi_model = KPIAutoencoder(config.data, config.kpi).to(device)
    load_weights(log_model, log_checkpoint or latest_weights(config.log.checkpoint_dir), device)
    load_weights(kpi_model, kpi_checkpoint or latest_weights(config.kpi.checkpoint_dir), device)
    return log_model, kpi_model


def run(args):
    config = ExperimentConfig()
    if args.source is not None:
        config.data.source_csv = args.source
    if args.bert_model is not None:
        config.data.bert_model_dir = args.bert_model
    if args.bert_vocab is not None:
        config.data.bert_vocab_file = args.bert_vocab
    if args.processed is not None:
        config.data.processed_file = args.processed
    config.data.max_rows = args.max_rows
    config.data.train_rows = args.train_rows
    config.data.cluster_rows = args.cluster_rows
    seed_everything(config.seed)
    device = get_device()
    print("PyTorch device:", device)

    if args.stage in ("preprocess", "all"):
        frame = build_processed_dataset(config.data, device)
        print("Processed %d windows -> %s" % (len(frame), config.data.processed_file))
        if args.stage == "preprocess":
            return
    else:
        frame = load_processed_dataset(config.data)

    loaders = make_dataloaders(frame, config.data, config.kpi.batch_size)
    print("Window counts:", {
        split: len(loaders["log_" + split].dataset)
        for split in ("train", "cluster", "test")
    })
    kpi_model = None
    log_model = None
    if args.stage in ("pretrain-kpi", "all"):
        kpi_model = pretrain_kpi(
            loaders["kpi_train"], loaders["kpi_cluster"],
            config.data, config.kpi, device,
        )
        if args.stage == "pretrain-kpi":
            return
    if args.stage in ("pretrain-log", "all"):
        log_model = pretrain_logs(
            loaders["log_train"], loaders["log_cluster"],
            config.data.window_length, config.log, device,
        )
        if args.stage == "pretrain-log":
            return

    if args.stage in ("cluster", "all"):
        if log_model is None or kpi_model is None:
            log_model, kpi_model = load_pretrained(
                config, device, args.log_checkpoint, args.kpi_checkpoint
            )
        encoder = MultimodalFeatureEncoder(
            log_model,
            kpi_model,
            config.log.d_model,
            config.data.window_length // 2,
            config.cmkt.modality_dimension,
        ).to(device)
        regularizer = CMKTRegularizer(config.cmkt).to(device)
        pretrain_cross_modal(
            loaders["log_cluster"], loaders["kpi_cluster"],
            encoder, regularizer, config.cmkt, device,
        )
        centers = initialize_cluster_centers(
            loaders["log_cluster"], loaders["kpi_cluster"],
            encoder, regularizer, config.cluster.clusters, device,
        )
        model = DECModel(encoder, regularizer, centers, config.cmkt.dec_alpha).to(device)
        train_dec(
            loaders["log_cluster"], loaders["kpi_cluster"],
            model, config.cluster, device,
        )
        evaluate_dec(
            loaders["log_test"], loaders["kpi_test"], model, device
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Stronger CMKT multimodal clustering")
    parser.add_argument(
        "stage", nargs="?", default="all",
        choices=("preprocess", "pretrain-kpi", "pretrain-log", "cluster", "all"),
        help="pipeline stage (default: all)",
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--bert-model", type=Path)
    parser.add_argument("--bert-vocab", type=Path)
    parser.add_argument("--processed", type=Path)
    parser.add_argument("--max-rows", type=int, default=36_000)
    parser.add_argument("--train-rows", type=int, default=20_000)
    parser.add_argument("--cluster-rows", type=int, default=8_000)
    parser.add_argument("--log-checkpoint", type=Path)
    parser.add_argument("--kpi-checkpoint", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

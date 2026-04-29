"""
Flags:
- data_dir: str (path to training data directory)
- model_dir: str (path to saved model directory)
- config_file: str (path to training config file)
- train: bool (train the model)
- infer: bool (run inference)
- infer_dir: str (path to inference data directory)
- model_path: str (path to saved model)

Example:

python src/jepacluster/main.py --train --data_dir data/ --model_dir models/ --config_file config.yaml
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from utils.logger import get_logger
from utils.config import Config
from core.trainer import Trainer
from engine.embedder import Embedder
from engine.clusterer import Clusterer
from engine.analyzer import Analyzer

import torch


logger = get_logger(__name__)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test torch availability")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--infer", action="store_true", help="Run inference")
    parser.add_argument("--infer_dir", type=str, required=False, help="Path to inference data directory", default="infer_data")
    parser.add_argument("--data_dir", type=str, required=False, help="Path to training data directory", default="data")
    parser.add_argument("--model_dir", type=str, required=False, help="Path to saved model directory", default="models")
    parser.add_argument("--model_path", type=str, required=False, help="Path to saved model", default="jepacluster-v1.pt")
    parser.add_argument("--config_file", type=str, required=False, help="Path to training config file", default="config.yaml")
    parser.add_argument("--resume_checkpoint", type=str, required=False, default=None, help="Path to a training checkpoint to resume from")

    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting jepacluster with args: {args}")

    config = Config.validate_config(args.config_file)
    logger.info(f"Config: {config}")

    if args.test:
        logger.info("Testing torch availability")
        logger.info(f"CUDA is available: {torch.cuda.is_available()}")
    elif args.train:
        logger.info("Training the model")

        if args.resume_checkpoint:
            config.training.resume_checkpoint = args.resume_checkpoint

        trainer = Trainer(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            model_config=config.model,
            training_config=config.training
        )
        trainer.train()

        logger.info("Model trained successfully")
    elif args.infer:
        logger.info("Running inference")

        embedder = Embedder(model_path=args.model_path)
        embeddings, samples = embedder.embed(args.infer_dir)
        embeddings_np = embeddings.detach().cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.asarray(embeddings)

        clusterer = Clusterer(clustering_config=config.clustering)
        clusters = clusterer.cluster(embeddings=embeddings_np)

        analyzer = Analyzer(clusters=clusters, samples=samples, embeddings=embeddings_np)
        summary = analyzer.analyze()
        logger.info("Inference summary: %s", summary)
        analyzer.visualize(output_path=str(Path(args.model_dir) / "cluster_viz.png"))
    
        logger.info("Inference completed successfully")

if __name__ == "__main__":
    main()

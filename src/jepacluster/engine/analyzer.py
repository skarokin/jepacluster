"""
This file contains the analyzer: cluster interpretation.
"""
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Sequence

from utils.logger import get_logger

logger = get_logger(__name__)

class Analyzer:
    def __init__(self, clusters: np.ndarray, samples: Sequence[str] | None = None, embeddings: np.ndarray | None = None):
        self.clusters = clusters
        self.samples = list(samples) if samples is not None else None
        self.embeddings = embeddings

    def analyze(self):
        if self.clusters.size == 0:
            logger.info("No clusters to analyze.")
            return {"total": 0, "clusters": {}}

        counts = Counter(self.clusters.tolist())
        noise = counts.pop(-1, 0)
        cluster_examples: dict[int, list[str]] = {}
        if self.samples is not None:
            for label in sorted(counts):
                members = [sample for sample, cluster in zip(self.samples, self.clusters.tolist()) if cluster == label]
                cluster_examples[int(label)] = members[:3]

        summary = {
            "total": int(self.clusters.size),
            "noise": int(noise),
            "clusters": {int(label): int(count) for label, count in sorted(counts.items())},
            "examples": cluster_examples,
        }
        logger.info("Cluster summary: %s", summary)
        return summary

    def visualize(self, output_path: str | None = None):
        if self.embeddings is None or self.embeddings.size == 0 or self.clusters.size == 0:
            logger.info("No embeddings or clusters available for visualization.")
            return None

        try:
            import matplotlib.pyplot as plt  # type: ignore[import-not-found]
            import seaborn as sns  # type: ignore[import-not-found]
            from sklearn.decomposition import PCA
        except ImportError:
            logger.warning("matplotlib, seaborn, or scikit-learn is not installed; skipping visualization.")
            return None

        reduced = PCA(n_components=2, random_state=42).fit_transform(self.embeddings)
        fig, ax = plt.subplots(figsize=(10, 7))
        palette = sns.color_palette("tab10", n_colors=max(1, len(set(self.clusters.tolist()))))

        labels = self.clusters.tolist()
        unique_labels = sorted(set(labels))
        for idx, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            color = "#999999" if label == -1 else palette[idx % len(palette)]
            name = "noise" if label == -1 else f"cluster {label}"
            ax.scatter(reduced[mask, 0], reduced[mask, 1], s=24, alpha=0.8, label=name, color=color)
            if label != -1 and self.samples is not None and mask.any():
                center = reduced[mask].mean(axis=0)
                best_idx = np.argmin(((reduced[mask] - center) ** 2).sum(axis=1))
                cluster_samples = np.array(self.samples, dtype=object)[mask]
                label_text = str(cluster_samples[best_idx])
                ax.annotate(
                    label_text[:80],
                    (reduced[mask][best_idx, 0], reduced[mask][best_idx, 1]),
                    fontsize=7,
                    alpha=0.8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )

        ax.set_title("JEPA log embedding clusters")
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180)
            logger.info("Saved cluster visualization to %s", path)
        return fig
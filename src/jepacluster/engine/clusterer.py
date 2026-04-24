"""
This file contains the clusterer: clustering the latent vectors.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

from utils.logger import get_logger
from utils.config import ClusteringConfig

logger = get_logger(__name__)

class Clusterer:
    def __init__(self, clustering_config: ClusteringConfig):
        self.clustering_config = clustering_config

    def cluster(self, embeddings: np.ndarray):
        if embeddings.size == 0:
            return np.empty(0, dtype=np.int64)

        scaled = StandardScaler().fit_transform(embeddings)
        try:
            import umap  # type: ignore[import-not-found]
            import hdbscan  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("umap or hdbscan is not installed; falling back to no-op clustering.")
            return np.full(scaled.shape[0], -1, dtype=np.int64)

        reducer = umap.UMAP(
            n_components=self.clustering_config.umap.n_components,
            n_neighbors=self.clustering_config.umap.n_neighbors,
            min_dist=self.clustering_config.umap.min_dist,
            metric="cosine",
            random_state=42,
        )
        reduced = reducer.fit_transform(scaled)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.clustering_config.hdbscan.min_cluster_size,
            min_samples=self.clustering_config.hdbscan.min_samples,
            prediction_data=True,
        )
        labels = clusterer.fit_predict(reduced)
        logger.info("Clustered %d embeddings into %d clusters", len(labels), len(set(labels)) - (1 if -1 in labels else 0))
        return labels
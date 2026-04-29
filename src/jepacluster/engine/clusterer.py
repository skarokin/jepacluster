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

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings)

        # reshape to 2D array if 1D because HDBSCAN expects a 2D array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)

        # no embeddings - return empty array
        if embeddings.size == 0 or embeddings.shape[0] == 0:
            return np.empty(0, dtype=np.int64)

        # not a 2D array - raise error
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be a 2D array, got shape {embeddings.shape}")

        # less than 2 embeddings - return all -1
        if embeddings.shape[0] < 2:
            return np.full(embeddings.shape[0], -1, dtype=np.int64)

        scaled = StandardScaler().fit_transform(embeddings)
        try:
            import umap  # type: ignore[import-not-found]
            import hdbscan  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("umap or hdbscan is not installed; falling back to no-op clustering.")
            return np.full(scaled.shape[0], -1, dtype=np.int64)

        # calculate n_neighbors for UMAP
        n_neighbors = min(self.clustering_config.umap.n_neighbors, max(2, scaled.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=self.clustering_config.umap.n_components,
            n_neighbors=n_neighbors,
            min_dist=self.clustering_config.umap.min_dist,
            metric="cosine",
            random_state=42,
        )
        reduced = reducer.fit_transform(scaled)

        # set HDBSCAN parameters
        min_cluster_size = min(self.clustering_config.hdbscan.min_cluster_size, scaled.shape[0])
        min_samples = self.clustering_config.hdbscan.min_samples
        if min_samples is not None:
            min_samples = min(min_samples, scaled.shape[0])
        cluster_selection_epsilon = max(0.0, float(self.clustering_config.hdbscan.prediction_threshold))

        # create HDBSCAN clusterer
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, min_cluster_size),
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            prediction_data=True,
        )
        # fit and predict clusters
        labels = clusterer.fit_predict(reduced)
        # calculate number of clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info("Clustered %d embeddings into %d clusters", len(labels), n_clusters)
        return labels
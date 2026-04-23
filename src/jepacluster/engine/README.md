# jepacluster/src/jepacluster/engine

This directory handles inference.

- **embedder.py**: Loads the trained context encoder.
- **clusterer.py**: Logic for grouping behaviors. Utilizes **UMAP** for dimensionality reduction and **HDBSCAN** for clustering.
- **analyzer.py**: Interprets the clusters. Extracts representative logs (centroids) from each cluster to provide human-readable labels for discovered system states.
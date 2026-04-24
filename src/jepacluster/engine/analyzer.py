"""
This file contains the analyzer: cluster interpretation.
"""
import numpy as np
from collections import Counter

from utils.logger import get_logger

logger = get_logger(__name__)

class Analyzer:
    def __init__(self, clusters: np.ndarray):
        self.clusters = clusters

    def analyze(self):
        if self.clusters.size == 0:
            logger.info("No clusters to analyze.")
            return {"total": 0, "clusters": {}}

        counts = Counter(self.clusters.tolist())
        noise = counts.pop(-1, 0)
        summary = {
            "total": int(self.clusters.size),
            "noise": int(noise),
            "clusters": {int(label): int(count) for label, count in sorted(counts.items())},
        }
        logger.info("Cluster summary: %s", summary)
        return summary
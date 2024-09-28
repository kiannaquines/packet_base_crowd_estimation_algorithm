from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class ClusteringAlgorithm:
    def __init__(self, eps, min_samples, X):
        self.eps = eps
        self.min_samples = min_samples
        self.features = X
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = None
        self.score = None
        self.minimum_labels = 1

    def fit_predict(self):
        """Performs DBSCAN clustering and calculates the silhouette score."""
        try:
            self.labels = self.dbscan.fit_predict(self.features)
            if len(set(self.labels)) > self.minimum_labels:
                self.score = silhouette_score(self.features, self.labels)
                logging.info(f"Silhouette Score: {self.score}")
                return self.eps, self.min_samples, self.score
            else:
                logging.info("Clustering resulted in too few labels.")
        except Exception as e:
            logging.error(f"Error during DBSCAN clustering: {e}")
        
        return None


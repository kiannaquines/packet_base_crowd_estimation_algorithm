import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from joblib import Parallel, delayed
from utilities.database import Database
from utilities.dbscan import ClusteringAlgorithm
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class PackBaseEstimationAlgorithm:
    def __init__(self):
        self.eps_values = np.arange(0.1, 1.0, 0.1)
        self.min_samples_values = range(1, 10)
        self.label_encoder = LabelEncoder()
        self.minimum_probe_packets = 25
        self.best_eps = None
        self.best_min_samples = None
        self.scaler = MinMaxScaler()
        self.database = Database()
        self.dataframe = None
        self.X_features = None
        self.results = None

    def load_data(self, data):
        self.dataframe = data

    def data_processing(self):
        mask = self.dataframe['frame_type'].isin(['Probe Request', 'Probe Response'])
        filtered_probe_packets = self.dataframe[mask]
        processed_data = filtered_probe_packets.groupby('device_addr').filter(lambda x: len(x) > self.minimum_probe_packets)
        combined_packets = pd.concat([processed_data, self.dataframe[~mask]])
        logging.info("Data processed successfully.")
        return combined_packets

    def data_scaler(self, data):
        rssi_values = data['device_power'].values.reshape(-1, 1)
        normalized_rssi = self.scaler.fit_transform(rssi_values)
        data['normalized_power'] = normalized_rssi
        logging.info("RSSI values scaled successfully.")
        return data

    def data_encoder(self, data):
        data['device_addr_encoded'] = self.label_encoder.fit_transform(data['device_addr'])
        data['zone_encoded'] = self.label_encoder.fit_transform(data['zone'])
        logging.info("Data encoding completed.")
        return data

    def features(self, data):
        self.X_features = data[['device_addr_encoded', 'is_randomized', 'zone_encoded']].values
        return self.X_features

    def evaluate_best_params_grid_search(self):
        self.results = Parallel(n_jobs=-1)(
            delayed(self._run_clustering)(eps, min_samples)
            for eps in self.eps_values
            for min_samples in self.min_samples_values
        )
        
        self.results = [r for r in self.results if r is not None]

        if self.results:
            self.best_eps, self.best_min_samples, best_silhouette_score = max(self.results, key=lambda x: x[2])
            logging.info(f"Best params found: eps={self.best_eps}, min_samples={self.best_min_samples}, silhouette_score={best_silhouette_score}")
            return self.best_eps, self.best_min_samples, best_silhouette_score
        else:
            logging.warning("No valid clustering results found during grid search.")
            return None

    def _run_clustering(self, eps, min_samples):
        clustering = ClusteringAlgorithm(eps=eps, min_samples=min_samples, X=self.X_features)
        return clustering.predict()

    def cluster_with_best_result(self, data):
        if self.best_eps and self.best_min_samples:
            clustering = ClusteringAlgorithm(eps=self.best_eps, min_samples=self.best_min_samples, X=self.X_features)
            labels, _ = clustering.fit_predict()
            data['cluster'] = labels
            logging.info("Clustering performed with best parameters.")
        else:
            logging.error("Best parameters not set. Call 'evaluate_best_params_grid_search()' first.")
            raise ValueError("Best parameters not set.")

    def export_csv(self, data, filename='clustered_data.csv'):
        export_columns = ['device_addr', 'device_addr_encoded', 'zone', 'zone_encoded', 'device_power', 'timestamp', 'cluster']
        try:
            data[export_columns].sort_values(by=['cluster', 'zone', 'device_addr']).to_csv(filename, index=False)
            logging.info(f"Data exported to {filename}.")
        except Exception as e:
            logging.error(f"Failed to export data: {e}")
            raise

    def aggregate_cluster_counts(self, data):
        cluster_aggregation = data.groupby('zone').agg(
            device_count=('cluster', 'nunique'),
            time_start=('timestamp', 'min'),
            time_ended=('timestamp', 'max')
        ).reset_index()

        cluster_aggregation['scanned_minutes'] = (cluster_aggregation['time_ended'] - cluster_aggregation['time_start']).dt.total_seconds() / 60
        logging.info("Cluster counts aggregated successfully.")
        return cluster_aggregation

    def save_cluster_counts(self, aggregated_data):
        try:
            self.database.insert_estimated_people(aggregated_data)
            logging.info("Cluster counts saved to the database.")
        except Exception as e:
            logging.error(f"Failed to save cluster counts: {e}")
            raise

    def validate_data(self, data):
        if not isinstance(data, pd.DataFrame):
            logging.error("The input data should be a pandas DataFrame.")
            return False

        if data.empty:
            logging.error("The DataFrame is empty.")
            return False

        return True

    def run(self):

        if not self.validate_data(self.dataframe):
            logging.error("Data validation failed. Aborting execution.")
            return
        
        processed_data = self.data_processing()
        scaled_data = self.data_scaler(processed_data)
        encoded_data = self.data_encoder(scaled_data)
        self.features(encoded_data)
        self.evaluate_best_params_grid_search()
        self.cluster_with_best_result(encoded_data)
        self.export_csv(encoded_data)
        aggregated_data = self.aggregate_cluster_counts(encoded_data)
        self.save_cluster_counts(aggregated_data)
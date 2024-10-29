import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
from utilities.database import Database
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class PackBaseEstimationAlgorithm:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.minimum_probe_packets = 25
        self.scaler = MinMaxScaler()
        self.database = Database()
        self.dataframe = None
        self.X_features = None
        self.results = None
        self.mean_shift = MeanShift()
        self.cluster_score = None
        self.data_to_send = []

    def load_data(self, data):
        self.dataframe = data

    def data_processing(self):
        mask = self.dataframe['frame_type'].isin(['Probe Request', 'Probe Response'])
        filtered_probe_packets = self.dataframe[mask]
        processed_data = filtered_probe_packets.groupby('device_addr').filter(lambda x: len(x) > self.minimum_probe_packets)
        
        if processed_data.empty:
            logging.warning("No valid processed data found.")
            return pd.DataFrame()

        combined_packets = pd.concat([processed_data, self.dataframe[~mask]], ignore_index=True)
        logging.info("Data processed successfully.")
        return combined_packets

    def data_scaler(self, data):
        rssi_values = data['device_power'].values.reshape(-1, 1)
        normalized_rssi = self.scaler.fit_transform(rssi_values)
        data['normalized_power'] = normalized_rssi
        return data

    def data_encoder(self, data):
        data['device_addr_encoded'] = self.label_encoder.fit_transform(data['device_addr'])
        data['zone_encoded'] = self.label_encoder.fit_transform(data['zone'])
        return data

    def features(self, data):
        self.X_features = data[['device_addr_encoded', 'is_randomized', 'zone_encoded']].values
        return self.X_features

    def mean_shift_clustering(self, features):
        try:
            labels = self.mean_shift.fit_predict(features)
            return labels
        except Exception as e:
            logging.error(f"Error during MeanShift clustering: {e}")
            return None

    def aggregate_cluster_counts(self, data):
        data.to_csv('cluster_check.csv',index=False)
        cluster_aggregation = data.groupby(['zone','zone_id']).agg(
            device_count=('cluster', 'nunique'),
            time_start=('date_detected', 'min'),
            time_ended=('date_detected', 'max')
        ).reset_index()

        cluster_aggregation['scanned_minutes'] = (cluster_aggregation['time_ended'] - cluster_aggregation['time_start']).dt.total_seconds() / 60
        logging.info("Cluster counts aggregated successfully.")
        return cluster_aggregation

    def save_cluster_counts(self, aggregated_data):
        try:
            self.database.insert_estimated_people(aggregated_data, silhouette_score=self.cluster_score)
            logging.info("Cluster counts saved to the database.")
            
        except Exception as e:
            logging.error(f"Failed to save cluster counts: {e}")
            raise

    def to_sent_data(self, aggregated_data):
        import random
        for _, row in aggregated_data.iterrows():
            result = {
                'id': random.randint(1, 10000),
                'zone': row['zone'],
                'estimated_count': row['device_count'],
                'first_seen': row['time_start'].strftime('%Y-%m-%d %H:%M:%S'),
                'last_seen': row['time_ended'].strftime('%Y-%m-%d %H:%M:%S'),
                'scanned_minutes': row['scanned_minutes']
            }
            self.data_to_send.append(result)

        logging.info(f"Cluster counts saved to the database and results appended to data_to_send")

    def validate_data(self, data):
        if not isinstance(data, pd.DataFrame):
            logging.error("The input data should be a pandas DataFrame.")
            return False

        if data.empty:
            logging.error("The DataFrame is empty.")
            return False

        return True
    
    def clear(self):
        self.data_to_send = []

    def run(self):
        if not self.validate_data(self.dataframe):
            logging.error("Data validation failed. Aborting execution.")
            return
        
        processed_data = self.data_processing()
        if processed_data.empty:
            logging.error("No data available for clustering after processing. Aborting execution.")
            return
        
        scaled_data = self.data_scaler(processed_data)
        encoded_data = self.data_encoder(scaled_data)
        self.features(encoded_data)
        
        cluster_labels = self.mean_shift_clustering(self.X_features)
        if cluster_labels is None:
            logging.error("No cluster result. Aborting execution.")
            return
        
        encoded_data['cluster'] = cluster_labels
        
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(self.X_features, cluster_labels)
            self.cluster_score = silhouette_avg
        else:
            logging.warning("Not enough clusters to calculate silhouette score.")

        aggregated_data = self.aggregate_cluster_counts(encoded_data)
        self.save_cluster_counts(aggregated_data)
        self.to_sent_data(aggregated_data)

        return self.data_to_send
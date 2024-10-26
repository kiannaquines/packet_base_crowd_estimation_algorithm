from urllib.parse import quote
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import logging
import pandas as pd

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class Database:
    def __init__(self):
        load_dotenv()
        self.username = os.getenv("MYSQL_USERNAME")
        self.password = os.getenv("MYSQL_PASSWORD")
        self.host = os.getenv("MYSQL_HOST")
        self.database_name = os.getenv("MYSQL_DATABASE_NAME")
        self.driver = os.getenv("DATABASE_DRIVER")
        self.connector = os.getenv("DATABASE_CONNECTOR")

        if not all([self.username, self.password, self.host, self.database_name, self.driver, self.connector]):
            raise ValueError("One or more environment variables are missing.")

        self.parsed_password = quote(self.password)
        self.engine = create_engine(
            f"{self.driver}+{self.connector}://{self.username}:{self.parsed_password}@{self.host}/{self.database_name}",
            pool_pre_ping=True,
        )

    def connect(self):
        try:
            connection = self.engine.connect()
            logging.info("Database connected successfully.")
            return connection
        except Exception as e:
            logging.error(f"Failed to connect to the database: {e}")
            raise

    def create_table(self, query):
        with self.connect() as connection:
            try:
                connection.execute(text(query))
                logging.info("Table created/modified successfully.")
            except Exception as e:
                logging.error(f"Error creating table: {e}", exc_info=True)
                raise

    def fetch_data(self):
        query = """
        SELECT devices.id AS id, device_addr, date_detected, is_randomized, device_power, frame_type, devices.zone AS zone_id, zones.name AS zone FROM devices JOIN zones ON devices.zone = zones.id WHERE devices.processed = False
        """
        with self.connect() as connection:
            try:
                return pd.read_sql(query, connection)
            except Exception as e:
                logging.error(f"Error fetching data: {e}", exc_info=True)
                raise

    def insert_estimated_people(self, data, silhouette_score=0.0):
        query = text("""
            INSERT INTO predictions (zone_id, score, estimated_count, first_seen, last_seen, scanned_minutes, is_displayed)
            VALUES (:zone_id, :score, :estimated_count, :first_seen, :last_seen, :scanned_minutes, :is_displayed)
        """)

        with self.connect() as connection:
            try:
                for i, row in data.iterrows():
                    connection.execute(query, {
                        'zone_id': row['zone_id'],
                        'score': silhouette_score,
                        'estimated_count': row['device_count'],
                        'first_seen': row['time_start'],
                        'last_seen': row['time_ended'],
                        'scanned_minutes': row['scanned_minutes'],
                        'is_displayed': False,
                    })
                    logging.info(f"Row {i+1} data inserted into 'predictions' table.")

                connection.commit()

            except Exception as e:
                connection.rollback()
                logging.error(f"Error inserting data: {e}", exc_info=True)
                raise

    def update_device_info(self, data):
        device_ids = [int(id_) for id_ in data['id'].tolist()]
        
        if not device_ids:
            return

        placeholders = ', '.join([':id' + str(i) for i in range(len(device_ids))])
        
        query = text(f"""
            UPDATE devices 
            SET processed = True
            WHERE id IN ({placeholders})
        """)
        
        with self.connect() as connection:
            try:
                params = {f'id{i}': device_ids[i] for i in range(len(device_ids))}
                connection.execute(query, params)
                connection.commit()
                logging.info(f"Successfully updated {len(device_ids)} devices")
                
            except Exception as e:
                logging.error(f"Error updating device info", exc_info=True)
                connection.rollback()
                raise

    def dispose(self):
        self.engine.dispose()
        logging.info("Database connection engine disposed.")
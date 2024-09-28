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
            SELECT device_id, device_addr, timestamp, is_randomized, device_power, ssid, frame_type, zone 
            FROM devices 
            WHERE processed = False
        """
        with self.connect() as connection:
            try:
                return pd.read_sql(query, connection)
            except Exception as e:
                logging.error(f"Error fetching data: {e}", exc_info=True)
                raise

    def insert_estimated_people(self, data):
        query = text("""
            INSERT INTO estimated_crowd (estimated_crowd, estimated_crowd_zone, first_seen, last_seen, scanned_minutes)
            VALUES (:estimated_crowd, :estimated_crowd_zone, :first_seen, :last_seen, :scanned_minutes)
        """)

        with self.connect() as connection:
            try:
                for i, row in data.iterrows():
                    connection.execute(query, {
                        'estimated_crowd': row['device_count'],
                        'estimated_crowd_zone': row['zone'],
                        'first_seen': row['time_start'],
                        'last_seen': row['time_ended'],
                        'scanned_minutes': row['scanned_minutes']
                    })
                    logging.info(f"Row {i} data inserted into 'estimated_crowd' table.")
            except Exception as e:
                logging.error(f"Error inserting data: {e}", exc_info=True)
                raise

    def update_device_info(self, data):
        query = text(
            "UPDATE devices SET processed = True WHERE device_id IN :device_ids"
        )

        with self.connect() as connection:
            try:
                device_ids = data['device_id'].tolist()
                connection.execute(query, {'device_ids': tuple(device_ids)})
                logging.info("Device data updated successfully.")
            except Exception as e:
                logging.error(f"Error updating device info: {e}", exc_info=True)
                raise

    def dispose(self):
        self.engine.dispose()
        logging.info("Database connection engine disposed.")
from utilities.packet_base_estimation_algorithm import PackBaseEstimationAlgorithm
from utilities.websocket import WebSocketServer
from utilities.database import Database
import json
import logging
import asyncio

class DataProcessingService:
    def __init__(self):
        self.database = Database()
        self.algorithm = PackBaseEstimationAlgorithm()
        self.ws_server = WebSocketServer()
        self.logger = logging.getLogger(__name__)
        
    async def process_single_batch(self) -> bool:
        try:
            data = self.database.fetch_data()
            print(data.head())
            if 'id' not in data.columns or data['id'].empty:
                self.logger.warning("No id column or data found in fetched data.")
                return False
                
            self.algorithm.load_data(data)
            result = self.algorithm.run()
            
            if result:
                for item in result:
                    message = json.dumps(item)
                    await self.ws_server.send_to_clients(message)
                    self.logger.debug(f"Broadcasted message: {message}")
                
                self.database.update_device_info(data[['id']])
                self.algorithm.clear()
                return True
            else:
                self.logger.debug('Empty result, wait for another clustering')
                return False
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}", exc_info=True)
            return False

    async def fetch_and_process_data(self):
        while True:
            try:
                success = await self.process_single_batch()
                if not success:
                    self.logger.warning("Batch processing failed, waiting before retry")
            except Exception as e:
                self.logger.error(f"Critical error in processing loop: {e}", exc_info=True)
            
            for _ in range(60):
                await asyncio.sleep(1)
                await self.ws_server.send_to_clients("ping")
                self.logger.debug("Sent ping to keep WebSocket alive")

async def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        data_service = DataProcessingService()
        
        ws_server_task = asyncio.create_task(
            data_service.ws_server.start(),
            name="websocket_server"
        )
        
        process_task = asyncio.create_task(
            data_service.fetch_and_process_data(),
            name="data_processing"
        )
        
        await asyncio.gather(ws_server_task, process_task)
        
    except Exception as e:
        logging.error(f"Critical error in main: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())

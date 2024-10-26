from websockets import serve, WebSocketException, ConnectionClosedError
import logging
import asyncio
from typing import Set
from contextlib import suppress

logging.getLogger('websockets').disabled = True

class WebSocketServer:
    def __init__(self):
        self.clients: Set = set()
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the WebSocket server"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def register(self, websocket) -> None:
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        self.logger.info(f"Client connected. Total clients: {len(self.clients)}")

    async def unregister(self, websocket) -> None:
        with suppress(KeyError):
            self.clients.remove(websocket)
            self.logger.info(f"Client disconnected. Total clients: {len(self.clients)}")

    async def safe_send(self, client, message: str) -> bool:
        try:
            await client.send(message)
            return True
        except ConnectionClosedError as e:
            self.logger.warning(f"Connection closed while sending message: {e.code}")
            await self.unregister(client)
            return False
        except WebSocketException as e:
            self.logger.error(f"WebSocket error while sending message: {e}")
            await self.unregister(client)
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error while sending message: {e}")
            await self.unregister(client)
            return False

    async def send_to_clients(self, message: str) -> None:
        if not self.clients:
            self.logger.debug("No clients connected. Message not sent.")
            return

        tasks = [self.safe_send(client, message) for client in self.clients.copy()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        self.logger.info(f"Message sent to {success_count}/{len(tasks)} clients")

    async def heartbeat(self, websocket) -> None:
        try:
            while True:
                try:
                    await websocket.ping()
                    await asyncio.sleep(30)
                except ConnectionClosedError as e:
                    if e.code == 1001:
                        self.logger.info("Client going away gracefully")
                    else:
                        self.logger.warning(f"Connection closed during heartbeat: {e.code}")
                    break
                except Exception as e:
                    self.logger.error(f"Error in heartbeat: {e}")
                    break
        finally:
            await self.unregister(websocket)

    async def handler(self, websocket, path: str) -> None:
        await self.register(websocket)
        try:

            heartbeat_task = asyncio.create_task(self.heartbeat(websocket))
            
            try:
                await websocket.wait_closed()
            except Exception as e:
                self.logger.error(f"Error in connection: {e}")
            
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
                
        except Exception as e:
            self.logger.error(f"Unexpected error in handler: {e}")
        finally:
            await self.unregister(websocket)

    async def start(self, host: str = "0.0.0.0", port: int = 6789) -> None:
        try:
            async with serve(self.handler, host, port):
                self.logger.info(f"WebSocket server started on ws://{host}:{port}")
                await asyncio.Future()
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
            raise
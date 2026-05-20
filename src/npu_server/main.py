import logging
import uvicorn
from npu_server.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
)

if __name__ == "__main__":
    uvicorn.run(
        "npu_server.app:app",
        host=settings.server_host,
        port=settings.server_port,
        log_level=settings.log_level,
        access_log=True,
    )

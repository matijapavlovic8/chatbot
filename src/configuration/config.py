import os
import logging
from dotenv import load_dotenv

load_dotenv()
LOG_FILE = os.getenv("LOG_FILE")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

logging.basicConfig(
    handlers=[
        logging.StreamHandler()
    ],
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def mask_sensitive(value):
    if value and len(value) > 4:
        return value[:2] + "*" * (len(value) - 4) + value[-2:]
    return value

PORT = os.getenv("PORT")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL_OPENAI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_OPENAI")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_OPENAI")
EMBEDDING_SIZE = os.getenv("EMBEDDING_SIZE")
MONGO_DB_URI = os.getenv("MONGO_DB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_DB_USERNAME = os.getenv("MONGO_DB_USERNAME")
MONGO_DB_PASSWORD = os.getenv("MONGO_DB_PASSWORD")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


logging.info("Application Configuration:")
for key in [
    "PORT",
    "OPENAI_MODEL", "OPENAI_API_BASE_URL",
    "EMBEDDING_MODEL", "EMBEDDING_SIZE"]:
    logging.info(f"{key}: {globals()[key]}")

logging.info(f"OPENAI_API_KEY: {mask_sensitive(OPENAI_API_KEY)}")

if DEBUG_MODE:
    logging.debug("Debug mode is enabled. Additional debug logs will be captured.")

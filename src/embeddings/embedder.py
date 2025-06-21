import logging

from langchain_openai import OpenAIEmbeddings

from configuration.config import OPENAI_API_KEY, OPENAI_API_BASE_URL, EMBEDDING_MODEL, DEBUG_MODE

logger = logging.getLogger(__name__)

def get_embedding_function():
    if DEBUG_MODE:
        logger.debug(f"Using embedding model: {EMBEDDING_MODEL}")
        logger.debug(f"OpenAI API Base URL: {OPENAI_API_BASE_URL}")

    return OpenAIEmbeddings(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE_URL, model=EMBEDDING_MODEL)

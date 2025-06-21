from typing import List

from qdrant_client import QdrantClient

from configuration.config import QDRANT_HOST, COLLECTION_NAME, QDRANT_PORT
from embeddings.embedder import get_embedding_function
from langchain.schema import Document


def retrieve_documents(query: str) -> List[Document]:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    embedding_function = get_embedding_function()
    query_vector = embedding_function.embed_query(query)

    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=5,
        score_threshold=0.5,
        with_payload=True,
        with_vectors=False
    )

    return [
        Document(
            page_content=result.payload.get("text", ""),
            metadata={key: value for key, value in result.payload.items() if key != "text"}
        )
        for result in search_results.points
    ]
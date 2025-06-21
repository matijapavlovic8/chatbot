import logging
from datetime import datetime, timezone
from typing import List

from pymongo import MongoClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from configuration.config import (
    DEBUG_MODE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_API_BASE_URL,
    MONGO_DB_NAME, MONGO_DB_USERNAME, MONGO_DB_PASSWORD
)
from retriever.retriever import retrieve_documents

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

mongo_uri = f"mongodb://{MONGO_DB_USERNAME}:{MONGO_DB_PASSWORD}@localhost:27017/chatbot?authSource=chatbot"
mongo_client = MongoClient(mongo_uri)
mongo_db = mongo_client[MONGO_DB_NAME]
conversations = mongo_db.conversations


def append_message(session_id: str, sender: str, message: str, user_id: str):
    """Append a message to the messages array inside the session document."""
    message_doc = {
        "sender": sender,
        "message": message,
        "timestamp": datetime.now(timezone.utc)
    }
    result = conversations.update_one(
        {"session_id": session_id, "user_id": user_id},
        {"$push": {"messages": message_doc}}
    )
    if result.matched_count == 0:
        raise ValueError(f"Session {session_id} not found for user {user_id}")


def fetch_session_history(session_id: str, user_id: str):
    """Retrieve the messages list from the session document."""
    session = conversations.find_one(
        {"session_id": session_id, "user_id": user_id},
        {"_id": 0, "messages": 1}
    )
    if not session or "messages" not in session:
        return []
    return session["messages"]


def format_retrieved_documents(documents: List[Document]) -> str:
    """Format retrieved documents into a readable context string."""
    if not documents:
        return "No relevant documents found."

    formatted_docs = []
    for i, doc in enumerate(documents, 1):
        # Include metadata if available
        metadata_str = ""
        if doc.metadata:
            metadata_items = [f"{k}: {v}" for k, v in doc.metadata.items()]
            metadata_str = f" ({', '.join(metadata_items)})"

        formatted_docs.append(f"Document {i}{metadata_str}:\n{doc.page_content}")

    return "\n\n".join(formatted_docs)


def extract_sources_from_documents(documents: List[Document]) -> List[str]:
    """Extract original_id values from document metadata to use as sources."""
    sources = []
    for doc in documents:
        if doc.metadata and "original_id" in doc.metadata:
            source_id = doc.metadata["original_id"]
            if source_id and source_id not in sources:
                sources.append(str(source_id))
    return sources





PROMPT_TEMPLATE = """You are a helpful assistant. Always answer in Croatian.

Here is relevant information from the knowledge base:
{rag_context}

Here is the previous conversation context:
{conversation_context}

User question:
{question}

Use the relevant information from the knowledge base to provide a helpful and accurate response. If the knowledge base doesn't contain relevant information, rely on your general knowledge but mention that you're not finding specific information in the available documents."""


def build_chat_chain():
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        base_url=OPENAI_API_BASE_URL
    )

    def get_rag_context(inputs):
        """Retrieve relevant documents for the query."""
        try:
            documents = retrieve_documents(inputs["query"])
            inputs["retrieved_documents"] = documents
            return format_retrieved_documents(documents)
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            inputs["retrieved_documents"] = []
            return "Error retrieving relevant documents."

    def get_conversation_context(inputs):
        """Get conversation history."""
        try:
            history = fetch_session_history(inputs["session_id"], inputs["user_id"])
            return "\n".join(
                f"{turn['sender']}: {turn['message']}"
                for turn in history
            )
        except Exception as e:
            logger.error(f"Error fetching conversation history: {e}")
            return ""

    generate_response = (
            RunnableMap({
                "rag_context": get_rag_context,
                "conversation_context": get_conversation_context,
                "question": lambda inputs: inputs["query"]
            })
            | prompt_template
            | llm
            | StrOutputParser()
    )

    return generate_response


def chatbot_query(query: str, session_id: str, user_id: str):
    """
    Enhanced chatbot that uses RAG (Retrieval-Augmented Generation) to provide
    context-aware responses based on relevant documents from Qdrant.
    """
    if DEBUG_MODE:
        logger.debug(f"Processing query: {query}")
        logger.debug(f"Session ID: {session_id}, User ID: {user_id}")

    try:
        append_message(session_id, sender="user", message=query, user_id=user_id)
    except Exception as e:
        logger.error(f"Error appending user message: {e}")
        raise

    retrieved_documents = []

    try:
        retrieved_documents = retrieve_documents(query)

        chain = build_chat_chain()
        inputs = {
            "session_id": session_id,
            "query": query,
            "user_id": user_id
        }

        response = chain.invoke(inputs)

        sources = extract_sources_from_documents(retrieved_documents)
        response_with_sources = response, sources

        if DEBUG_MODE:
            logger.debug(f"Extracted sources: {sources}")

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        response_with_sources = "Error"

    try:
        append_message(session_id, sender="bot", message=response_with_sources, user_id=user_id)
    except Exception as e:
        logger.error(f"Error appending bot message: {e}")

    if DEBUG_MODE:
        logger.debug(f"Generated response: {response_with_sources}")

    return response_with_sources


def get_relevant_documents_for_query(query: str) -> List[Document]:
    """
    Utility function to retrieve relevant documents for a query.
    Useful for debugging or testing the RAG functionality.
    """
    try:
        return retrieve_documents(query)
    except Exception as e:
        logger.error(f"Error retrieving documents for query '{query}': {e}")
        return []
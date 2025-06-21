from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime, timezone
import logging
import uuid
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from configuration.config import (
    MONGO_DB_NAME, MONGO_DB_USERNAME, MONGO_DB_PASSWORD
)

from bot import bot

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB konekcija
mongo_uri = f"mongodb://{MONGO_DB_USERNAME}:{MONGO_DB_PASSWORD}@localhost:27017/chatbot?authSource=chatbot"
mongo_client = MongoClient(mongo_uri)
mongo_db = mongo_client[MONGO_DB_NAME]
conversations = mongo_db.conversations


# Models
class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    query: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

class UserSessionInfo(BaseModel):
    session_id: str
    first_message_timestamp: datetime

class NewSessionResponse(BaseModel):
    session_id: str


# Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response, sources = bot.chatbot_query(
            query=request.query,
            session_id=request.session_id,
            user_id=request.user_id
        )
        return ChatResponse(response=response, sources=sources)
    except Exception as e:
        logging.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/create_new_session", response_model=NewSessionResponse)
async def create_new_session(user_id: str):
    try:
        new_session_id = str(uuid.uuid4())
        conversations.insert_one({
            "session_id": new_session_id,
            "user_id": user_id,
            "sender": "user",
            "message": "Initialization",
            "timestamp": datetime.now(timezone.utc)
        })
        return NewSessionResponse(session_id=new_session_id)
    except Exception as e:
        logging.error(f"Error creating new session for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/user_sessions/{user_id}", response_model=List[UserSessionInfo])
async def get_user_sessions(user_id: str):
    try:
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$session_id",
                "first_message_timestamp": {"$min": "$timestamp"}
            }},
            {"$sort": {"first_message_timestamp": -1}}
        ]
        results = list(conversations.aggregate(pipeline))
        sessions = [
            UserSessionInfo(
                session_id=res["_id"],
                first_message_timestamp=res["first_message_timestamp"]
            )
            for res in results
        ]
        return sessions
    except Exception as e:
        logging.error(f"Error fetching sessions for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


class MessageInfo(BaseModel):
    sender: str
    message: str
    timestamp: datetime

@app.get("/session_messages/{session_id}", response_model=List[MessageInfo])
async def get_session_messages(session_id: str):
    try:
        session_doc = conversations.find_one({"session_id": session_id})
        if not session_doc:
            return []

        messages = [
            MessageInfo(
                sender=msg["sender"],
                message=msg["message"],
                timestamp=msg["timestamp"]
            )
            for msg in session_doc.get("messages", [])
        ]
        return messages
    except Exception as e:
        logging.error(f"Error fetching messages for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)

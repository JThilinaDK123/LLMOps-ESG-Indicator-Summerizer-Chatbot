from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import json
import uuid
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from context import prompt
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

app = FastAPI()

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please add it to your .env file.")

## Initialize the ChatGroq model
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
)

## Memory storage configuration
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
MEMORY_DIR = os.getenv("MEMORY_DIR", "memory") # Using a relative path for local storage

## Initialize S3 client if needed
if USE_S3:
    s3_client = boto3.client("s3")


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class Message(BaseModel):
    role: str
    content: str
    timestamp: str

## Memory Management Functions

def get_memory_path(session_id: str) -> str:
    """Generates the file/key path for a given session ID."""
    return f"{session_id}.json"


def load_conversation(session_id: str) -> List[Dict]:
    """Load conversation history from storage (S3 or local file)."""
    if USE_S3:
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=get_memory_path(session_id))
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise
    else:
        ## Local file storage
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        print(f"Loading conversation from local file: {file_path}")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return []


def save_conversation(session_id: str, messages: List[Dict]):
    """Save conversation history to storage (S3 or local file)."""
    if USE_S3:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=get_memory_path(session_id),
            Body=json.dumps(messages, indent=2),
            ContentType="application/json",
        )
    else:
        ## Local file storage
        os.makedirs(MEMORY_DIR, exist_ok=True)
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        with open(file_path, "w") as f:
            json.dump(messages, f, indent=2) 


def to_lc_messages(conversation: List[Dict], system_prompt: str) -> List:
    """Converts a list of dict messages into a list of LangChain message objects."""
    lc_messages = [SystemMessage(content=system_prompt)]
    for msg in conversation:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages

## API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Healthproct API is running.",
        "memory_enabled": True,
        "storage": "S3" if USE_S3 else "local",
        "model_backend": "Groq/LangChain",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "use_s3": USE_S3}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        ## Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        ## Load conversation history
        conversation = load_conversation(session_id)

        lc_messages = to_lc_messages(conversation[-10:], prompt())

        ## Add current user message
        lc_messages.append(HumanMessage(content=request.message))

        ## Call Groq API via ChatGroq (LangChain's invoke method)
        response = llm.invoke(lc_messages)
        assistant_response = response.content

        ## Update conversation history (using the stored dictionary format)
        current_time = datetime.now().isoformat()
        conversation.append(
            {"role": "user", "content": request.message, "timestamp": current_time}
        )
        conversation.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "timestamp": current_time,
            }
        )

        ## Save conversation
        save_conversation(session_id, conversation)

        return ChatResponse(response=assistant_response, session_id=session_id)

    except Exception as e:
        import traceback
        print(f"Error in chat endpoint: {str(e)}")
        print(traceback.format_exc()) 
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Retrieve conversation history"""
    try:
        conversation = load_conversation(session_id)
        return {"session_id": session_id, "messages": conversation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    if not USE_S3:
        os.makedirs(MEMORY_DIR, exist_ok=True)
        print(f"Local memory directory created/checked: {os.path.abspath(MEMORY_DIR)}")
        
    uvicorn.run(app, host="0.0.0.0", port=8000)
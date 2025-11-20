from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agents.graph import graph

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    username: str
    history: List[Message] = []

class ChatResponse(BaseModel):
    response: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Scheduling Agent Backend is running"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert history to LangChain format
        langchain_messages = []
        for msg in request.history:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
            # System messages can be handled if needed, but usually defined in the graph
        
        # Add the current new message
        langchain_messages.append(HumanMessage(content=request.message))
        
        # Prepare state
        initial_state = {
            "messages": langchain_messages,
            "username": request.username
        }
        
        # Invoke the agent
        # We use ainvoke for async execution
        result = await graph.ainvoke(initial_state)
        
        # Get the last message (the agent's response)
        last_message = result["messages"][-1]
        response_content = last_message.content
        
        return ChatResponse(response=response_content)
            
    except Exception as e:
        print(f"Error in chat_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

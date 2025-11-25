from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agents.graph import graph
from agents.resched import resched_graph

# Initialize Supabase Client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

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
    participants: List[str] = []
    priority_participants: List[str] = []
    # Previous agent state (optional, for state persistence)
    previous_state: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    # Return agent state for next turn
    agent_state: Optional[Dict[str, Any]] = None

class RescheduleRequest(BaseModel):
    proposal_id: str
    feedback: str
    username: str

class FinalizeRequest(BaseModel):
    proposal_id: str

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
        
        # Prepare state - use previous state if provided, otherwise initialize fresh
        if request.previous_state:
            # Preserve state from previous turn, but update messages and reset some fields
            initial_state = {
                "messages": langchain_messages,
                "organizer_id": request.username,
                "participant_ids": request.participants,
                "priority_participant_ids": request.priority_participants,
                # Preserve these from previous turn (agent will decide whether to use them)
                "extracted_info": request.previous_state.get("extracted_info"),
                "all_calendars": request.previous_state.get("all_calendars", {}),
                "all_working_hours": request.previous_state.get("all_working_hours", {}),
                "candidate_slots": request.previous_state.get("candidate_slots", []),
                "proposed_slots": request.previous_state.get("proposed_slots", []),
                "proposed_slot": request.previous_state.get("proposed_slot"),
                "alternatives": request.previous_state.get("alternatives"),
                # Reset confirmation state for new turn
                "proposal_id": None,
                "confirmation_status": None,
                "debug_info": []
            }
        else:
            # Fresh conversation - initialize with empty state
            initial_state = {
                "messages": langchain_messages,
                "organizer_id": request.username,
                "participant_ids": request.participants,
                "priority_participant_ids": request.priority_participants,
                "extracted_info": None,
                "all_calendars": {},
                "all_working_hours": {},
                "candidate_slots": [],
                "proposed_slots": [],
                "proposed_slot": None,
                "alternatives": None,
                "proposal_id": None,
                "confirmation_status": None,
                "debug_info": []
            }
        
        # Invoke the agent
        # We use ainvoke for async execution
        result = await graph.ainvoke(initial_state)
        
        # Get the last message (the agent's response)
        last_message = result["messages"][-1]
        response_content = last_message.content
        
        # Prepare state to return (exclude messages to avoid duplication)
        state_to_return = {
            "extracted_info": result.get("extracted_info"),
            "all_calendars": result.get("all_calendars"),
            "all_working_hours": result.get("all_working_hours"),
            "candidate_slots": result.get("candidate_slots"),
            "proposed_slots": result.get("proposed_slots"),
            "proposed_slot": result.get("proposed_slot"),
            "alternatives": result.get("alternatives"),
        }
        
        return ChatResponse(
            response=response_content,
            agent_state=state_to_return
        )
            
    except Exception as e:
        print(f"Error in chat_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reschedule")
async def reschedule_endpoint(request: RescheduleRequest):
    try:
        initial_state = {
            "proposal_id": request.proposal_id,
            "feedback": request.feedback,
            "feedback_user": request.username,
            "messages": [],
            "debug_info": []
        }
        
        # Invoke the rescheduling graph
        result = await resched_graph.ainvoke(initial_state)
        
        return {
            "status": "success",
            "message": "Rescheduling process completed",
            "new_slot": result.get("proposed_slot"),
            "debug_info": result.get("debug_info")
        }
    except Exception as e:
        print(f"Error in reschedule_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/finalize_meeting")
async def finalize_meeting_endpoint(request: FinalizeRequest):
    try:
        # 1. Fetch the proposal
        proposal_response = supabase.table("meeting_proposals").select("*").eq("proposal_id", request.proposal_id).single().execute()
        proposal = proposal_response.data
        
        if not proposal:
            raise HTTPException(status_code=404, detail="Proposal not found")
            
        # 2. Update status to finalized (not confirmed, to match check constraint)
        # Constraint allows: 'pending', 'accepted', 'rejected', 'finalized', 'cancelled'
        supabase.table("meeting_proposals").update({"status": "finalized"}).eq("proposal_id", request.proposal_id).execute()
        
        # 3. Fetch participant responses to see who accepted
        responses_result = supabase.table("participant_responses").select("participant_id, response").eq("proposal_id", request.proposal_id).execute()
        
        # Identify who should get the calendar event
        # Organizer + Participants who accepted (or maybe everyone? User said "not to those who rejected")
        attendees = [proposal["organizer_id"]]
        
        for resp in responses_result.data:
            if resp["response"] == "accepted":
                attendees.append(resp["participant_id"])
        
        # Deduplicate just in case
        attendees = list(set(attendees))
        
        # 4. Insert into meetings table for each attendee
        # Table structure: meeting_id (auto), meeting_name, user, start_meeting, end_meeting
        
        meetings_to_insert = []
        for user in attendees:
            meetings_to_insert.append({
                "meeting_name": proposal["meeting_title"] or "Untitled Meeting",
                "user": user,
                "start_meeting": proposal["proposed_start"],
                "end_meeting": proposal["proposed_end"]
            })
            
        if meetings_to_insert:
            supabase.table("meetings").insert(meetings_to_insert).execute()
            
        # 5. Optional: Cleanup (User requested clearing out proposal and responses)
        # Delete participant responses first (due to FK)
        supabase.table("participant_responses").delete().eq("proposal_id", request.proposal_id).execute()
        # Delete proposal
        supabase.table("meeting_proposals").delete().eq("proposal_id", request.proposal_id).execute()
            
        return {
            "status": "success",
            "message": "Meeting finalized and added to calendars. Proposal cleanup complete.",
            "attendees": attendees
        }

    except Exception as e:
        print(f"Error in finalize_meeting_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

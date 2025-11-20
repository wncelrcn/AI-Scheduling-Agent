from typing import Annotated, TypedDict, List, Optional, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from supabase import create_client, Client
import os

# Initialize Supabase Client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Define Data Models
class MeetingParameters(BaseModel):
    date: Optional[str] = Field(None, description="Date of the meeting (e.g., '2023-10-27', 'tomorrow', 'next Friday')")
    time: Optional[str] = Field(None, description="Time of the meeting (e.g., '10:00 AM', '14:00')")
    duration: Optional[str] = Field(None, description="Duration of the meeting (e.g., '30 minutes', '1 hour')")
    participants: Optional[List[str]] = Field(None, description="List of participants to invite")
    # topic: Optional[str] = Field(None, description="Topic or subject of the meeting")
    # location: Optional[str] = Field(None, description="Location or platform (e.g., 'Zoom', 'Office')")

class MeetingDetails(BaseModel):
    """Extracts meeting details from user input."""
    intent: str = Field(description="The user's primary intent, e.g., 'schedule_meeting', 'query_availability', 'cancel_meeting', 'chat'.")
    parameters: MeetingParameters = Field(default_factory=MeetingParameters, description="Structured parameters extracted from the conversation.")
    constraints: List[str] = Field(default_factory=list, description="Any constraints mentioned by the user, e.g., 'no Fridays', 'after 2 PM'.")
    missing_info: List[str] = Field(default_factory=list, description="List of information needed to fulfill the request but currently missing.")

# Define the state of our agent
class AgentState(TypedDict):
    # The 'add_messages' annotation ensures that new messages are appended to the existing list
    messages: Annotated[List[BaseMessage], add_messages]
    username: str
    participants: List[str]
    extracted_info: Optional[dict] # Store the parsed JSON here
    participant_availability: Optional[dict] # Store fetched availability from Supabase

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Define the Parser Node
def parse_input(state: AgentState):
    messages = state['messages']
    username = state.get('username', 'User')
    participants = state.get('participants', [])
    
    # Define a parser specific system prompt
    parser_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent scheduling assistant parser. 
        Your job is to extract meeting details from the conversation.
        User: {username}
        Participants already selected in UI: {participants}
        
        Analyze the conversation and extract:
        1. Intent: What does the user want? (e.g., schedule_meeting, query_availability)
        2. Parameters: Extract specific values into the structured format. If participants are listed above, include them.
        3. Constraints: Any preferences or blockers.
        4. Missing Info: What is strictly necessary for the intent but missing?
        
        If the user is just saying hello or chatting, set intent to 'chat'.
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Configure LLM for structured output
    structured_llm = llm.with_structured_output(MeetingDetails)
    chain = parser_prompt | structured_llm
    
    # Invoke the chain
    try:
        result = chain.invoke({
            "username": username, 
            "messages": messages,
            "participants": str(participants)
        })
        return {"extracted_info": result.model_dump()}
    except Exception as e:
        # Fallback if parsing fails
        return {"extracted_info": {"error": str(e), "intent": "error", "parameters": {}, "constraints": [], "missing_info": []}}

# Define the Availability Fetcher Node
def fetch_availability(state: AgentState):
    participants = state.get('participants', [])
    
    # If no participants, skip
    if not participants:
        return {"participant_availability": {}}
    
    try:
        # Query Supabase for all participants
        response = supabase.table("users").select("name, work_start, work_end, working_days").in_("name", participants).execute()
        users_data = response.data
        
        availability_map = {}
        for user in users_data:
            name = user.get("name")
            # Parse work_start/end which might be timetz strings
            start = user.get("work_start", "09:00:00+00").split("+")[0][:5] # Simplistic parsing
            end = user.get("work_end", "17:00:00+00").split("+")[0][:5]
            days = user.get("working_days", [])
            
            availability_map[name] = {
                "working_hours": f"{start} - {end}",
                "working_days": ", ".join(days) if days else "Not specified"
            }
            
        return {"participant_availability": availability_map}
        
    except Exception as e:
        print(f"Error fetching availability: {e}")
        return {"participant_availability": {"error": "Could not fetch availability"}}

# Define the Responder Node
def respond(state: AgentState):
    messages = state['messages']
    username = state.get('username', 'User')
    extracted_info = state.get('extracted_info', {})
    participant_availability = state.get('participant_availability', {})
    
    # Prepare the prompt for the responder
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent scheduling assistant.
        User: {username}
        
        You have parsed the user's input:
        {extracted_info}
        
        Here is the availability for the selected participants (from database):
        {participant_availability}
        
        Your goal is to confirm what you understood and ask for missing information if any.
        
        Rules:
        1. If 'missing_info' is not empty, ask clarifying questions for those specific items.
        2. If 'intent' is 'schedule_meeting' and all info seems present:
           - Acknowledge the participants and their general availability constraints.
           - Summarize the meeting details for confirmation.
           - If the user proposed a time that conflicts with the database availability, politely warn them (e.g. "Note that John doesn't work on Fridays").
        3. If 'intent' is 'chat', respond naturally.
        4. Be helpful and polite.
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = response_prompt | llm
    response = chain.invoke({
        "username": username, 
        "messages": messages, 
        "extracted_info": str(extracted_info),
        "participant_availability": str(participant_availability)
    })
    
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("parse_input", parse_input)
workflow.add_node("fetch_availability", fetch_availability)
workflow.add_node("respond", respond)

# Set entry point
workflow.set_entry_point("parse_input")

# Add edges
workflow.add_edge("parse_input", "fetch_availability")
workflow.add_edge("fetch_availability", "respond")
workflow.add_edge("respond", END)

# Compile the graph
graph = workflow.compile()

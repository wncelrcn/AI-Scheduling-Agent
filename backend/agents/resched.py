from typing import Annotated, TypedDict, List, Optional, Any, Dict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from supabase import create_client, Client
import os
from datetime import datetime, timedelta
import pytz
import logging
import json

# Reuse functions from the main graph
from agents.graph import (
    fetch_calendars_node,
    fetch_working_hours_node,
    find_slots_node,
    select_best_slot,
    rank_slots,
    suggest_next_days,
    suggest_time_adjustments,
    suggest_partial_attendance,
    suggest_duration_flexibility,
    parse_time_to_object,
    TIMEZONE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase Client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Define State (compatible with AgentState but with rescheduling specific fields)
class ReschedState(TypedDict):
    # Standard fields required by reused nodes
    messages: Annotated[List[BaseMessage], add_messages]
    organizer_id: str
    participant_ids: List[str]
    extracted_info: Optional[dict]
    all_calendars: Dict[str, List[dict]]
    all_working_hours: Dict[str, dict]
    candidate_slots: List[dict]
    proposed_slots: List[dict]
    proposed_slot: Optional[dict]
    alternatives: Optional[dict]
    debug_info: List[str]
    
    # Rescheduling specific fields
    proposal_id: str
    feedback: str
    feedback_user: str
    reschedule_status: str  # Track status (suggested, confirmed, needs_feedback)
    edge_case_type: Optional[str]  # Track which edge case was detected

# Node: Fetch Proposal Details
async def fetch_proposal_node(state: ReschedState):
    """
    Fetch the existing proposal to get context (organizer, participants, original request).
    """
    proposal_id = state.get("proposal_id")
    debug_info = state.get("debug_info", [])
    
    debug_info.append(f"Fetching proposal {proposal_id}")
    
    try:
        # Fetch proposal details
        response = supabase.table("meeting_proposals").select("*").eq("proposal_id", proposal_id).single().execute()
        proposal = response.data
        
        if not proposal:
            raise ValueError(f"Proposal {proposal_id} not found")
            
        organizer_id = proposal["organizer_id"]
        # participant_ids is an array in DB
        participant_ids = proposal.get("participant_ids", [])
        
        # Initial extracted info from proposal
        # We need to set up 'parameters' so find_slots_node works
        
        # Calculate duration from proposed start/end
        start_dt = datetime.fromisoformat(proposal["proposed_start"])
        end_dt = datetime.fromisoformat(proposal["proposed_end"])
        duration_minutes = int((end_dt - start_dt).total_seconds() / 60)
        
        initial_params = {
            "title": proposal.get("meeting_title"),
            "duration_minutes": duration_minutes,
            # We don't set date/time yet, as we want to reschedule
            # But we might want to keep the original date as a reference or default
            "original_date": start_dt.strftime("%Y-%m-%d"),
            "original_start": start_dt.strftime("%H:%M"),
        }
        
        extracted_info = {
            "intent": "reschedule",
            "parameters": initial_params,
            "constraints": [],
            "missing_info": []
        }
        
        debug_info.append(f"Loaded proposal: {proposal.get('meeting_title')} by {organizer_id}")
        debug_info.append(f"Participants: {participant_ids}")
        
        return {
            "organizer_id": organizer_id,
            "participant_ids": participant_ids,
            "extracted_info": extracted_info,
            "debug_info": debug_info
        }
        
    except Exception as e:
        logger.error(f"Error fetching proposal: {e}")
        debug_info.append(f"ERROR fetching proposal: {str(e)}")
        return {"debug_info": debug_info}

# Node: Process Feedback (LLM)
async def process_feedback_node(state: ReschedState):
    """
    Analyze the rejection feedback to determine new constraints.
    Updates extracted_info with new date/time preferences or exclusions.
    """
    feedback = state.get("feedback")
    feedback_user = state.get("feedback_user")
    extracted_info = state.get("extracted_info", {})
    params = extracted_info.get("parameters", {})
    debug_info = state.get("debug_info", [])
    
    debug_info.append(f"Processing feedback from {feedback_user}: '{feedback}'")
    
    current_time = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M %Z")
    
    prompt = f"""You are an intelligent scheduling assistant helping to reschedule a meeting.
    
    Current Context:
    - Original Date: {params.get('original_date')}
    - Original Time: {params.get('original_start')}
    - Duration: {params.get('duration_minutes')} minutes
    - Current Time: {current_time}
    
    Rejection Feedback from {feedback_user}: "{feedback}"
    
    Your goal is to extract new scheduling parameters based on this feedback.
    
    Output JSON with the following fields:
    - date: (string, YYYY-MM-DD) specific date if mentioned, or null
    - start_time: (string, HH:MM) specific start time if mentioned, or null
    - end_time: (string, HH:MM) specific end time if mentioned, or null
    - duration_minutes: (int) updated duration if mentioned, else keep {params.get('duration_minutes')}
    - search_strategy: (string) "next_days" (default), "specific_date", "flexible"
    
    Rules:
    1. If they say "I'm busy", "Not free", etc., implies we should look for OTHER times/days.
    2. If they suggest a specific time (e.g. "How about Tuesday?"), set that as the target date.
    3. If they don't specify a time, default to searching next available days.
    """
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        # Basic JSON parsing (assuming LLM follows instruction, usually robust with Gemini/OpenAI)
        # Ideally use structured output or json mode
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        result = json.loads(content)
        
        # Update parameters
        new_params = params.copy()
        
        if result.get("date"):
            new_params["date"] = result["date"]
            debug_info.append(f"New date preference: {result['date']}")
        else:
            # If currently has a date but feedback implies "not this time", remove the date to search broader
            # Unless the feedback was just about time on that day
            if "date" in new_params and result.get("search_strategy") != "specific_date":
                del new_params["date"]
        
        if result.get("start_time"):
            new_params["start_time"] = result["start_time"]
        else:
            # Clear specific start time if we are rescheduling broadly
             if "start_time" in new_params and result.get("search_strategy") != "specific_date":
                 del new_params["start_time"]
                 
        if result.get("duration_minutes"):
            new_params["duration_minutes"] = result["duration_minutes"]
            
        extracted_info["parameters"] = new_params
        debug_info.append(f"Updated parameters: {new_params}")
        
        return {
            "extracted_info": extracted_info,
            "debug_info": debug_info
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        debug_info.append(f"ERROR processing feedback: {str(e)}")
        return {"debug_info": debug_info}

# Node: Suggest to Organizer (replaces update_proposal_node)
async def suggest_to_organizer_node(state: ReschedState):
    """
    Store suggested slots for organizer approval instead of automatically updating.
    Sets status to 'awaiting_organizer_approval'.
    """
    proposed_slots = state.get("proposed_slots", [])
    alternatives = state.get("alternatives")
    proposal_id = state.get("proposal_id")
    debug_info = state.get("debug_info", [])
    edge_case_type = state.get("edge_case_type")
    
    try:
        # Fetch current iteration count
        current_proposal = supabase.table("meeting_proposals").select("iteration_count").eq("proposal_id", proposal_id).single().execute()
        current_count = current_proposal.data.get("iteration_count", 0)
        
        if not proposed_slots and not alternatives:
            # No slots and no alternatives - critical failure
            debug_info.append("No valid slots or alternatives found to reschedule.")
            update_data = {
                "status": "awaiting_organizer_approval",
                "iteration_count": current_count + 1,
                "suggested_slots": json.dumps({"error": "No available slots found", "edge_case": edge_case_type}),
                "reasoning": "Rescheduling failed - no available slots. Manual intervention needed."
            }
        elif not proposed_slots and alternatives:
            # No perfect slots but have alternatives
            debug_info.append("No perfect slots, storing alternatives for organizer review")
            update_data = {
                "status": "awaiting_organizer_approval",
                "iteration_count": current_count + 1,
                "suggested_slots": json.dumps({
                    "slots": [],
                    "alternatives": alternatives,
                    "edge_case": edge_case_type
                }),
                "reasoning": "No perfect match found. Alternatives suggested for organizer approval."
            }
        else:
            # Have valid slots - store top 3 for organizer to choose
            debug_info.append(f"Storing {len(proposed_slots)} suggested slots for organizer approval")
            slots_to_store = []
            for i, slot in enumerate(proposed_slots[:3], 1):
                slots_to_store.append({
                    "index": i,
                    "start": slot["start"],
                    "end": slot["end"],
                    "score": slot.get("ranking_score", 0),
                    "reasons": slot.get("ranking_reasons", []),
                    "available_participants": slot.get("available_participants", [])
                })
            
            update_data = {
                "status": "awaiting_organizer_approval",
                "iteration_count": current_count + 1,
                "suggested_slots": json.dumps({
                    "slots": slots_to_store,
                    "alternatives": alternatives if alternatives else None,
                    "edge_case": edge_case_type
                }),
                "reasoning": f"Rescheduling suggestions ready. Top slot: {', '.join(proposed_slots[0].get('ranking_reasons', []))}"
            }
        
        # Update proposal with suggestions
        supabase.table("meeting_proposals").update(update_data).eq("proposal_id", proposal_id).execute()
        debug_info.append(f"Updated proposal {proposal_id} status to 'awaiting_organizer_approval'")
        
        # Keep participant responses as-is (they'll be reset when organizer confirms)
        
        return {
            "debug_info": debug_info,
            "reschedule_status": "suggested"
        }
        
    except Exception as e:
        logger.error(f"Error storing suggestions: {e}")
        debug_info.append(f"ERROR storing suggestions: {str(e)}")
        return {"debug_info": debug_info, "reschedule_status": "error"}

# Node: Generate alternatives when no perfect slots found
def generate_alternatives(state: ReschedState) -> dict:
    """
    Generate comprehensive alternatives when no perfect slots are found.
    Returns dict with different types of suggestions.
    """
    alternatives = {
        "next_days": suggest_next_days(state, num_days=3),
        "time_adjustments": suggest_time_adjustments(state),
        "partial_attendance": suggest_partial_attendance(state),
        "duration_flexibility": suggest_duration_flexibility(state)
    }
    
    return alternatives

# Node: Respond with edge case handling
async def respond_node(state: ReschedState):
    """
    Prepare response message for organizer with suggestions and edge case info.
    This doesn't send messages but prepares status information.
    """
    debug_info = state.get("debug_info", [])
    proposed_slots = state.get("proposed_slots", [])
    alternatives = state.get("alternatives")
    edge_case_type = state.get("edge_case_type")
    
    # Detect edge cases from debug_info
    past_time_detected = any("PAST TIME" in log for log in debug_info)
    before_hours_detected = any("BEFORE WORKING HOURS" in log for log in debug_info)
    after_hours_detected = any("AFTER WORKING HOURS" in log for log in debug_info)
    non_working_day_detected = any("NON-WORKING DAY" in log for log in debug_info)
    
    # Set edge case type if detected
    updated_edge_case = edge_case_type
    if past_time_detected:
        updated_edge_case = "past_time"
    elif before_hours_detected:
        updated_edge_case = "before_working_hours"
    elif after_hours_detected:
        updated_edge_case = "after_working_hours"
    elif non_working_day_detected:
        updated_edge_case = "non_working_day"
    elif not proposed_slots and not alternatives:
        updated_edge_case = "no_availability"
    
    debug_info.append(f"Response prepared. Edge case: {updated_edge_case}, Slots: {len(proposed_slots)}, Has alternatives: {alternatives is not None}")
    
    return {
        "debug_info": debug_info,
        "edge_case_type": updated_edge_case
    }

# Define the Rescheduling Graph
workflow = StateGraph(ReschedState)

workflow.add_node("fetch_proposal", fetch_proposal_node)
workflow.add_node("process_feedback", process_feedback_node)
workflow.add_node("fetch_calendars", fetch_calendars_node)
workflow.add_node("fetch_working_hours", fetch_working_hours_node)
workflow.add_node("find_slots", find_slots_node)
workflow.add_node("select_best_slot", select_best_slot)
workflow.add_node("suggest_to_organizer", suggest_to_organizer_node)
workflow.add_node("respond", respond_node)

# Set entry point
workflow.set_entry_point("fetch_proposal")

# Define flow: fetch -> process -> calendars -> hours -> slots -> select -> respond -> suggest -> END
workflow.add_edge("fetch_proposal", "process_feedback")
workflow.add_edge("process_feedback", "fetch_calendars")
workflow.add_edge("fetch_calendars", "fetch_working_hours")
workflow.add_edge("fetch_working_hours", "find_slots")
workflow.add_edge("find_slots", "select_best_slot")
workflow.add_edge("select_best_slot", "respond")
workflow.add_edge("respond", "suggest_to_organizer")
workflow.add_edge("suggest_to_organizer", END)

# Compile
resched_graph = workflow.compile()

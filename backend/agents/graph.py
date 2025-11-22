from typing import Annotated, TypedDict, List, Optional, Any, Dict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from supabase import create_client, Client
import os
from datetime import datetime, timedelta, time
import pytz
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Constants
TIMEZONE = pytz.timezone('Asia/Singapore')
SLOT_INCREMENT_MINUTES = 30
DEFAULT_SEARCH_DAYS = 5
DEFAULT_DURATION_MINUTES = 30
MAX_SLOTS_RETURNED = 10

# Initialize Supabase Client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Utility Functions
def parse_timetz(t_str: Optional[str]) -> Optional[str]:
    """Parse database timetz string to HH:MM format."""
    if not t_str:
        return None
    # Split "+00" or "-05" etc and extract HH:MM
    base_time = t_str.split("+")[0].split("-")[0]
    return base_time[:5]  # HH:MM

def parse_time_to_object(time_str: str) -> Optional[time]:
    """Parse HH:MM string to time object."""
    try:
        return datetime.strptime(time_str, "%H:%M").time()
    except (ValueError, TypeError):
        return None

def calculate_duration_minutes(start_str: str, end_str: str) -> Optional[int]:
    """Calculate duration in minutes from start and end time strings (HH:MM format)."""
    try:
        start = datetime.strptime(start_str, "%H:%M")
        end = datetime.strptime(end_str, "%H:%M")
        duration = (end - start).total_seconds() / 60
        return int(duration) if duration > 0 else None
    except (ValueError, TypeError):
        return None

def parse_to_datetime(timestamp_str: str) -> datetime:
    """
    Parse Supabase timestamptz string to timezone-aware datetime object in Asia/Singapore timezone.
    Handles both ISO format with timezone info and without.
    Note: Supabase stores in UTC (+00) but we convert to Asia/Singapore (+08).
    """
    try:
        # Handle 'Z' suffix (Zulu time = UTC)
        if timestamp_str.endswith('Z'):
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        else:
            dt = datetime.fromisoformat(timestamp_str)
        
        # If naive datetime, assume UTC
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        
        # Convert to Asia/Singapore timezone
        return dt.astimezone(TIMEZONE)
    except (ValueError, AttributeError) as e:
        logger.error(f"Failed to parse timestamp '{timestamp_str}': {e}")
        raise

def has_conflict(existing_meeting: dict, proposed_slot: dict) -> bool:
    """
    Check if an existing meeting conflicts with a proposed slot.
    Both should have 'start' and 'end' keys with datetime objects.
    Uses proper overlap detection: max(start1, start2) < min(end1, end2)
    """
    overlap_start = max(proposed_slot["start"], existing_meeting["start"])
    overlap_end = min(proposed_slot["end"], existing_meeting["end"])
    return overlap_start < overlap_end

def rank_slots(candidate_slots: List[dict], extracted_info: dict) -> List[dict]:
    """
    Rank candidate slots using multi-factor scoring.
    
    Scoring priorities:
    1. Preferred Time Match (highest priority)
    2. Time of Day Quality
    3. Day Proximity
    4. Full Attendance
    
    Returns slots sorted by score (highest first).
    """
    params = extracted_info.get('parameters', {})
    pref_start_time = parse_time_to_object(params.get('start_time')) if params.get('start_time') else None
    pref_date_str = params.get('date')
    pref_date = datetime.strptime(pref_date_str, "%Y-%m-%d").date() if pref_date_str else None
    
    ranked_slots = []
    
    for slot in candidate_slots:
        score = 0
        reasons = []
        
        # Parse slot datetime
        slot_start = datetime.fromisoformat(slot['start'])
        slot_end = datetime.fromisoformat(slot['end'])
        slot_start_time = slot_start.time()
        slot_date = slot_start.date()
        
        # 1. Preferred Time Match (highest priority: 0-100 points)
        if pref_start_time:
            # Calculate time difference in hours
            pref_datetime = datetime.combine(slot_date, pref_start_time)
            
            # Ensure pref_datetime is timezone-aware to match slot_start
            if pref_datetime.tzinfo is None:
                pref_datetime = TIMEZONE.localize(pref_datetime)
                
            time_diff_hours = abs((slot_start - pref_datetime).total_seconds() / 3600)
            
            if time_diff_hours == 0:
                score += 100
                reasons.append("Exact match to preferred time")
            elif time_diff_hours <= 1:
                score += 50
                reasons.append("Within 1 hour of preferred time")
            elif time_diff_hours <= 2:
                score += 25
                reasons.append("Within 2 hours of preferred time")
        
        # 2. Time of Day Quality (0-20 points)
        hour = slot_start_time.hour
        if 9 <= hour < 11:
            score += 20
            reasons.append("Morning slot")
        elif 11 <= hour < 13:
            score += 10
            reasons.append("Late morning")
        elif 14 <= hour < 16:
            score += 15
            reasons.append("Afternoon slot")
        elif 16 <= hour < 17:
            score += 5
            reasons.append("Late afternoon")
        
        # 3. Day Proximity (0-30 points)
        if pref_date:
            day_diff = abs((slot_date - pref_date).days)
            if day_diff == 0:
                score += 30
                reasons.append("Same day as requested")
            elif day_diff == 1:
                score += 20
                reasons.append("Next day")
            elif day_diff <= 3:
                score += 10
                reasons.append(f"{day_diff} days from requested")
        
        # 4. Full Attendance (0-50 points)
        available_participants = slot.get('available_participants', [])
        total_expected = slot.get('score', 3)  # Using the existing score field as indicator
        if total_expected == 3:  # Full attendance
            score += 50
            reasons.append("All participants available")
        
        # Add scored slot
        ranked_slot = slot.copy()
        ranked_slot['ranking_score'] = score
        ranked_slot['ranking_reasons'] = reasons
        ranked_slots.append(ranked_slot)
    
    # Sort by score (highest first)
    ranked_slots.sort(key=lambda x: x['ranking_score'], reverse=True)
    
    return ranked_slots

# Define Data Models
class Constraint(BaseModel):
    type: str = Field(description="Type of constraint: 'unavailable' or 'preference'")
    start: Optional[str] = Field(None, description="ISO start time string of the constraint if applicable")
    end: Optional[str] = Field(None, description="ISO end time string of the constraint if applicable")
    day: Optional[str] = Field(None, description="Day of week if recurring, e.g., 'Monday'")
    user_id: Optional[str] = Field(None, description="The user this constraint applies to")

class MeetingParameters(BaseModel):
    date: Optional[str] = Field(None, description="Date of the meeting in YYYY-MM-DD format. Example: '2023-10-27'.")
    start_time: Optional[str] = Field(None, description="Start time of the meeting preference in HH:MM (24h) format. Example: '14:00'.")
    end_time: Optional[str] = Field(None, description="End time of the meeting preference in HH:MM (24h) format. Example: '15:00'.")
    duration_minutes: Optional[int] = Field(30, description="Duration of the meeting in minutes. Example: 60 for 1 hour.")
    title: Optional[str] = Field(None, description="Title of the meeting (e.g., 'Meeting with John')")
    participants: Optional[List[str]] = Field(None, description="List of participants to invite")

class MeetingDetails(BaseModel):
    """Extracts meeting details from user input."""
    intent: str = Field(description="The user's primary intent, e.g., 'schedule_meeting', 'query_availability', 'cancel_meeting', 'confirm_schedule', 'chat'.")
    parameters: MeetingParameters = Field(default_factory=MeetingParameters, description="Structured parameters extracted from the conversation.")
    constraints: List[Constraint] = Field(default_factory=list, description="Any constraints mentioned by the user.")
    missing_info: List[str] = Field(default_factory=list, description="List of information needed to fulfill the request but currently missing.")
    slot_selection: Optional[int] = Field(None, description="Which slot number the user selected (1, 2, 3, etc.) for confirmation")

# Define the state of our agent
class AgentState(TypedDict):
    # The 'add_messages' annotation ensures that new messages are appended to the existing list
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Core identifiers (names are the primary keys in users table)
    organizer_id: str  # The username (same as users.name)
    participant_ids: List[str]  # Selected participants (names, same as users.name)
    
    # Extracted from user input
    extracted_info: Optional[dict]  # Parsed constraints (duration, time, date)
    
    # Fetched data (separate concerns)
    all_calendars: Dict[str, List[dict]]  # name -> list of meeting events (datetime objects)
    all_working_hours: Dict[str, dict]  # name -> {working_days: [], start: "09:00", end: "17:00"}
    
    # Computed results
    candidate_slots: List[dict]  # All valid slots from find_slots_node
    proposed_slots: List[dict]  # Top 3 ranked slots from select_best_slot
    proposed_slot: Optional[dict]  # Legacy field, kept for compatibility
    alternatives: Optional[dict]  # Alternative suggestions when no slots found
    
    # Meeting proposal tracking
    proposal_id: Optional[str]  # UUID of created proposal
    confirmation_status: Optional[str]  # Track confirmation state
    
    # Debug tracking
    debug_info: List[str]

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Define the Parser Node
async def parse_input(state: AgentState):
    """Parse user input and extract meeting details with timezone awareness."""
    messages = state['messages']
    organizer_id = state.get('organizer_id', 'User')
    participant_ids = state.get('participant_ids', [])
    
    # Initialize debug_info if not present
    debug_info = state.get('debug_info', [])
    debug_info.append(f"Parsing input for organizer: {organizer_id}, participants: {participant_ids}")
        
    # Define a parser specific system prompt with timezone context
    current_time = datetime.now(TIMEZONE)
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M %Z")
    
    parser_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent scheduling assistant parser that handles both new requests and follow-up conversations.
        
        Current Time: {current_time} (Singapore/Manila Time, UTC+8)
        Organizer: {organizer}
        Participants already selected: {participants}
        
        IMPORTANT: This is a conversation. Look at the FULL chat history to understand context.
        
        Analyze the conversation and extract:
        
        1. **Intent Detection:**
           - schedule_meeting: User wants to schedule a new meeting or is revising a request
           - query_availability: User is asking about availability
           - confirm_schedule: User is confirming/booking a previously suggested slot
           - chat: Just chatting or saying hello
        
        2. **Confirmation Detection (HIGH PRIORITY):**
           If the assistant has ALREADY suggested time slots in previous messages, check if user is confirming:
           - Explicit slot numbers: "book slot 1", "confirm slot 2", "I'll take slot 3"
           - Natural confirmation: "I'll take the first one", "book the Tuesday meeting", "yes, let's do it", "perfect", "sounds good"
           - If confirming, set intent to 'confirm_schedule' and extract slot_selection number (1, 2, or 3)
           - Look at previous assistant messages to see if slots were offered
        
        3. **Follow-up Detection:**
           Check if this is a follow-up to a previous scheduling request:
           - Alternative selection: "try Tuesday instead", "check next Monday", "what about afternoon"
           - Constraint revision: "make it 30 minutes", "change to 2pm", "any time works"
           - If it's a follow-up, look at what parameters were discussed before and update ONLY what changed
        
        4. **Parameter Extraction:**
           - **Slot Selection:** For confirm_schedule intent, extract the slot number (1, 2, or 3)
             * "book slot 1" → slot_selection: 1
             * "I'll take the first one" → slot_selection: 1
             * "let's go with the second option" → slot_selection: 2
             * "yes" or "sounds good" (without specific number) → slot_selection: 1 (default to first)
           - **Date:** Convert relative dates (today, tomorrow, Monday, next week) to YYYY-MM-DD
           - **Time:** Convert times (2pm, 14:00, afternoon) to HH:MM (24h format)
           - **Duration:** Extract from phrases:
             * "2 hour meeting" → duration_minutes: 120
             * "30 minute meeting" → duration_minutes: 30
             * "1 hour" → duration_minutes: 60
             * "half hour" → duration_minutes: 30
             * If start_time AND end_time given, calculate duration
             * Default: 30 minutes if not specified
           - **Title:** Extract meeting title/purpose if mentioned
        
        5. **Smart Context Awareness:**
           - If user says "try Tuesday" and duration/time were mentioned before, keep those
           - If user says "make it 2pm" and date was mentioned before, keep that date
           - If user picks "slot 1" from suggestions, set intent to confirm_schedule
           - Preserve unchanged information from earlier in the conversation
        
        6. **Missing Info:**
           What is strictly necessary for the intent but still missing?
           - For confirm_schedule: No missing info needed (slot confirmation is sufficient)
        
        Examples:
        - "Book slot 1" → intent: confirm_schedule, slot_selection: 1
        - "I'll take the first one" → intent: confirm_schedule, slot_selection: 1
        - "Yes, sounds good" (after slots suggested) → intent: confirm_schedule, slot_selection: 1
        - "Try Tuesday instead" → intent: schedule_meeting, update date to next Tuesday, keep other params
        - "Make it 30 minutes" → intent: schedule_meeting, update duration only
        - "What about 2pm?" → intent: schedule_meeting, update start_time only
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Configure LLM for structured output
    structured_llm = llm.with_structured_output(MeetingDetails)
    chain = parser_prompt | structured_llm
    
    # Invoke the chain
    try:
        result = await chain.ainvoke({
            "organizer": organizer_id, 
            "messages": messages, 
            "participants": str(participant_ids),
            "current_time": current_time_str
        })
        
        extracted_dict = result.model_dump()
        params = extracted_dict.get('parameters', {})
        
        # MERGE LOGIC: If confirming, preserve previous parameters (like title) if missing
        if extracted_dict.get('intent') == 'confirm_schedule':
            prev_extracted = state.get('extracted_info', {})
            prev_params = prev_extracted.get('parameters', {})
            
            # List of fields to preserve if missing in new params
            fields_to_preserve = ['title', 'duration_minutes', 'date', 'start_time', 'end_time']
            
            for field in fields_to_preserve:
                if not params.get(field) and prev_params.get(field):
                    params[field] = prev_params[field]
                    debug_info.append(f"Preserved {field} from previous state: {params[field]}")
        
        # Post-processing: Calculate duration if both start_time and end_time are provided
        if params.get('start_time') and params.get('end_time') and not params.get('duration_minutes'):
            duration = calculate_duration_minutes(params['start_time'], params['end_time'])
            if duration:
                params['duration_minutes'] = duration
                debug_info.append(f"Calculated duration: {duration} minutes from {params['start_time']} to {params['end_time']}")
        
        # Ensure default duration if none provided
        if not params.get('duration_minutes'):
            params['duration_minutes'] = DEFAULT_DURATION_MINUTES
            debug_info.append(f"Using default duration: {DEFAULT_DURATION_MINUTES} minutes")
        
        debug_info.append(f"Extracted parameters: {params}")
        
        return {
            "extracted_info": extracted_dict,
            "debug_info": debug_info
        }
    except Exception as e:
        logger.error(f"Error in parse_input: {str(e)}", exc_info=True)
        debug_info.append(f"Error in parse_input: {str(e)}")
        return {
            "extracted_info": {"error": str(e), "intent": "error", "parameters": {}, "constraints": [], "missing_info": []},
            "debug_info": debug_info
        }

def suggest_next_days(state: AgentState, num_days: int = 5) -> List[dict]:
    """
    Search next N working days for available slots.
    Returns list of {day, time, all_available}.
    """
    organizer_id = state.get('organizer_id')
    participant_ids = state.get('participant_ids', [])
    all_user_names = [organizer_id] + participant_ids
    all_calendars = state.get('all_calendars', {})
    all_working_hours = state.get('all_working_hours', {})
    extracted_info = state.get('extracted_info', {})
    params = extracted_info.get('parameters', {})
    
    duration_minutes = params.get('duration_minutes', DEFAULT_DURATION_MINUTES)
    pref_start_time = parse_time_to_object(params.get('start_time')) if params.get('start_time') else None
    
    now = datetime.now(TIMEZONE)
    suggestions = []
    week_days_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    # Calculate common working days
    common_working_days = None
    for user_name in all_user_names:
        user_hours = all_working_hours.get(user_name, {})
        user_days = set(user_hours.get("working_days", []))
        if common_working_days is None:
            common_working_days = user_days
        else:
            common_working_days = common_working_days.intersection(user_days)
    
    if not common_working_days:
        return []
    
    # Search next days
    current_date = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    days_searched = 0
    
    while len(suggestions) < num_days and days_searched < 14:
        day_name = week_days_map[current_date.weekday()]
        
        if day_name in common_working_days:
            # Try to find a slot on this day
            if pref_start_time:
                slot_start = current_date.replace(hour=pref_start_time.hour, minute=pref_start_time.minute)
            else:
                # Default to 10 AM
                slot_start = current_date.replace(hour=10, minute=0)
            
            slot_end = slot_start + timedelta(minutes=duration_minutes)
            
            # Check if all users are available
            all_available = True
            for user_name in all_user_names:
                user_calendar = all_calendars.get(user_name, [])
                for meeting in user_calendar:
                    if has_conflict(meeting, {"start": slot_start, "end": slot_end}):
                        all_available = False
                        break
                if not all_available:
                    break
            
            suggestions.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "day_name": day_name,
                "start": slot_start.isoformat(),
                "end": slot_end.isoformat(),
                "all_available": all_available
            })
        
        current_date += timedelta(days=1)
        days_searched += 1
    
    return suggestions

def suggest_time_adjustments(state: AgentState) -> List[dict]:
    """
    Suggest different time windows on the same day.
    Returns earlier and later alternatives.
    """
    extracted_info = state.get('extracted_info', {})
    params = extracted_info.get('parameters', {})
    
    duration_minutes = params.get('duration_minutes', DEFAULT_DURATION_MINUTES)
    pref_date_str = params.get('date')
    
    if not pref_date_str:
        return []
    
    # Parse the requested date
    try:
        pref_date = datetime.strptime(pref_date_str, "%Y-%m-%d")
        pref_date = TIMEZONE.localize(pref_date)
    except:
        return []
    
    suggestions = []
    
    # Suggest earlier times (9 AM, 10 AM)
    for hour in [9, 10]:
        slot_start = pref_date.replace(hour=hour, minute=0)
        slot_end = slot_start + timedelta(minutes=duration_minutes)
        suggestions.append({
            "type": "earlier",
            "start": slot_start.isoformat(),
            "end": slot_end.isoformat()
        })
    
    # Suggest later times (4 PM, 5 PM)
    for hour in [16, 17]:
        slot_start = pref_date.replace(hour=hour, minute=0)
        slot_end = slot_start + timedelta(minutes=duration_minutes)
        suggestions.append({
            "type": "later",
            "start": slot_start.isoformat(),
            "end": slot_end.isoformat()
        })
    
    return suggestions

def suggest_partial_attendance(state: AgentState) -> List[dict]:
    """
    Find slots where most (but not all) participants are available.
    Returns slots with list of available and unavailable users.
    """
    organizer_id = state.get('organizer_id')
    participant_ids = state.get('participant_ids', [])
    all_user_names = [organizer_id] + participant_ids
    all_calendars = state.get('all_calendars', {})
    all_working_hours = state.get('all_working_hours', {})
    extracted_info = state.get('extracted_info', {})
    params = extracted_info.get('parameters', {})
    
    duration_minutes = params.get('duration_minutes', DEFAULT_DURATION_MINUTES)
    pref_start_time = parse_time_to_object(params.get('start_time')) if params.get('start_time') else time(14, 0)
    pref_date_str = params.get('date')
    
    if not pref_date_str:
        return []
    
    try:
        pref_date = datetime.strptime(pref_date_str, "%Y-%m-%d")
        pref_date = TIMEZONE.localize(pref_date)
    except:
        return []
    
    slot_start = pref_date.replace(hour=pref_start_time.hour, minute=pref_start_time.minute)
    slot_end = slot_start + timedelta(minutes=duration_minutes)
    
    available_users = []
    unavailable_users = []
    
    for user_name in all_user_names:
        user_calendar = all_calendars.get(user_name, [])
        is_available = True
        
        for meeting in user_calendar:
            if has_conflict(meeting, {"start": slot_start, "end": slot_end}):
                is_available = False
                break
        
        if is_available:
            available_users.append(user_name)
        else:
            unavailable_users.append(user_name)
    
    if len(available_users) > 0 and len(unavailable_users) > 0:
        return [{
            "start": slot_start.isoformat(),
            "end": slot_end.isoformat(),
            "available": available_users,
            "unavailable": unavailable_users,
            "attendance_rate": len(available_users) / len(all_user_names)
        }]
    
    return []

def suggest_duration_flexibility(state: AgentState) -> List[dict]:
    """
    Check if shorter durations would work.
    Returns suggestions for 30 min, 45 min, 60 min alternatives.
    """
    extracted_info = state.get('extracted_info', {})
    params = extracted_info.get('parameters', {})
    
    requested_duration = params.get('duration_minutes', DEFAULT_DURATION_MINUTES)
    
    # Only suggest shorter durations
    shorter_durations = [30, 45, 60]
    suggestions = []
    
    for duration in shorter_durations:
        if duration < requested_duration:
            suggestions.append({
                "duration_minutes": duration,
                "reduction": requested_duration - duration
            })
    
    return suggestions

def generate_alternatives(state: AgentState) -> dict:
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

def normalize_working_days(days: List[str]) -> List[str]:
    """
    Normalize working days to Short English names (Mon, Tue, etc.).
    Handles full names (Monday), short names (Mon), and ensures consistency.
    """
    mapping = {
        "monday": "Mon", "mon": "Mon",
        "tuesday": "Tue", "tue": "Tue",
        "wednesday": "Wed", "wed": "Wed",
        "thursday": "Thu", "thu": "Thu",
        "friday": "Fri", "fri": "Fri",
        "saturday": "Sat", "sat": "Sat",
        "sunday": "Sun", "sun": "Sun"
    }
    normalized = []
    for d in days:
        if isinstance(d, str):
            clean_d = d.lower().strip()
            if clean_d in mapping:
                normalized.append(mapping[clean_d])
    return list(set(normalized))

# Define the Fetch Calendars Node
async def fetch_calendars_node(state: AgentState):
    """
    Fetch all existing meetings for organizer and participants from Supabase.
    Converts timestamptz to timezone-aware datetime objects.
    """
    organizer_id = state.get('organizer_id')
    participant_ids = state.get('participant_ids', [])
    extracted_info = state.get('extracted_info', {})
    debug_info = state.get('debug_info', [])
    
    # Get all users (organizer + participants)
    all_user_names = [organizer_id] + participant_ids
    
    debug_info.append(f"Fetching calendars for {len(all_user_names)} users: {all_user_names}")
    
    try:
        # Determine search date range
        now = datetime.now(TIMEZONE)
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        params = extracted_info.get('parameters', {})
        if params.get('date'):
            try:
                naive_date = datetime.strptime(params.get('date'), "%Y-%m-%d")
                start_date = TIMEZONE.localize(naive_date)
                debug_info.append(f"Using specific date: {start_date.date()}")
            except (ValueError, AttributeError):
                debug_info.append(f"Failed to parse date '{params.get('date')}', using today")
        
        # Search range: specific day or next DEFAULT_SEARCH_DAYS
        end_date = start_date + timedelta(days=1) if params.get('date') else start_date + timedelta(days=DEFAULT_SEARCH_DAYS)
        
        debug_info.append(f"Search range: {start_date.date()} to {end_date.date()}")
        
        # Fetch all meetings for all users in the date range
        # Note: Using correct overlap logic: Meeting Start < Range End AND Meeting End > Range Start
        meetings_response = (supabase.table("meetings")
            .select("meeting_id, meeting_name, user, start_meeting, end_meeting")
            .in_("user", all_user_names)
            .lt("start_meeting", end_date.isoformat())
            .gt("end_meeting", start_date.isoformat())
            .execute())
        
        meetings_data = meetings_response.data
        debug_info.append(f"Fetched {len(meetings_data)} total meetings from database")
        
        # Organize by user and convert to datetime objects
        all_calendars = {}
        for user_name in all_user_names:
            user_meetings = [m for m in meetings_data if m.get("user") == user_name]
            
            parsed_meetings = []
            for meeting in user_meetings:
                try:
                    parsed_meeting = {
                        "start": parse_to_datetime(meeting["start_meeting"]),
                        "end": parse_to_datetime(meeting["end_meeting"]),
                        "title": meeting.get("meeting_name", ""),
                        "meeting_id": meeting.get("meeting_id")
                    }
                    parsed_meetings.append(parsed_meeting)
                    debug_info.append(f"  {user_name}: {parsed_meeting['title']} at {parsed_meeting['start'].strftime('%Y-%m-%d %H:%M')} - {parsed_meeting['end'].strftime('%H:%M')}")
                except Exception as e:
                    logger.error(f"Failed to parse meeting for {user_name}: {e}")
                    debug_info.append(f"  ERROR parsing meeting for {user_name}: {e}")
            
            all_calendars[user_name] = parsed_meetings
            debug_info.append(f"User {user_name}: {len(parsed_meetings)} meetings")
        
        return {
            "all_calendars": all_calendars,
            "debug_info": debug_info
        }
        
    except Exception as e:
        logger.error(f"Error fetching calendars: {str(e)}", exc_info=True)
        debug_info.append(f"ERROR in fetch_calendars_node: {str(e)}")
        return {
            "all_calendars": {},
            "debug_info": debug_info
        }

# Define the Fetch Working Hours Node
async def fetch_working_hours_node(state: AgentState):
    """
    Fetch working hours and working days for organizer and participants from Supabase.
    Normalizes working days to consistent format (Mon, Tue, etc.).
    """
    organizer_id = state.get('organizer_id')
    participant_ids = state.get('participant_ids', [])
    debug_info = state.get('debug_info', [])
    
    # Get all users (organizer + participants)
    all_user_names = [organizer_id] + participant_ids
    
    debug_info.append(f"Fetching working hours for {len(all_user_names)} users")
    
    try:
        # Query Supabase for all users' working preferences
        response = supabase.table("users").select("name, work_start, work_end, working_days").in_("name", all_user_names).execute()
        users_data = response.data
        
        debug_info.append(f"Retrieved data for {len(users_data)} users from database")
        
        all_working_hours = {}
        for user in users_data:
            name = user.get("name")
            
            # Parse timetz fields (e.g., "09:00:00+08:00" -> "09:00")
            work_start = parse_timetz(user.get("work_start", "09:00:00+00"))
            work_end = parse_timetz(user.get("work_end", "17:00:00+00"))
            
            # Normalize working days to ["Mon", "Tue", etc.]
            raw_days = user.get("working_days", [])
            working_days = normalize_working_days(raw_days)
            
            all_working_hours[name] = {
                "start": work_start,
                "end": work_end,
                "working_days": working_days
            }
            
            debug_info.append(f"User {name}: {work_start}-{work_end}, days: {working_days}")
        
        # Check if any users are missing
        missing_users = set(all_user_names) - set(all_working_hours.keys())
        if missing_users:
            debug_info.append(f"WARNING: Missing working hours for users: {missing_users}")
            logger.warning(f"Missing working hours for users: {missing_users}")
        
        return {
            "all_working_hours": all_working_hours,
            "debug_info": debug_info
        }
        
    except Exception as e:
        logger.error(f"Error fetching working hours: {str(e)}", exc_info=True)
        debug_info.append(f"ERROR in fetch_working_hours_node: {str(e)}")
        return {
            "all_working_hours": {},
            "debug_info": debug_info
        }

# Define the Confirm Schedule Node
async def confirm_schedule_node(state: AgentState):
    """
    Confirm a meeting schedule by creating a proposal in Supabase.
    Creates meeting_proposals record and participant_responses for all participants.
    
    Since proposed_slots may not persist between conversation turns, we need to:
    1. Check if we need to re-fetch calendars and find slots
    2. Or ask the user to re-request the meeting
    """
    organizer_id = state.get('organizer_id')
    participant_ids = state.get('participant_ids', [])
    proposed_slots = state.get('proposed_slots', [])
    extracted_info = state.get('extracted_info', {})
    debug_info = state.get('debug_info', [])
    messages = state.get('messages', [])
    
    debug_info.append("=== Confirming Schedule ===")
    
    try:
        # Validate that we have proposed slots to confirm
        if not proposed_slots:
            debug_info.append("ERROR: No proposed slots available to confirm")
            logger.error("confirm_schedule_node called without proposed_slots")
            logger.info("This may happen if state wasn't persisted between turns")
            
            # Return error status - respond node will handle asking user to re-request
            return {
                "confirmation_status": "needs_scheduling",
                "debug_info": debug_info
            }
        
        # Extract slot selection from extracted_info
        slot_selection = extracted_info.get('slot_selection', 1)
        if slot_selection is None or slot_selection < 1:
            slot_selection = 1  # Default to first slot
        
        # Validate slot selection is within range
        if slot_selection > len(proposed_slots):
            debug_info.append(f"WARNING: Slot {slot_selection} requested but only {len(proposed_slots)} available, using slot 1")
            slot_selection = 1
        
        # Get the selected slot (convert to 0-based index)
        selected_slot = proposed_slots[slot_selection - 1]
        
        debug_info.append(f"Selected slot {slot_selection}: {selected_slot['start']} - {selected_slot['end']}")
        
        # Extract meeting details
        params = extracted_info.get('parameters', {})
        meeting_title = params.get('title', 'Meeting')
        reasoning = ', '.join(selected_slot.get('ranking_reasons', ['Best available time']))
        
        # Prepare proposal data
        proposal_data = {
            "organizer_id": organizer_id,
            "participant_ids": participant_ids,  # PostgreSQL TEXT[] array
            "proposed_start": selected_slot['start'],
            "proposed_end": selected_slot['end'],
            "meeting_title": meeting_title,
            "reasoning": reasoning,
            "status": "pending",
            "iteration_count": 1
        }
        
        debug_info.append(f"Creating proposal: {meeting_title} from {selected_slot['start']} to {selected_slot['end']}")
        debug_info.append(f"Participants: {participant_ids}")
        
        # Insert into meeting_proposals table
        proposal_response = supabase.table("meeting_proposals").insert(proposal_data).execute()
        
        if not proposal_response.data or len(proposal_response.data) == 0:
            debug_info.append("ERROR: Failed to create meeting proposal")
            logger.error("No data returned from meeting_proposals insert")
            return {
                "confirmation_status": "error",
                "debug_info": debug_info
            }
        
        proposal = proposal_response.data[0]
        proposal_id = proposal.get('proposal_id')
        
        debug_info.append(f"✓ Created proposal with ID: {proposal_id}")
        
        # Create participant_responses records for each participant
        participant_responses = []
        for participant_id in participant_ids:
            response_data = {
                "proposal_id": proposal_id,
                "participant_id": participant_id,
                "response": "pending"
            }
            participant_responses.append(response_data)
        
        # Bulk insert participant responses
        if participant_responses:
            responses_result = supabase.table("participant_responses").insert(participant_responses).execute()
            debug_info.append(f"✓ Created {len(participant_responses)} participant response records")
            logger.info(f"Created participant responses for proposal {proposal_id}")
        
        return {
            "proposal_id": proposal_id,
            "confirmation_status": "confirmed",
            "debug_info": debug_info
        }
        
    except Exception as e:
        logger.error(f"Error in confirm_schedule_node: {str(e)}", exc_info=True)
        debug_info.append(f"ERROR in confirm_schedule_node: {str(e)}")
        return {
            "confirmation_status": "error",
            "debug_info": debug_info
        }

# Define the Find Slots Node
async def find_slots_node(state: AgentState):
    """
    Find available meeting slots by checking:
    1. Working hours for ALL users (organizer + participants)
    2. Working days for ALL users
    3. No conflicts with existing meetings (using datetime overlap detection)
    
    Uses the new state structure with all_calendars and all_working_hours.
    """
    organizer_id = state.get('organizer_id')
    participant_ids = state.get('participant_ids', [])
    extracted_info = state.get('extracted_info', {})
    all_calendars = state.get('all_calendars', {})
    all_working_hours = state.get('all_working_hours', {})
    debug_info = state.get('debug_info', [])
    
    # Get all users (organizer + participants)
    all_user_names = [organizer_id] + participant_ids
    
    debug_info.append(f"Finding slots for {len(all_user_names)} users")
    
    # Validate we have required data
    if not extracted_info or "error" in extracted_info:
        debug_info.append("ERROR: Missing or invalid extracted_info")
        logger.warning("Missing required extracted_info for slot calculation")
        return {"candidate_slots": [], "debug_info": debug_info}
    
    if not all_calendars:
        debug_info.append("ERROR: Missing all_calendars data")
        logger.warning("Missing all_calendars data")
        return {"candidate_slots": [], "debug_info": debug_info}
    
    if not all_working_hours:
        debug_info.append("ERROR: Missing all_working_hours data")
        logger.warning("Missing all_working_hours data")
        return {"candidate_slots": [], "debug_info": debug_info}
        
    params = extracted_info.get('parameters', {})
    
    # 1. Define Scope (Date Range) with timezone awareness
    now = datetime.now(TIMEZONE).replace(second=0, microsecond=0)
    start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    debug_info.append(f"Current time (now): {now.strftime('%Y-%m-%d %H:%M %Z')}")
    
    if params.get('date'):
        try:
            naive_date = datetime.strptime(params.get('date'), "%Y-%m-%d")
            start_date = TIMEZONE.localize(naive_date)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse date '{params.get('date')}': {e}")
            pass  # Fallback to today if parse fails

    # If no specific date, default to searching next DEFAULT_SEARCH_DAYS
    # If specific date, just search that day
    end_date = start_date + timedelta(days=1) if params.get('date') else start_date + timedelta(days=DEFAULT_SEARCH_DAYS)
    
    duration_minutes = params.get('duration_minutes', DEFAULT_DURATION_MINUTES)
    if not duration_minutes or duration_minutes <= 0:
        duration_minutes = DEFAULT_DURATION_MINUTES
    
    debug_info.append(f"Duration: {duration_minutes} minutes, Range: {start_date.date()} to {end_date.date()}")
    
    # Parse preferred time window
    pref_start_time = None
    pref_end_time = None
    if params.get('start_time'):
        pref_start_time = parse_time_to_object(params.get('start_time'))
        debug_info.append(f"Preferred start time: {pref_start_time}")
        
    if params.get('end_time'):
        pref_end_time = parse_time_to_object(params.get('end_time'))
        debug_info.append(f"Preferred end time: {pref_end_time}")

    # EDGE CASE 1: Check if requested time is in the past
    if params.get('date') and params.get('start_time'):
        try:
            date_obj = datetime.strptime(params['date'], "%Y-%m-%d")
            time_obj = parse_time_to_object(params['start_time'])
            if time_obj:
                requested_datetime = TIMEZONE.localize(
                    datetime.combine(date_obj.date(), time_obj)
                )
                
                if requested_datetime < now:
                    debug_info.append(f"❌ PAST TIME: Requested {requested_datetime.strftime('%Y-%m-%d %H:%M')} but current time is {now.strftime('%Y-%m-%d %H:%M')}")
                    # Return empty slots with special flag for respond node
                    return {
                        "candidate_slots": [],
                        "debug_info": debug_info
                    }
        except Exception as e:
            debug_info.append(f"Could not validate past time: {e}")

    # 2. Calculate common working days and hours (intersection)
    # Only generate slots on days when ALL users can work
    common_working_days = None
    earliest_start = None
    latest_end = None
    
    week_days_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    for user_name in all_user_names:
        user_hours = all_working_hours.get(user_name)
        if not user_hours:
            debug_info.append(f"WARNING: No working hours for user: {user_name}")
            logger.warning(f"No working hours data for user: {user_name}")
            continue
            
        user_days = set(user_hours.get("working_days", []))
        if common_working_days is None:
            common_working_days = user_days
        else:
            common_working_days = common_working_days.intersection(user_days)
        
        # Track earliest start and latest end times (intersection = most restrictive)
        w_start = parse_time_to_object(user_hours.get("start", "09:00"))
        w_end = parse_time_to_object(user_hours.get("end", "17:00"))
        
        if w_start:
            earliest_start = max(earliest_start, w_start) if earliest_start else w_start
        if w_end:
            latest_end = min(latest_end, w_end) if latest_end else w_end
    
    if not common_working_days:
        debug_info.append("ERROR: No common working days found among all users")
        logger.info("No common working days found among users")
        return {"candidate_slots": [], "debug_info": debug_info}
    
    debug_info.append(f"Common working days: {list(common_working_days)}, hours: {earliest_start}-{latest_end}")
    
    # EDGE CASE 2: Check if requested time is before working hours
    if params.get('start_time') and earliest_start:
        pref_start = parse_time_to_object(params['start_time'])
        if pref_start and pref_start < earliest_start:
            debug_info.append(f"❌ BEFORE WORKING HOURS: Requested {pref_start.strftime('%H:%M')} but earliest working hour is {earliest_start.strftime('%H:%M')}")
            # Return early - don't generate any slots
            return {
                "candidate_slots": [],
                "debug_info": debug_info
            }
    
    # EDGE CASE 2B: Check if requested time is after working hours
    if params.get('start_time') and latest_end:
        pref_start = parse_time_to_object(params['start_time'])
        if pref_start and pref_start >= latest_end:
            debug_info.append(f"❌ AFTER WORKING HOURS: Requested {pref_start.strftime('%H:%M')} but latest working hour ends at {latest_end.strftime('%H:%M')}")
            # Return early - don't generate any slots
            return {
                "candidate_slots": [],
                "debug_info": debug_info
            }
    
    # EDGE CASE 4: Check if requested date is a non-working day
    if params.get('date') and common_working_days:
        try:
            requested_date = datetime.strptime(params['date'], "%Y-%m-%d")
            requested_day_name = week_days_map[requested_date.weekday()]
            
            if requested_day_name not in common_working_days:
                debug_info.append(f"❌ NON-WORKING DAY: Requested {requested_day_name} but common working days are {list(common_working_days)}")
                # Continue to generate alternatives for next working days
                # Don't return early - let alternatives be generated
        except (ValueError, AttributeError) as e:
            debug_info.append(f"Could not validate non-working day: {e}")
    
    # 3. Generate Candidate Slots
    candidate_slots = []
    current = start_date
    
    # Start from earliest common working hour if no preference specified
    if not pref_start_time and earliest_start:
        current = current.replace(hour=earliest_start.hour, minute=earliest_start.minute)
    
    while current < end_date:
        # Skip if not a common working day
        day_name = week_days_map[current.weekday()]
        if day_name not in common_working_days:
            # Jump to next day at the start time
            current = (current + timedelta(days=1)).replace(hour=earliest_start.hour if earliest_start else 0, minute=earliest_start.minute if earliest_start else 0)
            continue
        
        slot_start = current
        slot_end = current + timedelta(minutes=duration_minutes)
        
        # Filter: Don't suggest slots in the past
        if slot_start < now:
            current += timedelta(minutes=SLOT_INCREMENT_MINUTES)
            continue

        # Filter: Preferred Time Window
        if pref_start_time and slot_start.time() < pref_start_time:
            current += timedelta(minutes=SLOT_INCREMENT_MINUTES)
            continue
        
        if pref_end_time:
            # Slot should end by pref_end_time
            if slot_end.time() > pref_end_time and slot_end.time() != time(0,0):
                current += timedelta(minutes=SLOT_INCREMENT_MINUTES)
                continue
        
        # Filter: Skip slots outside common working hours
        if latest_end and slot_end.time() > latest_end and slot_end.time() != time(0,0):
            # Jump to next day
            current = (current + timedelta(days=1)).replace(hour=earliest_start.hour if earliest_start else 0, minute=earliest_start.minute if earliest_start else 0)
            continue
        
        candidate_slots.append({
            "start": slot_start,
            "end": slot_end,
            "score": 0,
            "available_participants": []
        })
        current += timedelta(minutes=SLOT_INCREMENT_MINUTES)
    
    debug_info.append(f"Generated {len(candidate_slots)} candidate slots")
    
    # 4. Check each candidate slot against ALL users' calendars and working hours
    valid_slots = []
    
    debug_info.append(f"\n=== Validating {len(candidate_slots)} candidate slots ===")
    
    for idx, slot in enumerate(candidate_slots):
        slot_start = slot['start']
        slot_end = slot['end']
        available_users = []
        
        debug_info.append(f"\nSlot {idx+1}: {slot_start.strftime('%Y-%m-%d %H:%M')} - {slot_end.strftime('%H:%M')} ({slot_start.strftime('%A')})")
        
        # Check each user's availability for this slot
        for user_name in all_user_names:
            # Get user's working hours and calendar
            user_hours = all_working_hours.get(user_name)
            user_calendar = all_calendars.get(user_name, [])
            
            if not user_hours:
                debug_info.append(f"  ❌ {user_name}: No working hours data")
                continue
                
            # --- CHECK A: Working Day ---
            day_name = week_days_map[slot_start.weekday()]
            working_days = user_hours.get("working_days", [])
            if day_name not in working_days:
                debug_info.append(f"  ❌ {user_name}: {day_name} not in working days {working_days}")
                continue
            
            # --- CHECK B: Working Hours ---
            w_start_str = user_hours.get("start")
            w_end_str = user_hours.get("end")
            
            if not w_start_str or not w_end_str:
                debug_info.append(f"  ❌ {user_name}: Missing working hours")
                continue
                
            w_start_time = parse_time_to_object(w_start_str)
            w_end_time = parse_time_to_object(w_end_str)
            
            if not w_start_time or not w_end_time:
                debug_info.append(f"  ❌ {user_name}: Could not parse working hours")
                continue
                
            slot_start_time = slot_start.time()
            slot_end_time = slot_end.time()
            
            # Check if slot is within working hours
            if slot_start_time < w_start_time or slot_start_time > w_end_time:
                debug_info.append(f"  ❌ {user_name}: Slot start {slot_start_time} outside work hours {w_start_time}-{w_end_time}")
                continue
            
            # Handle midnight edge case
            if slot_end_time != time(0, 0) and slot_end_time > w_end_time:
                debug_info.append(f"  ❌ {user_name}: Slot end {slot_end_time} after work end {w_end_time}")
                continue

            # --- CHECK C: Existing Meetings (using has_conflict function) ---
            is_busy = False
            conflict_meeting = None
            
            for meeting in user_calendar:
                # meetings are already parsed as datetime objects with "start" and "end" keys
                proposed_slot = {"start": slot_start, "end": slot_end}
                
                if has_conflict(meeting, proposed_slot):
                    is_busy = True
                    conflict_meeting = meeting
                    break
            
            if is_busy:
                debug_info.append(f"  ❌ {user_name}: Conflict with '{conflict_meeting.get('title', 'meeting')}' at {conflict_meeting['start'].strftime('%H:%M')}-{conflict_meeting['end'].strftime('%H:%M')}")
                continue
            
            # User is available!
            available_users.append(user_name)
            debug_info.append(f"  ✅ {user_name}: Available")
        
        # Scoring Logic: Only add slot if ALL users are available
        debug_info.append(f"  Result: {len(available_users)}/{len(all_user_names)} users available")
        
        if len(available_users) == len(all_user_names):
            valid_slots.append({
                "start": slot_start.isoformat(),
                "end": slot_end.isoformat(),
                "score": 3,  # Full attendance
                "available_participants": available_users
            })
            debug_info.append(f"  ✅ VALID SLOT - All users available!")
        else:
            debug_info.append(f"  ❌ REJECTED - Not all users available")
    
    # Sort by start time ascending
    valid_slots.sort(key=lambda x: x['start'])
    
    debug_info.append(f"Found {len(valid_slots)} valid slots (returning top {MAX_SLOTS_RETURNED})")
    
    return {
        "candidate_slots": valid_slots[:MAX_SLOTS_RETURNED],
        "debug_info": debug_info
    }

# Define the Select Best Slot Node
async def select_best_slot(state: AgentState):
    """
    Rank candidate slots using multi-factor scoring,
    or generate alternatives if no slots are found.
    """
    candidate_slots = state.get('candidate_slots', [])
    extracted_info = state.get('extracted_info', {})
    debug_info = state.get('debug_info', [])
    
    debug_info.append(f"\n=== Selecting best slot from {len(candidate_slots)} candidates ===")
    
    # Case 1: Good slots found - rank them
    if candidate_slots:
        ranked_slots = rank_slots(candidate_slots, extracted_info)
        top_3 = ranked_slots[:3]
        
        debug_info.append(f"Ranked {len(ranked_slots)} slots, selecting top {len(top_3)}")
        
        for i, slot in enumerate(top_3, 1):
            score = slot.get('ranking_score', 0)
            reasons = slot.get('ranking_reasons', [])
            start_dt = datetime.fromisoformat(slot['start'])
            debug_info.append(f"  {i}. Score {score}: {start_dt.strftime('%Y-%m-%d %H:%M')} - {', '.join(reasons)}")
        
        return {
            "proposed_slots": top_3,
            "alternatives": None,
            "debug_info": debug_info
        }
    
    # Case 2: No slots found - generate alternatives
    else:
        debug_info.append("No perfect slots found, generating alternatives...")
        alternatives = generate_alternatives(state)
        
        # Count alternatives
        next_days_count = len(alternatives.get('next_days', []))
        time_adj_count = len(alternatives.get('time_adjustments', []))
        partial_count = len(alternatives.get('partial_attendance', []))
        duration_count = len(alternatives.get('duration_flexibility', []))
        
        debug_info.append(f"  Next available days: {next_days_count}")
        debug_info.append(f"  Time adjustments: {time_adj_count}")
        debug_info.append(f"  Partial attendance: {partial_count}")
        debug_info.append(f"  Duration flexibility: {duration_count}")
        
        return {
            "proposed_slots": [],
            "alternatives": alternatives,
            "debug_info": debug_info
        }

# Define the LLM-based Responder Node
async def respond(state: AgentState):
    """
    LLM-based responder that generates natural, conversational responses
    while maintaining structure for better UX.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    
    organizer_id = state.get('organizer_id', 'Unknown')
    participant_ids = state.get('participant_ids', [])
    extracted_info = state.get('extracted_info', {})
    proposed_slots = state.get('proposed_slots', [])
    alternatives = state.get('alternatives')
    
    params = extracted_info.get('parameters', {})
    debug_info = state.get('debug_info', [])
    all_working_hours = state.get('all_working_hours', {})
    
    # Prepare structured context for LLM
    now = datetime.now(TIMEZONE)
    
    # Detect edge cases
    missing_info = extracted_info.get('missing_info', [])
    past_time_detected = any("PAST TIME" in log for log in debug_info)
    before_hours_detected = any("BEFORE WORKING HOURS" in log for log in debug_info)
    
    # Build context object
    context = {
        "organizer": organizer_id,
        "participants": participant_ids,
        "duration_minutes": params.get('duration_minutes'),
        "requested_date": params.get('date'),
        "requested_time": params.get('start_time'),
        "meeting_title": params.get('title'),
        "current_time": now.strftime('%A, %B %d at %I:%M %p'),
        "has_slots": len(proposed_slots) > 0,
        "has_alternatives": alternatives is not None,
        "missing_info": missing_info,
        "past_time": past_time_detected,
        "before_hours": before_hours_detected,
        "working_hours": all_working_hours
    }
    
    # Prepare LLM prompt based on scenario
    scenario = ""
    scenario_data = {}
    
    # EDGE CASE HANDLING: Check for special conditions first
    
    # Edge Case 1: Missing critical information
    if missing_info and not proposed_slots and not alternatives:
        scenario = "missing_info"
        scenario_data = {
            "missing_fields": missing_info,
            "needs_date": 'date or time preference' in missing_info or not params.get('date'),
            "needs_duration": 'duration' in missing_info,
            "needs_title": 'title' in missing_info or not params.get('title')
        }
        
        prompt = f"""You are a helpful scheduling assistant. The user wants to schedule a meeting, but some information is missing.

Missing information: {', '.join(missing_info)}
Participants: {', '.join([organizer_id] + participant_ids)}

Generate a friendly, concise response asking for the missing information. Use this structure:

1. Start with a warm greeting acknowledging their request
2. For each missing piece of information, ask specifically:
   - If date/time is missing: Ask when they'd like to meet (give 2-3 natural examples)
   - If duration is missing: Ask how long the meeting should be (give time options)
   - If title is missing: Ask what to name the meeting (give professional examples)
3. End with an encouraging call to action

Use markdown formatting:
- Use **bold** for field labels
- Use emojis sparingly (📅 for date, ⏱️ for duration, 📝 for title)
- Use bullet points for examples
- Keep it conversational and friendly

Generate the response now:"""

        response_text = await llm.ainvoke([HumanMessage(content=prompt)])
        ai_message = AIMessage(content=response_text.content)
        return {"messages": [ai_message]}
    
    # Edge Case 2: Requested time is in the past
    if past_time_detected and not proposed_slots:
        requested_datetime = ""
        if params.get('date') and params.get('start_time'):
            req_date = datetime.strptime(params['date'], "%Y-%m-%d").strftime('%A, %B %d')
            start_time_obj = parse_time_to_object(params['start_time'])
            if start_time_obj:
                requested_datetime = f"{req_date} at {start_time_obj.strftime('%I:%M %p')}"
            else:
                requested_datetime = req_date
        
        prompt = f"""You are a helpful scheduling assistant. The user requested a meeting in the past.

Requested time: {requested_datetime}
Current time: {now.strftime('%A, %B %d at %I:%M %p')}
Participants: {', '.join([organizer_id] + participant_ids)}

Generate a friendly, understanding response that:
1. Gently points out that the requested time has already passed
2. Shows empathy (no criticism)
3. Offers 3-4 helpful alternatives:
   - Later today (if applicable)
   - Tomorrow at the same time
   - Next available slot
   - Any time that works for them

Use markdown formatting with emojis (⏰). Keep the tone warm and solution-oriented.

Generate the response now:"""

        response_text = await llm.ainvoke([HumanMessage(content=prompt)])
        ai_message = AIMessage(content=response_text.content)
        return {"messages": [ai_message]}
    
    # Edge Case 3: Requested time is before working hours
    if before_hours_detected and not proposed_slots:
        requested_time = ""
        if params.get('start_time'):
            start_time_obj = parse_time_to_object(params['start_time'])
            if start_time_obj:
                requested_time = start_time_obj.strftime('%I:%M %p')
        
        # Get earliest available time
        earliest_start = None
        for user_name in [organizer_id] + participant_ids:
            if user_name in all_working_hours:
                w_start = parse_time_to_object(all_working_hours[user_name].get('start', '09:00'))
                if w_start:
                    earliest_start = max(earliest_start, w_start) if earliest_start else w_start
        
        # Format working hours info
        hours_info = []
        for user_name in [organizer_id] + participant_ids:
            if user_name in all_working_hours:
                hours = all_working_hours[user_name]
                start = hours.get('start', '09:00')
                end = hours.get('end', '17:00')
                hours_info.append(f"{user_name}: {start} - {end}")
        
        prompt = f"""You are a helpful scheduling assistant. The user requested a meeting outside working hours.

Requested time: {requested_time}
Working hours:
{chr(10).join(f"- {h}" for h in hours_info)}
Earliest available: {earliest_start.strftime('%I:%M %p') if earliest_start else '09:00 AM'}

Generate a friendly response that:
1. Politely explains the time is outside working hours
2. Shows each person's working hours clearly
3. Suggests 3-4 alternatives:
   - The earliest working hour time
   - Mid-morning (10am)
   - Any time during working hours
   - Flexible timing

Use markdown formatting with emojis (⏰). Be understanding and helpful.

Generate the response now:"""

        response_text = await llm.ainvoke([HumanMessage(content=prompt)])
        ai_message = AIMessage(content=response_text.content)
        return {"messages": [ai_message]}
    
    # Edge Case 3B: Requested time is after working hours
    after_hours_detected = any("AFTER WORKING HOURS" in log for log in debug_info)
    if after_hours_detected and not proposed_slots:
        requested_time = ""
        if params.get('start_time'):
            start_time_obj = parse_time_to_object(params['start_time'])
            if start_time_obj:
                requested_time = start_time_obj.strftime('%I:%M %p')
        
        # Get latest available time
        latest_end = None
        for user_name in [organizer_id] + participant_ids:
            if user_name in all_working_hours:
                w_end = parse_time_to_object(all_working_hours[user_name].get('end', '17:00'))
                if w_end:
                    latest_end = min(latest_end, w_end) if latest_end else w_end
        
        # Format working hours info
        hours_info = []
        for user_name in [organizer_id] + participant_ids:
            if user_name in all_working_hours:
                hours = all_working_hours[user_name]
                start = hours.get('start', '09:00')
                end = hours.get('end', '17:00')
                hours_info.append(f"{user_name}: {start} - {end}")
        
        prompt = f"""You are a helpful scheduling assistant. The user requested a meeting after working hours.

Requested time: {requested_time}
Working hours:
{chr(10).join(f"- {h}" for h in hours_info)}
Latest working hour ends: {latest_end.strftime('%I:%M %p') if latest_end else '05:00 PM'}

Generate a friendly response that:
1. Politely explains the time is after working hours
2. Shows each person's working hours clearly
3. Suggests 3-4 alternatives:
   - An earlier time today (if still during work hours)
   - Tomorrow at a reasonable time
   - Any time during working hours
   - Flexible morning/afternoon options

Use markdown formatting with emojis (⏰). Be understanding and helpful.

Generate the response now:"""

        response_text = await llm.ainvoke([HumanMessage(content=prompt)])
        ai_message = AIMessage(content=response_text.content)
        return {"messages": [ai_message]}
    
    # Edge Case 4: Non-working day requested
    non_working_day_detected = any("NON-WORKING DAY" in log for log in debug_info)
    if non_working_day_detected and not proposed_slots:
        requested_date = ""
        requested_day = ""
        if params.get('date'):
            req_dt = datetime.strptime(params['date'], "%Y-%m-%d")
            requested_date = req_dt.strftime('%A, %B %d')
            requested_day = req_dt.strftime('%A')
        
        # Get common working days
        common_days = []
        if all_working_hours:
            common_set = None
            for user_name in [organizer_id] + participant_ids:
                if user_name in all_working_hours:
                    user_days = set(all_working_hours[user_name].get('working_days', []))
                    if common_set is None:
                        common_set = user_days
                    else:
                        common_set = common_set.intersection(user_days)
            if common_set:
                common_days = sorted(list(common_set))
        
        # Get next working day alternatives from alternatives dict
        next_days_info = []
        if alternatives and alternatives.get('next_days'):
            for alt in alternatives.get('next_days', [])[:3]:
                start_dt = datetime.fromisoformat(alt['start'])
                next_days_info.append(f"{start_dt.strftime('%A, %B %d at %I:%M %p')}")
        
        prompt = f"""You are a helpful scheduling assistant. The user requested a meeting on a non-working day.

Requested date: {requested_date} ({requested_day})
Common working days: {', '.join(common_days)}
Participants: {', '.join([organizer_id] + participant_ids)}

Next available working days:
{chr(10).join(f"- {day}" for day in next_days_info) if next_days_info else "- No specific alternatives found"}

Generate a friendly response that:
1. Gently explains that {requested_day} is not a working day for all participants
2. Shows which days ARE working days
3. Lists the next 2-3 available working days as specific alternatives
4. Includes clear action prompts like "Say: 'Try Monday instead'"
5. Offers flexibility ("or suggest any day that works for you")

Use markdown formatting with emojis (📅). Be helpful and understanding about the mix-up.

Generate the response now:"""

        response_text = await llm.ainvoke([HumanMessage(content=prompt)])
        ai_message = AIMessage(content=response_text.content)
        return {"messages": [ai_message]}
    
    # CONFIRMATION FLOW: Check if a proposal was just created
    proposal_id = state.get('proposal_id')
    confirmation_status = state.get('confirmation_status')
    
    # Handle case where user tried to confirm but slots weren't available
    if confirmation_status == 'needs_scheduling':
        prompt = f"""You are a helpful scheduling assistant. The user tried to confirm a meeting slot, but there are no active proposals to confirm.

This happens when:
- The conversation was interrupted or reset
- They're trying to confirm before requesting a meeting
- The previous suggestions expired

Generate a friendly, helpful response that:
1. Explains they need to first request a meeting to see available times
2. Gives them a clear example of what to say:
   - "Schedule a meeting with [participants] on [date] at [time]"
   - "I need to meet with Bob tomorrow afternoon"
3. Reassures them it's an easy two-step process:
   - Step 1: Request meeting → See suggestions
   - Step 2: Confirm preferred slot
4. Use a warm, encouraging tone

Use markdown formatting with emojis (📅, 💡). Keep it brief and actionable.

Generate the response now:"""

        response_text = await llm.ainvoke([HumanMessage(content=prompt)])
        ai_message = AIMessage(content=response_text.content)
        return {"messages": [ai_message]}
    
    if proposal_id and confirmation_status == 'confirmed':
        # Get the confirmed slot details from proposed_slots
        slot_selection = extracted_info.get('slot_selection', 1)
        if slot_selection > len(proposed_slots):
            slot_selection = 1
        
        confirmed_slot = proposed_slots[slot_selection - 1] if proposed_slots else None
        
        confirmed_duration = 30  # Default
        if confirmed_slot:
            start_dt = datetime.fromisoformat(confirmed_slot['start'])
            end_dt = datetime.fromisoformat(confirmed_slot['end'])
            confirmed_time = f"{start_dt.strftime('%A, %B %d at %I:%M %p')} - {end_dt.strftime('%I:%M %p')}"
            
            # Calculate actual duration from the slot
            duration_seconds = (end_dt - start_dt).total_seconds()
            confirmed_duration = int(duration_seconds / 60)
        else:
            confirmed_time = "the selected time"
            confirmed_duration = params.get('duration_minutes', 30)
        
        prompt = f"""You are a helpful scheduling assistant. The user just confirmed a meeting!

Meeting Confirmation Details:
- Proposal ID: {proposal_id}
- Meeting Title: {params.get('title', 'Meeting')}
- Time: {confirmed_time if confirmed_slot else 'TBD'}
- Duration: {confirmed_duration} minutes
- Organizer: {organizer_id}
- Participants: {', '.join(participant_ids)}
- Status: Pending participant responses

Generate an enthusiastic confirmation response that:
1. Celebrates the booking with 🎉 emoji
2. Shows a clear summary of the confirmed meeting details
3. Explains the next steps:
   - Proposal has been created and saved
   - All participants have been notified through their dashboards.
   - Waiting for participant responses (status: pending)
   - Meeting will be finalized once participants respond
4. Provides the proposal ID for reference
5. Offers helpful options:
   - Participants can accept or decline
   - Organizer can cancel if needed
   - Organizer can check status anytime

Use markdown formatting:
- Use **bold** for important details (time, participants)
- Use bullet points for next steps
- Include relevant emojis (🎉, ✅, 📧, ⏰)
- Keep tone professional but warm and encouraging

Generate the confirmation message now:"""

        response_text = await llm.ainvoke([HumanMessage(content=prompt)])
        ai_message = AIMessage(content=response_text.content)
        return {"messages": [ai_message]}
    
    # NORMAL FLOW: If no edge cases, proceed with standard responses
    
    # Case 1: Slots were found
    if proposed_slots:
        # Format slot information
        slots_info = []
        for i, slot in enumerate(proposed_slots, 1):
            start_dt = datetime.fromisoformat(slot['start'])
            end_dt = datetime.fromisoformat(slot['end'])
            reasons = slot.get('ranking_reasons', [])
            time_str = f"{start_dt.strftime('%A, %B %d at %I:%M %p')} - {end_dt.strftime('%I:%M %p')}"
            slots_info.append({
                "number": i,
                "time": time_str,
                "reasons": reasons,
                "is_top": i == 1
            })
        
        # Get request details
        req_details = ""
        if params.get('date') and params.get('start_time'):
            req_date = datetime.strptime(params['date'], "%Y-%m-%d").strftime('%A, %B %d')
            start_time_obj = parse_time_to_object(params['start_time'])
            if start_time_obj:
                req_details = f"{req_date} from {start_time_obj.strftime('%I:%M %p')}"
        
        prompt = f"""You are a helpful scheduling assistant. You've found available meeting times!

Meeting Details:
- Title: {params.get('title', 'Meeting')}
- Duration: {params.get('duration_minutes', 30)} minutes
- Participants: {', '.join([organizer_id] + participant_ids)}
- Requested: {req_details if req_details else 'Not specified'}

Available Slots (ranked by preference):
{chr(10).join(f"{s['number']}. {'⭐ ' if s['is_top'] else ''}{s['time']}" + (f" - {', '.join(s['reasons'])}" if s['reasons'] else '') for s in slots_info)}

Generate an enthusiastic, helpful response that:
1. Celebrates finding available times (use 🎉 emoji)
2. Summarizes the meeting details clearly
3. Lists the top {len(proposed_slots)} slots with:
   - Slot numbers for easy reference
   - Times in readable format
   - Brief reasons why each works (from the data)
   - Highlight the top recommendation with ⭐
4. Provides clear instructions for booking:
   - "Book slot 1" or similar phrases
   - Option to request different times
5. Keep it conversational and encouraging

Use markdown formatting. Be enthusiastic but professional.

Generate the response now:"""

        response_text = await llm.ainvoke([HumanMessage(content=prompt)])
        ai_message = AIMessage(content=response_text.content)
        return {"messages": [ai_message]}
    
    # Case 2: No slots found - show alternatives
    elif alternatives:
        # Get request details
        req_details = ""
        if params.get('date'):
            req_date = datetime.strptime(params['date'], "%Y-%m-%d").strftime('%A, %B %d')
            req_details = req_date
            if params.get('start_time'):
                start_time_obj = parse_time_to_object(params['start_time'])
                if start_time_obj:
                    req_details += f" at {start_time_obj.strftime('%I:%M %p')}"
        
        # Format alternatives
        alt_sections = []
        
        # Next available days
        next_days = alternatives.get('next_days', [])
        if next_days:
            days_list = []
            for alt in next_days[:3]:
                start_dt = datetime.fromisoformat(alt['start'])
                end_dt = datetime.fromisoformat(alt['end'])
                status = "All participants available" if alt['all_available'] else "Check availability"
                days_list.append(f"{start_dt.strftime('%A, %B %d at %I:%M %p')} - {end_dt.strftime('%I:%M %p')} ({status})")
            alt_sections.append(("Different Days", days_list))
        
        # Time adjustments
        time_adj = alternatives.get('time_adjustments', [])
        if time_adj:
            earlier = [t for t in time_adj if t['type'] == 'earlier'][:2]
            later = [t for t in time_adj if t['type'] == 'later'][:2]
            times_list = []
            if earlier:
                for alt in earlier:
                    start_dt = datetime.fromisoformat(alt['start'])
                    end_dt = datetime.fromisoformat(alt['end'])
                    times_list.append(f"Earlier: {start_dt.strftime('%I:%M %p')} - {end_dt.strftime('%I:%M %p')}")
            if later:
                for alt in later:
                    start_dt = datetime.fromisoformat(alt['start'])
                    end_dt = datetime.fromisoformat(alt['end'])
                    times_list.append(f"Later: {start_dt.strftime('%I:%M %p')} - {end_dt.strftime('%I:%M %p')}")
            if times_list:
                alt_sections.append(("Different Times", times_list))
        
        # Partial attendance
        partial = alternatives.get('partial_attendance', [])
        if partial:
            partial_info = []
            for alt in partial[:1]:
                start_dt = datetime.fromisoformat(alt['start'])
                end_dt = datetime.fromisoformat(alt['end'])
                available = ', '.join(alt['available'])
                unavailable = ', '.join(alt['unavailable'])
                partial_info.append(f"{start_dt.strftime('%I:%M %p')} - {end_dt.strftime('%I:%M %p')} (Available: {available}, Unavailable: {unavailable})")
            alt_sections.append(("Partial Attendance", partial_info))
        
        # Duration flexibility
        duration_flex = alternatives.get('duration_flexibility', [])
        if duration_flex:
            dur_list = [f"{alt['duration_minutes']} minutes (saves {alt['reduction']} minutes)" for alt in duration_flex]
            alt_sections.append(("Shorter Duration", dur_list))
        
        prompt = f"""You are a helpful scheduling assistant. No perfect slots were found, but you have alternatives.

Your Request:
- Duration: {params.get('duration_minutes', 30)} minutes
- Participants: {', '.join([organizer_id] + participant_ids)}
- Requested: {req_details if req_details else 'Not specified'}

Available Alternatives:
{chr(10).join(f"{section[0]}:{chr(10)}{chr(10).join(f'  - {item}' for item in section[1])}" for section in alt_sections)}

Generate an empathetic, solution-oriented response that:
1. Acknowledges no perfect match was found (use ❌ emoji sparingly)
2. Briefly summarizes their request
3. Explains why it didn't work (be concise and understanding)
4. Presents alternatives in organized sections:
   - 📅 Different days (if available)
   - ⏰ Different times (if available)
   - 👥 Partial attendance (if available)
   - ⏱️ Shorter duration (if available)
5. For each alternative, include:
   - Clear, readable format
   - Action prompts (e.g., "Say: 'Try Monday'")
   - Status indicators (✅/⚠️)
6. End with an encouraging question about what they'd like to do

Use markdown formatting. Be empathetic, positive, and helpful. Focus on solutions, not problems.

Generate the response now:"""

        response_text = await llm.ainvoke([HumanMessage(content=prompt)])
        ai_message = AIMessage(content=response_text.content)
        return {"messages": [ai_message]}
    
    # Case 3: Error or no results (fallback)
    else:
        prompt = f"""You are a helpful scheduling assistant. Something unexpected happened - no slots or alternatives were generated.

Request details:
- Participants: {', '.join([organizer_id] + participant_ids)}
- Duration: {params.get('duration_minutes', 30)} minutes

Generate a friendly, apologetic response that:
1. Acknowledges something went wrong
2. Suggests possible reasons:
   - No common working days
   - All participants busy
   - Unexpected scheduling conflict
3. Encourages them to:
   - Try a different day or time range
   - Provide more flexible timing
   - Contact you with specific preferences

Be apologetic but helpful. Use markdown formatting. Keep it brief and actionable.

Generate the response now:"""

        response_text = await llm.ainvoke([HumanMessage(content=prompt)])
        ai_message = AIMessage(content=response_text.content)
        return {"messages": [ai_message]}

# Conditional routing function
def should_fetch_data(state: AgentState) -> str:
    """
    Determine if we should fetch calendars/working hours, confirm schedule, or skip directly to respond.
    Skip scheduling logic for casual chat, error states, or missing critical info.
    """
    extracted_info = state.get('extracted_info', {})
    intent = extracted_info.get('intent', 'chat')
    missing_info = extracted_info.get('missing_info', [])
    
    # Check if user wants to confirm a previously suggested schedule
    if intent == 'confirm_schedule':
        logger.info(f"Intent 'confirm_schedule' - routing to confirm_schedule node")
        return "confirm_schedule"
    
    # EDGE CASE 3: If schedule_meeting but missing critical info, ask user first
    if intent == 'schedule_meeting' and missing_info:
        logger.info(f"Missing info: {missing_info} - asking user before scheduling")
        return "respond"
    
    # Only fetch data for scheduling-related intents
    if intent in ['schedule_meeting', 'query_availability']:
        logger.info(f"Intent '{intent}' - proceeding with data fetch")
        return "fetch_calendars"
    else:
        logger.info(f"Intent '{intent}' - skipping to respond")
        return "respond"

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("parse_input", parse_input)
workflow.add_node("fetch_calendars", fetch_calendars_node)
workflow.add_node("fetch_working_hours", fetch_working_hours_node)
workflow.add_node("find_slots", find_slots_node)
workflow.add_node("select_best_slot", select_best_slot)
workflow.add_node("confirm_schedule", confirm_schedule_node)
workflow.add_node("respond", respond)

# Set entry point
workflow.set_entry_point("parse_input")

# Add conditional edge from parse_input
workflow.add_conditional_edges(
    "parse_input",
    should_fetch_data,
    {
        "fetch_calendars": "fetch_calendars",
        "confirm_schedule": "confirm_schedule",
        "respond": "respond"
    }
)

# Add linear edges for scheduling flow
# parse_input -> fetch_calendars -> fetch_working_hours -> find_slots -> select_best_slot -> respond
workflow.add_edge("fetch_calendars", "fetch_working_hours")
workflow.add_edge("fetch_working_hours", "find_slots")
workflow.add_edge("find_slots", "select_best_slot")
workflow.add_edge("select_best_slot", "respond")

# Add edge for confirmation flow
# parse_input -> confirm_schedule -> respond
workflow.add_edge("confirm_schedule", "respond")

# All paths end at respond
workflow.add_edge("respond", END)

# Compile the graph
graph = workflow.compile()

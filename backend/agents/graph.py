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
    intent: str = Field(description="The user's primary intent, e.g., 'schedule_meeting', 'query_availability', 'cancel_meeting', 'chat'.")
    parameters: MeetingParameters = Field(default_factory=MeetingParameters, description="Structured parameters extracted from the conversation.")
    constraints: List[Constraint] = Field(default_factory=list, description="Any constraints mentioned by the user.")
    missing_info: List[str] = Field(default_factory=list, description="List of information needed to fulfill the request but currently missing.")

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
    candidate_slots: List[dict]
    proposed_slot: Optional[dict]
    
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
        ("system", """You are an intelligent scheduling assistant parser. 
        Your job is to extract meeting details from the conversation.
        Current Time: {current_time} (Singapore/Manila Time, UTC+8)
        Organizer: {organizer}
        Participants already selected: {participants}
        
        Analyze the conversation and extract:
        1. Intent: What does the user want? (e.g., schedule_meeting, query_availability)
        2. Parameters: Extract specific values into the structured format. 
           - Convert relative dates (today, tomorrow) to YYYY-MM-DD based on Current Time.
           - Convert relative times (2pm, 14:00) to HH:MM (24h).
           - IMPORTANT: Extract duration from phrases like:
             * "2 hour meeting" -> duration_minutes: 120
             * "30 minute meeting" -> duration_minutes: 30
             * "1 hour meeting" -> duration_minutes: 60
             * "half hour meeting" -> duration_minutes: 30
           - If BOTH start_time AND end_time are provided (e.g., "between 2pm and 4pm"), calculate duration_minutes from the difference.
           - If only duration is mentioned (e.g., "1 hour meeting"), set duration_minutes directly.
           - Default to 30 minutes if no duration specified.
        3. Constraints: Any preferences or blockers.
           - Example: "I'm not available Tuesday mornings" -> type='unavailable', day='Tuesday', start='08:00', end='12:00'
           - Example: "I prefer afternoons" -> type='preference', start='12:00', end='17:00'
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
        result = await chain.ainvoke({
            "organizer": organizer_id, 
            "messages": messages, 
            "participants": str(participant_ids),
            "current_time": current_time_str
        })
        
        extracted_dict = result.model_dump()
        params = extracted_dict.get('parameters', {})
        
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

# Define the Simplified Responder Node (For Testing)
async def respond(state: AgentState):
    """
    Simplified responder that returns structured JSON for testing.
    Shows what data was extracted, fetched, and computed.
    """
    from langchain_core.messages import AIMessage
    
    organizer_id = state.get('organizer_id', 'Unknown')
    participant_ids = state.get('participant_ids', [])
    extracted_info = state.get('extracted_info', {})
    all_calendars = state.get('all_calendars', {})
    all_working_hours = state.get('all_working_hours', {})
    candidate_slots = state.get('candidate_slots', [])
    debug_info = state.get('debug_info', [])
    
    # Build a structured response for testing
    response_data = {
        "organizer": organizer_id,
        "participants": participant_ids,
        "extracted_parameters": extracted_info.get('parameters', {}),
        "extracted_intent": extracted_info.get('intent', 'unknown'),
        "calendars_fetched": {
            user: len(meetings) for user, meetings in all_calendars.items()
        },
        "working_hours_fetched": {
            user: f"{hours.get('start')}-{hours.get('end')} on {hours.get('working_days')}"
            for user, hours in all_working_hours.items()
        },
        "candidate_slots_found": len(candidate_slots),
        "top_slots": candidate_slots[:5],  # Show top 5
        "debug_log": debug_info
    }
    
    # Format as readable text
    response_text = "## Scheduling Agent Debug Output\n\n"
    response_text += f"**Organizer:** {organizer_id}\n"
    response_text += f"**Participants:** {', '.join(participant_ids) if participant_ids else 'None'}\n\n"
    
    response_text += "### Extracted Information\n"
    response_text += f"- **Intent:** {extracted_info.get('intent', 'unknown')}\n"
    params = extracted_info.get('parameters', {})
    if params:
        response_text += f"- **Duration:** {params.get('duration_minutes', 'N/A')} minutes\n"
        response_text += f"- **Date:** {params.get('date', 'Not specified')}\n"
        response_text += f"- **Time Window:** {params.get('start_time', 'N/A')} - {params.get('end_time', 'N/A')}\n"
    response_text += "\n"
    
    response_text += "### Calendars Fetched\n"
    for user, count in response_data["calendars_fetched"].items():
        response_text += f"- **{user}:** {count} existing meetings\n"
    response_text += "\n"
    
    # In the respond function, around line 710:
    response_text += "\n### Debug Log (Key Events)\n"
    # Show entries with ✅ first, then the last few entries
    important_entries = [e for e in debug_info if '✅' in e or '❌' in e or 'VALID SLOT' in e or 'Current time' in e or 'Generated' in e or 'Conflict' in e]
    # Add 'Conflict' to the filter ↑
    for log_entry in important_entries[:20]:  # Show more entries
        response_text += f"- {log_entry}\n"
    
    response_text += f"### Available Slots Found: {len(candidate_slots)}\n"
    if candidate_slots:
        response_text += "Top available slots:\n"
        for i, slot in enumerate(candidate_slots[:5], 1):
            start_dt = datetime.fromisoformat(slot['start'])
            end_dt = datetime.fromisoformat(slot['end'])
            response_text += f"{i}. **{start_dt.strftime('%A, %B %d at %I:%M %p')}** - {end_dt.strftime('%I:%M %p')}\n"
    else:
        response_text += "❌ No available slots found that work for all participants.\n"
    
    response_text += "\n### Debug Log (Key Events)\n"
    # Show entries with ✅ first, then the last few entries
    important_entries = [e for e in debug_info if '✅' in e or 'VALID SLOT' in e or 'Current time' in e or 'Generated' in e]
    for log_entry in important_entries:
        response_text += f"- {log_entry}\n"
    
    response_text += "\n### Recent Activity\n"
    for log_entry in debug_info[-10:]:
        response_text += f"- {log_entry}\n"
    
    ai_message = AIMessage(content=response_text)
    
    return {"messages": [ai_message]}

# Conditional routing function
def should_fetch_data(state: AgentState) -> str:
    """
    Determine if we should fetch calendars/working hours or skip directly to respond.
    Skip scheduling logic for casual chat or error states.
    """
    extracted_info = state.get('extracted_info', {})
    intent = extracted_info.get('intent', 'chat')
    
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
workflow.add_node("respond", respond)

# Set entry point
workflow.set_entry_point("parse_input")

# Add conditional edge from parse_input
workflow.add_conditional_edges(
    "parse_input",
    should_fetch_data,
    {
        "fetch_calendars": "fetch_calendars",
        "respond": "respond"
    }
)

# Add linear edges for scheduling flow
# parse_input -> fetch_calendars -> fetch_working_hours -> find_slots -> respond
workflow.add_edge("fetch_calendars", "fetch_working_hours")
workflow.add_edge("fetch_working_hours", "find_slots")
workflow.add_edge("find_slots", "respond")
workflow.add_edge("respond", END)

# Compile the graph
graph = workflow.compile()

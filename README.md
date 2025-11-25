# AI Scheduling Agent

An intelligent, agentic scheduling assistant that streamlines meeting coordination using AI. It handles the back-and-forth of finding a time that works for everyone, managing conflicts, and ensuring priority participants are available.

## 🚀 Features

### 🤖 AI Scheduling Agent
- **Natural Language Interface**: Chat with the agent just like you would with a human assistant (e.g., "Schedule a meeting with Alice and Bob next Tuesday afternoon").
- **Context Awareness**: The agent understands follow-up requests and maintains context throughout the conversation.
- **Smart Slot Ranking**: Automatically ranks potential time slots based on:
  - User preferences (time of day, specific dates).
  - Participant availability.
  - Minimizing conflicts.

### 📅 Dashboard & Management
- **Meeting Alerts**: A dedicated dashboard to view incoming meeting requests and manage your own proposals.
- **Calendar View**: Visual representation of your schedule to quickly spot free time.
- **Priority Participants**: Mark specific attendees as "Priority" to ensure the meeting doesn't go ahead without them.

### 🔄 Automated Rescheduling Workflow
- **Conflict Handling**: If a participant rejects a meeting, the system automatically triggers a rescheduling agent.
- **Smart Alternatives**: The agent analyzes calendars again to suggest the best alternative times.
- **Organizer Control**: The organizer receives a curated list of new options to approve, reject, or negotiate.

## 🛠️ Technology Stack

- **Frontend**: [Next.js](https://nextjs.org/) (App Router), Tailwind CSS, Shadcn UI
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **AI/Agents**: [LangGraph](https://python.langchain.com/docs/langgraph), Google Gemini 2.5 Flash (via `langchain-google-genai`)
- **Database**: [Supabase](https://supabase.com/) (PostgreSQL)

## 🔄 How It Works

1.  **Request**: You ask the agent to schedule a meeting via the chat interface.
2.  **Analysis**: The agent (powered by LangGraph) parses your intent, checks the Supabase database for all participants' availability and their working days and hours.
3.  **Proposal**: The agent proposes the best available time slots.
4.  **Confirmation**: Once you select a slot, a **Meeting Proposal** is created.
5.  **Notification**: Participants receive an alert in their dashboard to either Accept the proposal or Decline it.
6.  **Finalization**:
    -   **If everyone accepts**: The meeting is finalized and added to everyone's calendar.
    -   **If someone declines**:
        -   **Priority Participants**: If a *priority* participant declines, the meeting CANNOT be finalized. The **Rescheduling Agent** is triggered to find new options.
        -   **Non-Priority Participants**: If only non-priority participants decline, the organizer has the option to **Push Through** (Majority Vote) and finalize the meeting with the available attendees.
            > **Majority Calculation**: The meeting can proceed if `(Accepted Participants + Organizer) > (Total Invited + Organizer) / 2`.

## 🧠 Smart Slot Scoring System

The agent ranks potential meeting slots using a weighted scoring algorithm to ensure the best possible time is proposed.

| Criterion | Points | Description |
| :--- | :--- | :--- |
| **Preferred Time** | **0-100** | Highest priority. Exact matches get max points; slots within 1-2 hours get partial points. |
| **Full Attendance** | **0-50** | Bonus if *all* invited participants are available. |
| **Day Proximity** | **0-30** | Preference for the specific day requested, or close to it (next day, within 3 days). |
| **Time of Day** | **0-20** | Quality of the hour. Mornings (9-11) and Afternoons (14-16) are preferred over lunch or late hours. |

**Total Score** = Preferred Time + Full Attendance + Day Proximity + Time of Day

### 💡 Scoring Example

**Scenario**: User asks for *"Tuesday at 2 PM"*.

**Option A: Tuesday @ 2:00 PM** (Perfect Match)
- Preferred Time: **100 pts** (Exact match)
- Full Attendance: **50 pts** (Everyone available)
- Day Proximity: **30 pts** (Same day)
- Time of Day: **15 pts** (Afternoon slot)
- **Total: 195 points** 🏆

**Option B: Wednesday @ 10:00 AM** (Alternative)
- Preferred Time: **0 pts** (Different time)
- Full Attendance: **50 pts** (Everyone available)
- Day Proximity: **20 pts** (Next day)
- Time of Day: **20 pts** (Morning slot)
- **Total: 90 points**

## 📂 Project Structure

```
├── backend/
│   ├── agents/
│   │   ├── graph.py       # Main scheduling agent logic (LangGraph)
│   │   └── resched.py     # Rescheduling specific agent logic
│   ├── main.py            # FastAPI entry point and API endpoints
│   └── ...
├── components/
│   ├── dashboard/         # Dashboard UI components (MeetingAlerts, etc.)
│   ├── calendar/          # Calendar visualization components
│   └── ...
├── app/                   # Next.js App Router pages
```

## 🚦 Getting Started

### 1. Frontend Setup

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`.

### 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install python dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
python main.py
```

The backend will be available at `http://localhost:8000`.

## 🔐 Environment Variables

Ensure you have the following environment variables configured:

**Frontend (`.env.local`)**:
```
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

**Backend (`backend/.env`)**:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_role_key
GOOGLE_API_KEY=your_gemini_api_key
```

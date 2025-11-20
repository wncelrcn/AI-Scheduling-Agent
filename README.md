# AI-Scheduling-Agent

An agentic AI-powered scheduling assistant that helps users coordinate meetings based on preferences and availability.

## Project Structure

- **Frontend**: Next.js (App Router)
- **Backend**: FastAPI (Python)
- **Database**: Supabase
- **UI**: Tailwind CSS + Custom Components

## Getting Started

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

## Environment Variables

Make sure you have your Supabase credentials configured in `.env.local` (frontend) and `.env` (backend) if needed for direct DB access.

## Features

- **Dashboard**: View your calendar.
- **Chat Interface**: Interact with the AI agent to schedule meetings.
- **Agent**: Parses natural language to check availability and coordinate schedules (in progress).

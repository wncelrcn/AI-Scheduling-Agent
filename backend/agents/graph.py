from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

# Define the state of our agent
class AgentState(TypedDict):
    # The 'add_messages' annotation ensures that new messages are appended to the existing list
    messages: Annotated[List[BaseMessage], add_messages]
    username: str

# Initialize the LLM
# Ensure GOOGLE_API_KEY is set in your environment variables
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Define the system prompt
SYSTEM_TEMPLATE = """You are an intelligent scheduling assistant for {username}.
Your goal is to help the user manage their calendar, schedule meetings, and coordinate with others.

Current capabilities:
- Chatting with the user to understand their preferences.
- (Coming soon) Checking availability in the database.
- (Coming soon) Booking meetings.

When the user talks about their schedule, listen carefully to constraints like "I'm not available on Fridays" or "I prefer morning meetings".
"""

def call_model(state: AgentState):
    messages = state['messages']
    username = state.get('username', 'User')
    
    # Prepare the prompt with the username
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({"username": username, "messages": messages})
    
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)

# Set entry point
workflow.set_entry_point("agent")

# Add edges
workflow.add_edge("agent", END)

# Compile the graph
graph = workflow.compile()


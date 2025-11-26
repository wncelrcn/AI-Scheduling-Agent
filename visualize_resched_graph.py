from backend.agents.resched import resched_graph

if __name__ == "__main__":
    try:
        print(resched_graph.get_graph().draw_mermaid())
    except Exception as exc:
        print(f"Error generating rescheduling graph: {exc}")
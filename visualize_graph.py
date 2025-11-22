from backend.agents.graph import graph

try:
    # Print the mermaid graph definition directly
    print(graph.get_graph().draw_mermaid())
except Exception as e:
    print(f"Error generating graph: {e}")

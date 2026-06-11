from langgraph.graph import StateGraph, END
from agent.nodes import router_node, rag_qa_node, risk_flag_node
from agent.custom_agent_state import AgentState

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router",    router_node)
    g.add_node("rag_qa",    rag_qa_node)
    g.add_node("risk_flag", risk_flag_node)

    g.set_entry_point("router")

    g.add_conditional_edges("router", lambda s: s["intent"], {
        "qa":         "rag_qa",
        "risk_flag":  "risk_flag",
    })

    g.add_edge("rag_qa",    END)
    g.add_edge("risk_flag", END)

    return g.compile()
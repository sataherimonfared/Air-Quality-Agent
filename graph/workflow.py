from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import (
    validate_readings,
    detect_anomalies,
    alert_decision,
    classify_air_quality,
    generate_trend_summary,
    nl_summary,
    critique_summary
)

def create_workflow():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("validate_readings", validate_readings)
    workflow.add_node("detect_anomalies", detect_anomalies)
    workflow.add_node("classify_air_quality", classify_air_quality)
    workflow.add_node("alert_decision", alert_decision)
    workflow.add_node("generate_trend_summary", generate_trend_summary)
    workflow.add_node("nl_summary", nl_summary)
    workflow.add_node("critique_summary", critique_summary)

    # Set entry point
    workflow.set_entry_point("validate_readings")

    # Connect nodes
    workflow.add_edge("validate_readings", "detect_anomalies")
    workflow.add_edge("detect_anomalies", "classify_air_quality")
    workflow.add_edge("classify_air_quality", "alert_decision")
    workflow.add_edge("alert_decision", "generate_trend_summary")
    workflow.add_edge("generate_trend_summary", "nl_summary")
    workflow.add_edge("nl_summary", "critique_summary")

    # Conditional cycle for self-correction
    def should_continue_refining(state: AgentState):
        if state["feedback"] == "Good":
            return "finish"
        return "refine"

    workflow.add_conditional_edges(
        "critique_summary",
        should_continue_refining,
        {
            "finish": END,
            "refine": "nl_summary"
        }
    )

    # Set up memory
    memory = MemorySaver()

    # Compile with persistence and interrupt
    # We interrupt before 'alert_decision' to wait for human approval if needed
    return workflow.compile(
        checkpointer=memory,
        interrupt_before=["alert_decision"]
    )

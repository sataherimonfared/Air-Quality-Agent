from typing import TypedDict, List, Annotated
import pandas as pd

class AgentState(TypedDict):
    # The raw data (Will be stored as serializable list of dicts)
    data: List[dict]
    # List of indices where anomalies were detected
    anomalies: List[int]
    # Threshold for triggering alerts (percentage of anomalies)
    anomaly_threshold: float
    # Classification result (e.g., "Good", "Hazardous")
    air_quality_class: str
    # Summary of trends
    trend_summary: dict
    # Final AI-generated natural language summary
    final_summary: str
    # Whether an alert was triggered
    alert_triggered: bool
    # Human-in-the-loop approval status
    approved: bool
    # Feedback from the critique node
    feedback: str
    # Iteration count for self-correction cycles
    iterations: int
    # Output from tool calls
    tool_outputs: List[str]

import pandas as pd
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from .state import AgentState

# Using ChatOllama for better tool-calling support
llm = ChatOllama(model="mistral:7b", base_url="http://localhost:11434")

@tool
def get_health_guidelines(aqi_category: str) -> str:
    """Provides specific health recommendations based on the Air Quality Category."""
    guidelines = {
        "Good": "Air quality is satisfactory. Enjoy outdoor activities.",
        "Moderate": "Sensitive individuals should consider reducing prolonged outdoor exertion.",
        "Unhealthy for Sensitive Groups": "Children, active adults, and people with respiratory disease should limit outdoor exertion.",
        "Unhealthy": "Everyone should limit prolonged outdoor exertion.",
        "Hazardous": "Health warning of emergency conditions. The entire population is more likely to be affected."
    }
    return guidelines.get(aqi_category, "No specific guidelines available.")

tools = [get_health_guidelines]
llm_with_tools = llm.bind_tools(tools)

def get_aqi_label(pm25):
    if pm25 < 12: return "Good"
    if pm25 < 35: return "Moderate"
    if pm25 < 55: return "Unhealthy for Sensitive Groups"
    if pm25 < 150: return "Unhealthy"
    return "Hazardous"

def validate_readings(state: AgentState) -> dict:
    print("--- Validating Readings ---")
    df = pd.DataFrame(state["data"])
    
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Date"] = df["Timestamp"].dt.date
    else:
        # Fallback if no timestamp
        df["Date"] = "Unknown"
    
    # Simple validation: Convert to numeric, handle NAs
    df["PM2.5 (µg/m³)"] = pd.to_numeric(df["PM2.5 (µg/m³)"], errors='coerce')
    df["PM10 (µg/m³)"] = pd.to_numeric(df["PM10 (µg/m³)"], errors='coerce')
    
    # Fill NAs
    df["PM2.5 (µg/m³)" ] = df["PM2.5 (µg/m³)"].fillna(df["PM2.5 (µg/m³)"].mean())
    df["PM10 (µg/m³)"] = df["PM10 (µg/m³)"].fillna(df["PM10 (µg/m³)"].mean())
    
    # Store with Date for daily analysis
    return {"data": df.reset_index().to_dict('records')}

def detect_anomalies(state: AgentState) -> dict:
    print("--- Detecting Anomalies ---")
    df = pd.DataFrame(state["data"])
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df.set_index("Timestamp", inplace=True)
    
    # Simple Z-score anomaly detection
    z_scores = (df["PM2.5 (µg/m³)"] - df["PM2.5 (µg/m³)"].mean()) / df["PM2.5 (µg/m³)"].std()
    
    # Find indices where z-score is high
    anomaly_mask = abs(z_scores) > 3
    anomalies = df[anomaly_mask].index.tolist()
    
    # Ensure anomalies are strings if they are timestamps
    anomalies = [str(a) for a in anomalies]
    
    return {"anomalies": anomalies}

def alert_decision(state: AgentState) -> dict:
    print("--- Deciding on Alert ---")
    anomalies = state["anomalies"]
    data_len = len(state["data"])
    
    anomaly_ratio = len(anomalies) / data_len if data_len > 0 else 0
    alert = anomaly_ratio > state["anomaly_threshold"]
    
    return {"alert_triggered": alert}

def classify_air_quality(state: AgentState) -> dict:
    print("--- Classifying Air Quality (Daily Aggregation) ---")
    df = pd.DataFrame(state["data"])
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Date"] = df["Timestamp"].dt.date
    
    # Calculate daily averages
    daily_avg = df.groupby("Date")["PM2.5 (µg/m³)"].mean()
    
    # Count frequency of each category
    categories = daily_avg.apply(get_aqi_label).value_counts()
    primary_category = categories.idxmax()
    
    # Detailed breakdown for the LLM
    breakdown = categories.to_dict()
    res = f"{primary_category} (Frequency: {breakdown})"
    
    # Store the daily averages back for potential use in trends
    return {"air_quality_class": res}

def generate_trend_summary(state: AgentState) -> dict:
    print("--- Generating Trend Summary ---")
    df = pd.DataFrame(state["data"])
    summary = {
        "mean_pm25": float(df["PM2.5 (µg/m³)"].mean()),
        "max_pm25": float(df["PM2.5 (µg/m³)"].max()),
        "min_pm25": float(df["PM2.5 (µg/m³)"].min()),
        "mean_pm10": float(df["PM10 (µg/m³)"].mean())
    }
    return {"trend_summary": summary}

def nl_summary(state: AgentState) -> dict:
    print("--- Generating AI Summary (Tool-Aware) ---")
    trends = state["trend_summary"]
    classification = state["air_quality_class"]
    alert_status = "TRIGGERED" if state.get("alert_triggered") else "Not Triggered"
    feedback = state.get("feedback", "")
    tool_outputs = state.get("tool_outputs", [])
    
    prompt = f"""
    Analyze the following air quality report:
    - Average PM2.5: {trends['mean_pm25']:.2f}
    - Max PM2.5: {trends['max_pm25']:.2f}
    - Average PM10: {trends['mean_pm10']:.2f}
    - Classification: {classification}
    - Alert Status: {alert_status}
    
    {f"Health Guidelines Tool Output: {tool_outputs}" if tool_outputs else "You haven't checked the official health guidelines yet. If you need specific safety recommendations for this classification, mention that you are calling the tool."}
    {f"Previous Feedback for improvement: {feedback}" if feedback else ""}
    
    Provide a professional summary. If you have the health guidelines, include them. 
    Otherwise, if the status is not 'Good', you MUST request the 'get_health_guidelines' tool.
    """
    
    try:
        # Extract the primary category for the tool call
        cat_for_tool = classification.split(" (")[0]
        
        # We check if we already have tool output or if it's the first time
        if not tool_outputs and cat_for_tool != "Good":
            print(f"   (Agent decided it needs health guidelines tool for: {cat_for_tool})")
            res = get_health_guidelines.invoke({"aqi_category": cat_for_tool})
            tool_outputs = [res]
            # Call LLM again with the new info
            prompt += f"\n\nNew information from tool: {res}"
            
        response = llm.invoke(prompt).content
    except Exception as e:
        response = f"AI summary currently unavailable. (Error: {str(e)})"
        
    return {"final_summary": response, "iterations": state.get("iterations", 0) + 1, "tool_outputs": tool_outputs}

def critique_summary(state: AgentState) -> dict:
    print("--- Critiquing Summary ---")
    summary = state["final_summary"]
    iterations = state["iterations"]
    
    # Simple rule-based critique for demo purposes
    # You could use another LLM call here
    if len(summary.split()) < 30 and iterations < 3:
        return {"feedback": "The summary is too short. Please provide more detail and health recommendations."}
    
    return {"feedback": "Good"}

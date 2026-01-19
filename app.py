import streamlit as st
import pandas as pd
import glob
import os
import plotly.express as px
from graph.workflow import create_workflow

st.set_page_config(page_title="Air Quality Insight & Alert Agent", layout="wide")

st.title("🌫️ Air Quality Insight & Alert Agent")
st.markdown("""
Monitor, analyze, and detect anomalies in air quality data using **LangGraph** & **Ollama**.
""")

# Sidebar settings
st.sidebar.header("Settings")
data_files = glob.glob("data/*.csv")
selected_file = st.sidebar.selectbox("Select Dataset", data_files)
anomaly_threshold = st.sidebar.slider("Anomaly Alert Threshold (%)", 0.0, 5.0, 1.0) / 100

# Initialize session state for workflow
if "graph_app" not in st.session_state:
    st.session_state.graph_app = create_workflow()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"
if "current_result" not in st.session_state:
    st.session_state.current_result = None

if selected_file:
    df = pd.read_csv(selected_file)
    st.sidebar.info(f"Loaded {len(df)} records.")

    if st.sidebar.button("Run New Analysis"):
        st.session_state.current_result = None
        # Start fresh run
        initial_state = {
            "data": df.to_dict('records'), # Convert to serializable format
            "anomalies": [],
            "anomaly_threshold": anomaly_threshold,
            "air_quality_class": "Unknown",
            "trend_summary": {},
            "final_summary": "",
            "alert_triggered": False,
            "approved": False,
            "feedback": "",
            "iterations": 0,
            "tool_outputs": []
        }
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        with st.spinner("Agent initializing..."):
            # This will run until the interrupt before alert_decision
            for event in st.session_state.graph_app.stream(initial_state, config):
                pass
            
            st.session_state.current_result = st.session_state.graph_app.get_state(config).values

    # Handle Human-in-the-loop if interrupted
    if st.session_state.current_result:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        state = st.session_state.graph_app.get_state(config)
        
        if state.next: # Check if the graph is currently waiting/interrupted
            st.warning(f"⏸️ Agent is waiting for approval before: **{state.next[0]}**")
            col_app, col_rej = st.columns(2)
            if col_app.button("✅ Approve & Continue"):
                with st.spinner("Agent resuming..."):
                    # Resume execution
                    for event in st.session_state.graph_app.stream(None, config):
                        pass
                    st.session_state.current_result = st.session_state.graph_app.get_state(config).values
                    st.rerun()

    if st.session_state.current_result:
        result = st.session_state.current_result
        # Convert data back to DataFrame for plotting
        plot_df = pd.DataFrame(result["data"])
        if "Timestamp" in plot_df.columns:
            plot_df["Timestamp"] = pd.to_datetime(plot_df["Timestamp"])
            plot_df.set_index("Timestamp", inplace=True)
        
        # Display Results
        st.divider()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Air Quality Trends")
            fig = px.line(plot_df, x=plot_df.index, y=["PM2.5 (µg/m³)", "PM10 (µg/m³)"], 
                         title="PM2.5 and PM10 Time-Series")
            # Highlight anomalies
            if result["anomalies"]:
                # Result anomalies are strings (Timestamps), convert back for indexing
                anomaly_indices = pd.to_datetime(result["anomalies"]) if "Timestamp" in plot_df.index.dtype.name or True else result["anomalies"]
                try:
                    anomaly_df = plot_df.loc[anomaly_indices]
                    fig.add_scatter(x=anomaly_df.index, y=anomaly_df["PM2.5 (µg/m³)"], 
                                   mode='markers', name='Anomalies', marker=dict(color='red', size=8))
                except Exception as e:
                    st.error(f"Error plotting anomalies: {e}")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("🤖 AI Insights")
            if result.get("alert_triggered"):
                st.error("🚨 ALERT: Unusual air quality spikes detected!")
            
            st.metric("Classification", result["air_quality_class"])
            
            if "final_summary" in result and result["final_summary"]:
                st.write(result["final_summary"])
                if result.get("iterations") > 1:
                    st.caption(f"✨ Summary refined in {result['iterations']} iterations.")
            else:
                st.info("Analysis in progress... Summary will appear after approval.")
        
        st.divider()
        st.subheader("📊 Detailed Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Average PM2.5", f"{result['trend_summary'].get('mean_pm25', 0):.2f}")
        m2.metric("Max PM2.5", f"{result['trend_summary'].get('max_pm25', 0):.2f}")
        m3.metric("Average PM10", f"{result['trend_summary'].get('mean_pm10', 0):.2f}")
        m4.metric("Anomaly Count", len(result["anomalies"]))

else:
    st.warning("Please ensure data files are in the `data/` directory.")



#%% 

import os
import pandas as pd
from typing import TypedDict, Literal, Optional, List
from langgraph.graph import StateGraph, END

# --- Import Nodes from utils ---
from utils import (
    analyze_request_node, 
    model_selection_node, 
    simulation_node, 
    data_processing_node, 
    analysis_agent_node
)

# --- 1. Define the Agent State ---
class AgentState(TypedDict):
    user_prompt: str     
    analysis_result: Optional[dict]
    selected_models: Optional[List[dict]]  
    simulation_output: Optional[List[dict]]
    dataframes: Optional[dict] 
    final_message: Optional[str]
    current_status: Optional[str]        

# --- 2. Define Routing Logic ---
def route_request(state: AgentState) -> Literal["model_selector", "end"]:
    if state.get("final_message"): return "end"
    
    if state.get("analysis_result", {}).get("is_relevant"):
        print(f"\n[Router] Request Validated: {state['analysis_result']['verification']}")
        return "model_selector"
    
    state["final_message"] = "I am not designed for this. I specialize in OpenSim."
    return "end"

def route_model_selection(state: AgentState) -> Literal["simulator", "end"]:
    if state.get("final_message"): return "end"
    return "simulator"

# --- 3. Build the Graph ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("analyzer", analyze_request_node)
workflow.add_node("model_selector", model_selection_node)
workflow.add_node("simulator", simulation_node)
workflow.add_node("processor", data_processing_node)  # <--- NEW
workflow.add_node("analyst", analysis_agent_node)     # <--- NEW

# Set Entry Point
workflow.set_entry_point("analyzer")

# Add Edges
# 1. Analyze -> Select
workflow.add_conditional_edges("analyzer", route_request, {"model_selector": "model_selector", "end": END})

# 2. Select -> Simulate
workflow.add_conditional_edges("model_selector", route_model_selection, {"simulator": "simulator", "end": END})

# 3. Simulate -> Process Data -> Analyze(Chat) -> END
workflow.add_edge("simulator", "processor")
workflow.add_edge("processor", "analyst")
workflow.add_edge("analyst", END)

# Compile
app = workflow.compile()

# --- 4. Execution Example ---
if __name__ == "__main__":
    print("Initializing OpenSim Agent...")
    
    # Example Prompt
    user_input = "Simulate standing upright for 62 to 72 kg man with height of between 170 cm - 175 cm. Analyze the compression forces on L5_S1."

    print(f"\nUser Prompt: {user_input}\n" + "="*40)

    # Invoke the Agent
    result = app.invoke({"user_prompt": user_input})
    
    print("\n" + "="*40)
    print("FINAL AGENT RESPONSE:")
    print(result.get("final_message"))




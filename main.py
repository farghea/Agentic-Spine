

#%% 

import os
import pandas as pd
from typing import TypedDict, Literal, Optional, List
from langgraph.graph import StateGraph, END

# --- Import Nodes from utils ---
from utils import (
    analyze_request_node,
    activity_router_node,
    model_selection_node,
    simulation_node,
    data_processing_node,
    analysis_agent_node,
    # Image-pipeline nodes
    input_router_node,
    image_classifier_node,
    medical_mri_node,
    sam3d_node,
    lifting_analysis_node,
)

# --- 1. Define the Agent State ---
class AgentState(TypedDict):
    # ── shared ──────────────────────────────────────────────
    user_prompt:           str
    final_message:         Optional[str]
    current_status:        Optional[str]

    # ── text-only / OpenSim path ─────────────────────────────
    analysis_result:       Optional[dict]
    routed_activities:     Optional[List[dict]]
    selected_models:       Optional[List[dict]]
    simulation_output:     Optional[List[dict]]
    dataframes:            Optional[dict]

    # ── image path ───────────────────────────────────────────
    uploaded_image_path:   Optional[str]   # local path to the saved upload
    image_type:            Optional[str]   # "medical" | "lifting"
    image_analysis_result: Optional[dict]  # full output from lifting analysis
    mri_result:            Optional[dict]  # plot_path + match info from MRI pipeline


# --- 2. Routing Logic ---

def route_input(state: AgentState) -> Literal["medical_mri", "image_classifier", "analyzer"]:
    """Branch at the very entry:
       .mha file  → medical MRI pipeline (skip classifier — we know it's medical)
       other image → image classifier → lifting or medical
       text-only  → OpenSim pipeline
    """
    path = state.get("uploaded_image_path", "") or ""
    if path.lower().endswith(".mha"):
        return "medical_mri"
    if path:
        return "image_classifier"
    return "analyzer"


def route_image_type(state: AgentState) -> Literal["sam3d_processor", "medical_mri"]:
    """After classification, choose lifting vs medical pathway."""
    if state.get("image_type") == "medical":
        return "medical_mri"
    return "sam3d_processor"


def route_request(state: AgentState) -> Literal["model_selector", "end"]:
    """Original text-only route check."""
    # MRI path: model already selected by morphology matching — always proceed
    # (must be checked BEFORE final_message to avoid short-circuiting)
    if state.get("selected_models"):
        print("[Router] MRI model pre-selected — proceeding to simulation.")
        return "model_selector"
    if state.get("final_message"):
        return "end"
    if state.get("analysis_result", {}).get("is_relevant"):
        print(f"\n[Router] Request Validated: {state['analysis_result']['verification']}")
        return "model_selector"
    state["final_message"] = "I am not designed for this. I specialize in OpenSim."
    return "end"


def route_model_selection(state: AgentState) -> Literal["simulator", "end"]:
    # If the model has been successfully selected (either by LLM or MRI),
    # always proceed to simulation.
    if state.get("selected_models"):
        print("[Router] Model selected — proceeding to simulator.")
        return "simulator"
    if state.get("final_message"):
        return "end"
    return "simulator"


# --- 3. Build the Graph ---
workflow = StateGraph(AgentState)

# ── Nodes ────────────────────────────────────────────────────────────────────
workflow.add_node("input_router",     input_router_node)
workflow.add_node("image_classifier",  image_classifier_node)
workflow.add_node("medical_mri",       medical_mri_node)
workflow.add_node("sam3d_processor",   sam3d_node)
workflow.add_node("lifting_analyzer",  lifting_analysis_node)

workflow.add_node("analyzer",         analyze_request_node)
workflow.add_node("activity_router",  activity_router_node)
workflow.add_node("model_selector",   model_selection_node)
workflow.add_node("simulator",        simulation_node)
workflow.add_node("processor",        data_processing_node)
workflow.add_node("analyst",          analysis_agent_node)

# ── Entry point ───────────────────────────────────────────────────────────────
workflow.set_entry_point("input_router")

# ── Image pipeline edges ──────────────────────────────────────────────────────
workflow.add_conditional_edges(
    "input_router",
    route_input,
    {"medical_mri": "medical_mri", "image_classifier": "image_classifier", "analyzer": "analyzer"},
)
workflow.add_conditional_edges(
    "image_classifier",
    route_image_type,
    {"sam3d_processor": "sam3d_processor", "medical_mri": "medical_mri"},
)
# MRI pipeline feeds into the analyzer so the user's prompt is parsed for
# activities, then proceeds through the full OpenSim chain.
# model_selection_node has a guard that skips LLM selection when selected_models
# is already populated by the MRI pipeline.
workflow.add_edge("medical_mri",     "analyzer")
workflow.add_edge("sam3d_processor", "lifting_analyzer")
workflow.add_edge("lifting_analyzer", END)

# ── Text-only / OpenSim edges (unchanged) ────────────────────────────────────
workflow.add_conditional_edges(
    "analyzer",
    route_request,
    {"model_selector": "activity_router", "end": END},
)
workflow.add_edge("activity_router", "model_selector")
workflow.add_conditional_edges(
    "model_selector",
    route_model_selection,
    {"simulator": "simulator", "end": END},
)
workflow.add_edge("simulator",  "processor")
workflow.add_edge("processor",  "analyst")
workflow.add_edge("analyst",    END)

# ── Compile ───────────────────────────────────────────────────────────────────
app = workflow.compile()


# --- 4. Execution Example ---
if __name__ == "__main__":
    print("Initializing Biomechanics AI Agent...")

    user_input = (
        "Simulate standing upright for a 72 kg man with height of 170 cm. "
        "Analyze the compression forces on L5_S1."
    )
    print(f"\nUser Prompt: {user_input}\n" + "=" * 40)

    result = app.invoke({"user_prompt": user_input, "uploaded_image_path": None})

    print("\n" + "=" * 40)
    print("FINAL AGENT RESPONSE:")
    print(result.get("final_message"))

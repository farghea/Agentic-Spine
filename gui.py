
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os  

from main import app
from utils import analysis_agent_node, set_active_model, get_active_model

st.set_page_config(page_title="Biomechanics AI Agent", layout="wide")

st.title("🦴 Biomechanics Agent")

if "active_model" not in st.session_state:
    st.session_state.active_model = get_active_model()[0]

try:
    set_active_model(st.session_state.active_model)
except Exception:
    st.session_state.active_model = "gemini"
    set_active_model("gemini")

# --- SIDEBAR: Configuration ---
with st.sidebar:
    st.header("Settings")

    left_col, right_col = st.columns(2)
    with left_col:
        if st.button(
            "Use Gemini",
            use_container_width=True,
            type="primary" if st.session_state.active_model == "gemini" else "secondary",
        ):
            st.session_state.active_model = "gemini"
            set_active_model("gemini")
            st.rerun()

    with right_col:
        if st.button(
            "Use OpenAI",
            use_container_width=True,
            type="primary" if st.session_state.active_model == "openai" else "secondary",
        ):
            st.session_state.active_model = "openai"
            set_active_model("openai")
            st.rerun()

    active_provider, active_model_type = get_active_model()
    st.write(f"Current Model: {active_provider.upper()} ({active_model_type})")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload custom .mot or .osim file", type=['mot', 'osim'])
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")
        # Logic to save this file could go here

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "dataframes" not in st.session_state:
    st.session_state.dataframes = None
if "simulation_done" not in st.session_state:
    st.session_state.simulation_done = False


# --- SECTION 1: SIMULATION ---
if not st.session_state.simulation_done:
    st.subheader("Enter prompt:")
    user_query = st.text_area("Describe the task:", 
        "Simulate standing upright for  72 kg man with height of between 170 cm. Analyze the compression forces on L5_S1. ")
    
    if st.button("Start Simulation"):
        # Create a Status Container that updates in real-time
        with st.status("Agent in work ...", expanded=True) as status:
            try:
                # 0. Clean up old plots before starting
                if os.path.exists("static/generated_plot.png"):
                    try: os.remove("static/generated_plot.png")
                    except: pass
                
                final_state = {}
                
                # --- NEW STREAMING LOGIC ---
                # app.stream yields steps as they finish (Node by Node)
                for step in app.stream({"user_prompt": user_query}):
                    
                    # Get the name of the node that just finished (e.g., 'model_selector')
                    node_name = list(step.keys())[0]
                    node_output = step[node_name]
                    
                    # Check if there is a status message to display
                    if "current_status" in node_output:
                        status.write(f"**{node_name.replace('_', ' ').title()}**: {node_output['current_status']}")
                    
                    # Update our local state copy
                    final_state.update(node_output)

                # --- END OF STREAM ---
                status.update(label="Simulation Complete!", state="complete", expanded=False)

                if final_state.get("dataframes"):
                    st.session_state.dataframes = final_state["dataframes"]
                    st.session_state.simulation_done = True
                    
                    history_entry = {
                        "role": "assistant", 
                        "content": final_state.get("final_message", "Done.")
                    }

                    # Check for plot immediately
                    if os.path.exists("static/generated_plot.png"):
                        try:
                            with open("static/generated_plot.png", "rb") as img_file:
                                history_entry["image"] = img_file.read()
                        except Exception as e:
                            st.error(f"Error loading initial plot: {e}")

                    st.session_state.chat_history.append(history_entry)
                    st.rerun()
                else:
                    st.error(f"Simulation Failed: {final_state.get('final_message')}")
            
            except Exception as e:
                st.error(f"Error: {e}")



# --- SECTION 2: ANALYSIS & PLOTTING ---
else:
    st.subheader("2. Analysis Dashboard")
    
    # TABS
    tab1, tab2, tab3 = st.tabs(["💬 Chat Analysis", "📊 Data Explorer", "🔄 New Simulation"])
    
    # [PASTE THE CODE HERE]
    with tab1:
        # 1. Initialize Chat Container
        chat_container = st.container()

        # 2. Render EXISTING History
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    if "content" in msg:
                        st.markdown(msg["content"])
                    if "image" in msg:
                        st.image(msg["image"], caption="Generated Plot")
        
        # 3. Handle NEW Input
        if prompt := st.chat_input("Ask about muscle forces, spinal loads, or request plots..."):
            
            # A. Display User Message Immediately
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # B. Append User Message to History
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # C. Generate & Display Assistant Response
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing data..."):
                        
                        # Clean old plot
                        if os.path.exists("static/generated_plot.png"):
                            try: os.remove("static/generated_plot.png")
                            except: pass 

                        recent_history = st.session_state.chat_history[-6:-1] 
                        history_str = "\n".join(
                            [f"{msg['role'].upper()}: {msg.get('content','')}" for msg in recent_history]
                        )

                        state_payload = {
                            "user_prompt": prompt, 
                            "dataframes": st.session_state.dataframes,
                            "chat_context": history_str 
                        }
                        
                        response = analysis_agent_node(state_payload)
                        answer = response["final_message"]
                        
                        st.markdown(answer)
                        
                        image_data = None
                        if os.path.exists("static/generated_plot.png"):
                            try:
                                with open("static/generated_plot.png", "rb") as img_file:
                                    image_data = img_file.read()
                                    st.image(image_data, caption="Agent Generated Plot")
                            except Exception as e:
                                st.error(f"Error displaying plot: {e}")
            
            # D. Append Assistant Message to History
            history_entry = {"role": "assistant", "content": answer}
            if image_data:
                history_entry["image"] = image_data
            st.session_state.chat_history.append(history_entry)
            
            st.rerun()

    with tab2:
        st.write("### Simulation Results")
        if st.session_state.dataframes:
            # 1. Select Dataset
            df_choice = st.selectbox("Select Dataset:", ["Spinal Loads", "Muscle Forces", "Muscle Activations"])
            mapping = {"Spinal Loads": "spinal", "Muscle Forces": "forces", "Muscle Activations": "activations"}
            
            key = mapping[df_choice]
            if key in st.session_state.dataframes:
                active_df = st.session_state.dataframes[key]
                
                # 2. Controls for Plotting
                col1, col2 = st.columns(2)
                with col1:
                    # Select X Axis (Default to first column, usually Time)
                    x_col = st.selectbox("Select X-Axis:", active_df.columns, index=0)
                with col2:
                    # Select Y Axis (Default to all others)
                    y_cols = st.multiselect("Select Y-Axis:", 
                                            [c for c in active_df.columns if c != x_col],
                                            default=[active_df.columns[1]] if len(active_df.columns) > 1 else [])

                # 3. Generate Plot (Circles/Scatter)
                if x_col and y_cols:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    for y_col in y_cols:
                        # 'o' creates circles, alpha helps with overlapping points
                        ax.scatter(active_df[x_col], active_df[y_col], label=y_col, alpha=0.7)
                    
                    ax.set_xlabel(x_col)
                    ax.set_ylabel("Value")
                    ax.set_title(f"{df_choice} vs {x_col}")
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.6)
                    
                    st.pyplot(fig)
                
                # 4. Show Raw Data
                with st.expander("View Raw Data"):
                    st.dataframe(active_df)
            else:
                st.warning(f"No data available for {df_choice}")
        else:
            st.info("Run a simulation first to see data.")

    with tab3:
        if st.button("Reset & Start Over"):
            st.session_state.clear()
            st.rerun()
    


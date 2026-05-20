
import streamlit as st
import pandas as pd
import os

from main import app
from utils import analysis_agent_node

st.set_page_config(page_title="Biomechanics AI Agent", layout="wide")
st.title("🦴 Biomechanics Agent")

PLOT_PATH = "static/generated_plot.png"

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
    user_query = st.text_area(
        "Describe the task:",
        "Simulate standing upright for a 72 kg man with height of 170 cm. Analyze the compression forces on L5_S1."
    )

    if st.button("Start Simulation"):
        with st.status("Agent working...", expanded=True) as status:
            try:
                final_state = {}

                for step in app.stream({"user_prompt": user_query}):
                    node_name = list(step.keys())[0]
                    node_output = step[node_name]
                    if "current_status" in node_output:
                        status.write(f"**{node_name.replace('_', ' ').title()}**: {node_output['current_status']}")
                    final_state.update(node_output)

                status.update(label="Simulation Complete!", state="complete", expanded=False)

                if final_state.get("dataframes"):
                    st.session_state.dataframes = final_state["dataframes"]
                    st.session_state.simulation_done = True
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": final_state.get("final_message", "Simulation done.")
                    })
                    st.rerun()
                else:
                    st.error(f"Simulation Failed: {final_state.get('final_message')}")

            except Exception as e:
                st.error(f"Error: {e}")


# --- SECTION 2: ANALYSIS ---
else:
    tab1, tab2, tab3 = st.tabs(["💬 Chat Analysis", "📊 Data Explorer", "🔄 New Simulation"])

    # ── TAB 1: CHAT ───────────────────────────────────────────────────────────
    with tab1:
        chat_container = st.container()

        # Replay history including stored plot images
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg.get("content", ""))
                    if "image" in msg:
                        st.image(msg["image"], caption="Generated Plot")

        if prompt := st.chat_input("Ask about muscle forces, spinal loads..."):
            # Show user message immediately
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Delete any stale plot so we can detect if a NEW one was created
            if os.path.exists(PLOT_PATH):
                try:
                    os.remove(PLOT_PATH)
                except Exception:
                    pass

            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        recent_history = st.session_state.chat_history[-6:-1]
                        history_str = "\n".join(
                            [f"{m['role'].upper()}: {m.get('content', '')}" for m in recent_history]
                        )
                        state_payload = {
                            "user_prompt": prompt,
                            "dataframes": st.session_state.dataframes,
                            "chat_context": history_str
                        }
                        response = analysis_agent_node(state_payload)
                        answer = response["final_message"]
                        st.markdown(answer)

                        # Detect and display a freshly generated plot
                        image_bytes = None
                        if os.path.exists(PLOT_PATH):
                            try:
                                with open(PLOT_PATH, "rb") as f:
                                    image_bytes = f.read()
                                st.image(image_bytes, caption="Generated Plot")
                            except Exception as img_err:
                                st.warning(f"Plot found but couldn't load: {img_err}")

            # Persist in history (image bytes included so it survives rerun)
            history_entry = {"role": "assistant", "content": answer}
            if image_bytes:
                history_entry["image"] = image_bytes
            st.session_state.chat_history.append(history_entry)
            st.rerun()

    # ── TAB 2: DATA EXPLORER ──────────────────────────────────────────────────
    with tab2:
        st.write("### Simulation Results")
        if st.session_state.dataframes:
            mapping = {
                "Spinal Loads": "spinal",
                "Muscle Forces": "forces",
                "Muscle Activations": "activations"
            }
            df_choice = st.selectbox("Select DataFrame:", list(mapping.keys()))
            key = mapping[df_choice]
            df = st.session_state.dataframes.get(key)

            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.warning(f"No data available for '{df_choice}'.")
        else:
            st.info("Run a simulation first to see data.")

    # ── TAB 3: RESET ──────────────────────────────────────────────────────────
    with tab3:
        if st.button("Reset & Start Over"):
            st.session_state.clear()
            st.rerun()

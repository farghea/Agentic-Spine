
import streamlit as st
import pandas as pd
import os
import tempfile

from main import app
from utils import analysis_agent_node

st.set_page_config(page_title="Biomechanics AI Agent", layout="wide")
st.title("🦴 Biomechanics Agent")

PLOT_PATH = "static/generated_plot.png"

# --- SESSION STATE ---
if "chat_history"           not in st.session_state:
    st.session_state.chat_history = []
if "dataframes"             not in st.session_state:
    st.session_state.dataframes = None
if "simulation_done"        not in st.session_state:
    st.session_state.simulation_done = False
if "image_analysis_result"  not in st.session_state:
    st.session_state.image_analysis_result = None
if "image_analysis_done"    not in st.session_state:
    st.session_state.image_analysis_done = False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: INPUT  (simulation not yet done and image analysis not yet done)
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.simulation_done and not st.session_state.image_analysis_done:

    st.subheader("Enter your prompt:")

    # ── Image upload ──────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload an image (optional — lifting scene or medical scan)",
        type=["jpg", "jpeg", "png", "tiff", "bmp", "webp"],
        help="If you upload an image it will be analysed by the AI. "
             "No image → the OpenSim simulation pipeline is used.",
    )

    # ── Text prompt ───────────────────────────────────────────────────────────
    user_query = st.text_area(
        "Describe the task (or add body height / weight hints for image analysis):",
        "Simulate standing upright for a 72 kg man with height of 170 cm. "
        "Analyze the compression forces on L5_S1.",
    )

    if st.button("Submit"):
        with st.status("Agent working...", expanded=True) as status:
            try:
                # ── Save uploaded image to a temp file ────────────────────────
                image_path = None
                if uploaded_file is not None:
                    suffix = os.path.splitext(uploaded_file.name)[-1]
                    os.makedirs("static", exist_ok=True)
                    image_path = os.path.join("static", f"uploaded_image{suffix}")
                    with open(image_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    status.write(f"📁 Image saved: `{image_path}`")

                # ── Stream graph ──────────────────────────────────────────────
                final_state = {}
                for step in app.stream({
                    "user_prompt":         user_query,
                    "uploaded_image_path": image_path,
                }):
                    node_name   = list(step.keys())[0]
                    node_output = step[node_name]
                    if "current_status" in node_output:
                        status.write(
                            f"**{node_name.replace('_', ' ').title()}**: "
                            f"{node_output['current_status']}"
                        )
                    final_state.update(node_output)

                status.update(label="Done!", state="complete", expanded=False)

                # ── Route result to the correct post-processing branch ────────

                # A) Image analysis result
                if final_state.get("image_analysis_result"):
                    st.session_state.image_analysis_result = final_state["image_analysis_result"]
                    st.session_state.image_analysis_done   = True
                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": final_state.get("final_message", "Image analysis done."),
                    })
                    st.rerun()

                # B) Medical placeholder
                elif final_state.get("image_type") == "medical" or (
                    image_path and not final_state.get("dataframes")
                    and final_state.get("final_message")
                ):
                    st.info(final_state.get("final_message", "Medical image pathway placeholder."))

                # C) OpenSim simulation result
                elif final_state.get("dataframes"):
                    st.session_state.dataframes     = final_state["dataframes"]
                    st.session_state.simulation_done = True
                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": final_state.get("final_message", "Simulation done."),
                    })
                    st.rerun()

                else:
                    st.error(f"Failed: {final_state.get('final_message', 'Unknown error.')}")

            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: IMAGE ANALYSIS RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.image_analysis_done:

    res = st.session_state.image_analysis_result
    tab_img, tab_reset = st.tabs(["📷 Image Analysis", "🔄 New Session"])

    with tab_img:
        st.subheader("Lifting Analysis Results")

        # ── 3-panel static figure (original image + skeleton) ────────────────
        if res and res.get("plot_bytes"):
            _, col_img, _ = st.columns([0.05, 0.9, 0.05])
            with col_img:
                st.image(res["plot_bytes"],
                         caption="SAM-3D Analysis — Original Image · 3-D Skeleton",
                         use_container_width=True)

        # ── Interactive PLY mesh (Plotly Mesh3d) ──────────────────────────────
        if res and res.get("ply_fig") is not None:
            st.divider()
            st.subheader("🦴 Interactive 3-D Body Mesh")
            st.plotly_chart(res["ply_fig"], use_container_width=True)


        # ── Metrics summary ───────────────────────────────────────────────────
        if res:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input Parameters**")
                st.table({
                    "Body Height (cm)":    [res.get("BH")],
                    "Body Weight (kg)":    [res.get("BW")],
                    "Object Weight M (kg)":[res.get("M")],
                })
                st.markdown("**Pose Metrics**")
                st.table({
                    "Flexion F (°)":    [res.get("flexion_deg")],
                    "Asymmetry A (°)":  [res.get("asymmetry_deg")],
                    "Reach D (cm)":     [res.get("reach_cm")],
                })
            with col2:
                loads = res.get("spinal_loads", {})
                st.markdown(f"**Spinal Loads** *(model: {res.get('model_used', '')})*")
                st.table({
                    "L4-L5 Compression (N)": [loads.get("L4L5_compression")],
                    "L4-L5 Shear (N)":       [loads.get("L4L5_shear")],
                    "L5-S1 Compression (N)": [loads.get("L5S1_compression")],
                    "L5-S1 Shear (N)":       [loads.get("L5S1_shear")],
                })

        # ── LLM summary text ──────────────────────────────────────────────────
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg.get("content", ""))

    with tab_reset:
        if st.button("Reset & Start Over"):
            st.session_state.clear()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: OPENSIM SIMULATION RESULTS  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
else:
    tab1, tab2, tab3 = st.tabs(["💬 Chat Analysis", "📊 Data Explorer", "🔄 New Simulation"])

    # ── TAB 1: CHAT ───────────────────────────────────────────────────────────
    with tab1:
        chat_container = st.container()

        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg.get("content", ""))
                    if "image" in msg:
                        st.image(msg["image"], caption="Generated Plot")

        if prompt := st.chat_input("Ask about muscle forces, spinal loads..."):
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

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
                            [f"{m['role'].upper()}: {m.get('content', '')}"
                             for m in recent_history]
                        )
                        state_payload = {
                            "user_prompt":  prompt,
                            "dataframes":   st.session_state.dataframes,
                            "chat_context": history_str,
                        }
                        response    = analysis_agent_node(state_payload)
                        answer      = response["final_message"]
                        st.markdown(answer)

                        image_bytes = None
                        if os.path.exists(PLOT_PATH):
                            try:
                                with open(PLOT_PATH, "rb") as f:
                                    image_bytes = f.read()
                                st.image(image_bytes, caption="Generated Plot")
                            except Exception as img_err:
                                st.warning(f"Plot found but couldn't load: {img_err}")

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
                "Spinal Loads":        "spinal",
                "Muscle Forces":       "forces",
                "Muscle Activations":  "activations",
            }
            df_choice = st.selectbox("Select DataFrame:", list(mapping.keys()))
            key = mapping[df_choice]
            df  = st.session_state.dataframes.get(key)

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

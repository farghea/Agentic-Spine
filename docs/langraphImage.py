import os

def save_graph_image(graph_app, folder="docs/diagrams", filename="spine_agent_flow.png"):
    # Create the directory if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    full_path = os.path.join(folder, filename)
    
    try:
        png_data = graph_app.get_graph().draw_mermaid_png()
        with open(full_path, "wb") as f:
            f.write(png_data)
        print(f"✅ Graph saved to: {full_path}")
    except Exception as e:
        print(f"❌ Error: {e}")

# Usage
# save_graph_image(app)
# Spine Project Setup

This project requires two Conda environments:

## 1. OpenSim Environment (`opensim`)
**Python Version:** 3.11.14

Install using the YAML file:
```bash
conda env create -f environment_opensim.yml
```

*Or manually via requirements:*
```bash
conda create -n opensim python=3.11.14 -y
conda activate opensim
conda install -c opensim-org opensim -y
pip install -r requirements_opensim.txt
```

## 2. Agent Environment (`agent`)
**Python Version:** 3.12.12

Install using the YAML file:
```bash
conda env create -f environment_agent.yml
```

*Or manually via requirements:*
```bash
conda create -n agent python=3.12.12 -y
conda activate agent
pip install -r requirements_agent.txt
```

## 3. LangSmith tracing

Add these fields in `info_and_keys.json`:
- `langsmith_api_key`
- `langsmith_project` (set to `Agentic_Spine`)

Tracing uses `@traceable` with explicit `run_type` values:
- `run_type="llm"`: `_gemini_generate_text`, `_llm_split_simulation_and_analysis_request`
- `run_type="tool"`: `run_opensim_simulation`, `repl_execution_node`
- `run_type="chain"`: `analyze_request_node`, `model_selection_node`, `simulation_node`, `data_processing_node`, `code_generation_node`, `execution_output_node`, `router_node`, `analysis_agent_node`

OpenAI calls are wrapped with `langsmith.wrappers.wrap_openai(...)` so provider calls show up as traced sub-runs.

This keeps current Gemini/OpenAI call paths unchanged while still tracking runs in LangSmith.

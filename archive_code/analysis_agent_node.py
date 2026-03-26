

# def analysis_agent_node(state):
#     """
#     Node 5: The 'Analyst'. Connects to Chat logic using Pandas Dataframe Agents.
#     Integrates the specific prompt logic provided in utils.
#     """
#     print("--- Node 5: Agentic Analysis (Chat) ---")
    
#     dfs = state.get('dataframes')
#     if not dfs:
#         return {"final_message": "Error: No data available for analysis."}

#     # 1. Load Keys
#     try:
#         with open('info_and_keys.json') as f:
#             keys = json.load(f)
#     except:
#         return {"final_message": "Error loading keys for analysis agent."}

#     # 2. Init LLM
#     if MODEL == 'openai':
#         llm = ChatOpenAI(model=MODEL_TYPE, api_key=keys["openai_api_key"])
#     else:
#         llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=keys["gemini_api_key"])

#     # 3. Define Domain Knowledge (Prefixes)
#     domain_knowledge = """
#     You are a Biomechanics Assistant analyzing OpenSim data.
    
#     DATA DEFINITIONS:
#     1. SPINAL LOADS:
#        - *_fy = Compression forces (Vertical).
#        - *_fx, *_fz = Shear forces (Horizontal).
#        - T1_T2...L5_S1 = Spinal joints.
    
#     2. MUSCLE FORCES (Newtons) & ACTIVATIONS (0.0-1.0):
#        - Suffixes: '_r'/'_R' = Right, '_l'/'_L' = Left.
#        - No suffix usually implies Right/Primary.
#        - Groups: 
#          * Psoas ('Ps_...')
#          * Iliocostalis ('IL_...')
#          * Longissimus ('LTpT_...', 'LTpL_...')
#          * Quadratus Lumborum ('QL_...')
#          * Multifidus ('MF_...')
#          * Abdominals ('rect_abd', 'IO', 'EO')
#     """

#     # 4. Create Agent 
#     # We allow the agent to see all 3 DataFrames to cross-reference if needed
#     agent = create_pandas_dataframe_agent(
#         llm=llm,
#         df=[dfs['spinal'], dfs['forces'], dfs['activations']],
#         verbose=True,
#         allow_dangerous_code=True, # Required for Pandas Agent
#         agent_type="openai-tools",
#         handle_parsing_errors=True,
#         prefix=domain_knowledge
#     )

#     # 5. Construct Query
#     # We combine the User's original intent + a request for summary

#     plot_instructions = """
#     Only if asked for plotting; otherwise, ignore: RULES FOR PLOTTING (MUST FOLLOW STICTLY):
#     1. SETUP:
#        - You MUST use `plt.switch_backend('Agg')` at the start of your code to prevent GUI errors.
#        - Use `plt.figure(figsize=(10, 6))` for legibility.
#        - Use `plt.style.use('ggplot')` for professional formatting.
       
#     2. DATA SELECTION:
#        - Always verify the exact column name exists in `df.columns` before plotting. Use fuzzy matching if needed (e.g., if user asks for "L5 compression", look for "L5_S1_fy").
#        - The X-Axis must ALWAYS be the 'time' column.
    
#     3. RENDERING:
#        - clear the figure with `plt.clf()` before plotting.
#        - Add a title, X-label ('Time [s]'), and Y-label (with units like 'Newtons' or 'Activation').
#        - If plotting multiple muscles, include a legend.
       
#     4. SAVING:
#        - You MUST save the file to: 'static/generated_plot.png' (Create the directory if it doesn't exist).
#        - Use `plt.savefig('static/generated_plot.png', dpi=300, bbox_inches='tight')`.
#        - DO NOT use `plt.show()`.
    
#     5. OUTPUT:
#        - After saving, your final answer MUST confirm: "Plot saved to static/generated_plot.png".
#        - Provide a brief 1-sentence interpretation of the peak values in the plot.
#     """

#     query = (
#         f"The user asked: '{state['user_prompt']}'.\n"
#         "1. Analyze the provided DataFrames (df1=Spinal, df2=Forces, df3=Activations).\n"
#         "2. If the user asks for a PLOT or GRAPH:\n"
#         f"{plot_instructions}\n"
#         "3. Provide a text summary of what the plot shows."
#     )

#     try:
#         response = agent.invoke({"input": query})
#         output_text = response["output"] if isinstance(response, dict) else str(response)
#         return {"final_message": output_text}
#     except Exception as e:
#         return {"final_message": f"Analysis Agent Error: {e}"}




# def analysis_agent_node(state):
#     """
#     Node 5: The 'Analyst'. 
#     Architecture:
#     1. PLANNER LLM: Reads history -> Decides which of the 3 DataFrames to use.
#     2. ROUTER: Activates ONLY the specific agent for that DataFrame.
#     3. SPECIALIST AGENT: executes the query with strict domain knowledge.
#     """
#     print(f"--- Node 5: Agentic Analysis ({MODEL}) ---")
    
#     dfs = state.get('dataframes')
#     if not dfs:
#         return {"final_message": "Error: No data available for analysis."}
    
#     # 1. Load Keys & Init LLM (STRICTLY USING YOUR GLOBALS)
#     try:
#         with open('info_and_keys.json') as f: 
#             keys = json.load(f)
        
#         if MODEL == 'openai':
#             llm = ChatOpenAI(
#                 model=MODEL_TYPE, 
#                 api_key=keys["openai_api_key"], 
#                 temperature=0
#             )
#         elif MODEL == 'gemini':
#             llm = ChatGoogleGenerativeAI(
#                 model=MODEL_TYPE, 
#                 google_api_key=keys["gemini_api_key"], 
#                 temperature=0
#             )
#         else:
#             return {"final_message": f"Error: Unknown MODEL configuration '{MODEL}'"}
            
#     except Exception as e:
#         return {"final_message": f"Config/LLM Init Error: {e}"}

#     # 2. Prepare Context
#     chat_history = state.get('chat_context', "No prior history.")
#     user_input = state.get('user_prompt', "")

#     # ==================================================================================
#     # STEP 1: THE PLANNER (Decide Intent & DataFrame)
#     # ==================================================================================
#     planner_prompt = f"""
#     You are the "Planner". 
    
#     CONTEXT:
#     User Input: "{user_input}"
#     Chat History: 
#     {chat_history}
    
#     AVAILABLE AGENTS / DATAFRAMES:
#     1. 'spinal': For spinal joint loads (Compression/Shear).
#        - Columns: ['Model', 'Activity', 'Load_Name', 'Value']
#     2. 'forces': For specific muscle forces (Newtons).
#        - Columns: ['Model', 'Activity', 'Muscle_Name', 'Value']
#     3. 'activations': For muscle activation levels (0.0 to 1.0).
#        - Columns: ['Model', 'Activity', 'Muscle_Name', 'Value']
    
#     YOUR TASK:
#     1. Analyze what the user is asking for.
#     2. Select ONE agent ('spinal', 'forces', or 'activations').
#     3. Write a specific instruction for that agent.
    
#     RULES:
#     - If the user asks for "L5 compression" or "Shear", choose 'spinal'.
#     - If the user asks for "Longissimus" or "Psoas" force, choose 'forces'.
#     - If the user asks for "Effort" or "Activation", choose 'activations'.
#     - If vague, default to 'spinal' with a summary request.
    
#     OUTPUT JSON ONLY:
#     {{
#         "target_agent": "spinal" | "forces" | "activations",
#         "instruction": "Precise instruction for the coding agent..."
#     }}
#     """
    
#     try:
#         response = llm.invoke(planner_prompt)
#         # Clean markdown
#         content = response.content.replace('```json', '').replace('```', '').strip()
#         plan = json.loads(content)
        
#         target = plan.get('target_agent', 'spinal')
#         instruction = plan.get('instruction', 'Summarize the data.')
        
#         print(f"[Planner] Decided: {target.upper()}")
#         print(f"[Planner] Task: {instruction}")
        
#     except Exception as e:
#         print(f"Planner Error: {e}")
#         target = "spinal"
#         instruction = "Summarize the spinal load data."

#     # ==================================================================================
#     # STEP 2 & 3: DEFINE AND SELECT THE SPECIFIC AGENT
#     # ==================================================================================
    
#     # Common Plotting Rules for all agents
#     plot_rules = """
#     IF PLOTTING:
#     - You MUST use `plt.switch_backend('Agg')` first.
#     - `plt.clf()` to clear.
#     - Save to 'static/generated_plot.png'.
#     - DO NOT use `plt.show()`.
#     """

#     selected_df = None
#     agent_prefix = ""

#     # --- AGENT A: SPINAL AGENT ---
#     if target == 'spinal':
#         selected_df = dfs['spinal']
#         agent_prefix = f"""
#         You are the SPINAL LOAD SPECIALIST.
        
#         YOUR DATA (`df`):
#         - Format: Long format (Rows are individual measurements).
#         - Columns: ['Model', 'Activity', 'Age', 'Weight_kg', 'Height_m', 'Load_Name', 'Value']
        
#         DOMAIN KNOWLEDGE:
#         - `Load_Name` contains the joint and direction.
#         - `_fy` suffix = Compression Forces (Vertical).
#         - `_fx` or `_fz` suffix = Shear Forces (Horizontal).
#         - `Value` is in Newtons.
        
#         TASK: {instruction}
        
#         TIPS:
#         - To find L5 Compression: Filter `df` where `Load_Name` contains 'L5' and 'fy'.
#         - {plot_rules}
#         """

#     # --- AGENT B: MUSCLE FORCES AGENT ---
#     elif target == 'forces':
#         selected_df = dfs['forces']
#         agent_prefix = f"""
#         You are the MUSCLE FORCE SPECIALIST.
        
#         YOUR DATA (`df`):
#         - Format: Long format.
#         - Columns: ['Model', 'Activity', 'Age', 'Weight_kg', 'Height_m', 'Muscle_Name', 'Value']
#         - `Value` is Force in Newtons.
        
#         MUSCLE NAMING CONVENTION (in `Muscle_Name`):
#         - Suffixes: '_l' or '_L' (Left), '_r' or '_R' (Right).
#         - No suffix usually implies Right.
#         - Groups (Prefixes):
#           * 'Ps_' = Psoas
#           * 'IL_' = Iliocostalis
#           * 'LTpT_', 'LTpL_' = Longissimus (Thoracic/Lumbar)
#           * 'QL_' = Quadratus Lumborum
#           * 'MF_' = Multifidus
#           * 'rect_abd', 'IO', 'EO' = Abdominals
        
#         TASK: {instruction}
        
#         TIPS:
#         - Use string contains to filter groups (e.g. `df[df['Muscle_Name'].str.contains('LTp')]`).
#         - {plot_rules}
#         """

#     # --- AGENT C: ACTIVATIONS AGENT ---
#     elif target == 'activations':
#         selected_df = dfs['activations']
#         agent_prefix = f"""
#         You are the MUSCLE ACTIVATION SPECIALIST.
        
#         YOUR DATA (`df`):
#         - Format: Long format.
#         - Columns: ['Model', 'Activity', 'Age', 'Weight_kg', 'Height_m', 'Muscle_Name', 'Value']
#         - `Value` is Activation (0.0 to 1.0). 1.0 = Max effort.
        
#         MUSCLE NAMING CONVENTION:
#         - Suffixes: '_l' (Left), '_r' (Right).
#         - Groups: 'Ps_' (Psoas), 'IL_' (Iliocostalis), 'LTpT_'/'LTpL_' (Longissimus), 'QL_' (Quad Lumb), 'MF_' (Multifidus).
        
#         TASK: {instruction}
        
#         TIPS:
#         - {plot_rules}
#         """

#     # Sanity Check
#     if selected_df is None or selected_df.empty:
#         return {"final_message": f"Error: The chosen dataframe ({target}) is empty or missing."}

#     # ==================================================================================
#     # STEP 4: EXECUTE THE CHOSEN AGENT
#     # ==================================================================================
#     try:
#         agent = create_pandas_dataframe_agent(
#             llm=llm,
#             df=selected_df,
#             verbose=True,
#             allow_dangerous_code=True,
#             agent_type="openai-tools",
#             handle_parsing_errors=True,
#             prefix=agent_prefix
#         )

#         response = agent.invoke({"input": instruction})
#         output_text = response["output"] if isinstance(response, dict) else str(response)
        
#         # Check for plot file creation hint
#         if "plot" in instruction.lower() and "saved" not in output_text.lower():
#              output_text += "\n(Note: Check the dashboard for the generated plot.)"
             
#         return {"final_message": output_text}
        
#     except Exception as e:
#         return {"final_message": f"Analysis Error in {target.upper()} agent: {e}"}



# def analysis_agent_node(state):
#     """
#     Node 5: The 'Analyst'. 
#     - USES: 3-Agent Architecture (Planner -> Specialist).
#     - FIXES: "I don't have dataframe" errors (via strict system prompts).
#     - RESTORED: Full Domain Knowledge for mapping 'Psoas' -> 'Ps_', etc.
#     """
#     print(f"--- Node 5: Agentic Analysis ({MODEL}) ---")
    
#     dfs = state.get('dataframes')
#     if not dfs:
#         return {"final_message": "Error: No data available for analysis."}
    
#     # 1. Load Keys & Init LLM
#     try:
#         with open('info_and_keys.json') as f: 
#             keys = json.load(f)
        
#         if MODEL == 'openai':
#             llm = ChatOpenAI(
#                 model=MODEL_TYPE, 
#                 api_key=keys["openai_api_key"], 
#                 temperature=0
#             )
#         elif MODEL == 'gemini':
#             llm = ChatGoogleGenerativeAI(
#                 model=MODEL_TYPE, 
#                 google_api_key=keys["gemini_api_key"], 
#                 temperature=0
#             )
#         else:
#             return {"final_message": f"Error: Unknown MODEL configuration '{MODEL}'"}
            
#     except Exception as e:
#         return {"final_message": f"Config/LLM Init Error: {e}"}

#     # 2. Context
#     chat_history = state.get('chat_context', "No prior history.")
#     user_input = state.get('user_prompt', "")

#     # ==================================================================================
#     # STEP 1: THE PLANNER
#     # ==================================================================================
#     planner_prompt = f"""
#     You are the "Planner".
    
#     USER INPUT: "{user_input}"
#     CHAT HISTORY: 
#     {chat_history}
    
#     AVAILABLE DATAFRAMES:
#     1. 'spinal' (Columns: Model, Activity, Load_Name, Value)
#     2. 'forces' (Columns: Model, Activity, Muscle_Name, Value)
#     3. 'activations' (Columns: Model, Activity, Muscle_Name, Value)
    
#     TASK:
#     1. Select ONE dataframe ('spinal', 'forces', or 'activations').
#     2. Write a Python-focused instruction.
    
#     RULES:
#     - If user asks for "L5 compression" or "Shear", use 'spinal'.
#     - If user asks for "Muscle Force" (Newtons), use 'forces'.
#     - If user asks for "Activation", "Effort" or "Recruitment" (0-1), use 'activations'.
#     - If plotting is needed, explicitly say: "Plot the data and save to static/generated_plot.png".
    
#     OUTPUT JSON ONLY:
#     {{
#         "target_agent": "spinal" | "forces" | "activations",
#         "instruction": "Exact instruction for the agent..."
#     }}
#     """
    
#     try:
#         response = llm.invoke(planner_prompt)
#         content = response.content.replace('```json', '').replace('```', '').strip()
#         plan = json.loads(content)
#         target = plan.get('target_agent', 'spinal')
#         instruction = plan.get('instruction', 'Summarize data.')
#         print(f"[Planner] Target: {target} | Instruction: {instruction}")
        
#     except Exception as e:
#         print(f"Planner Error: {e}")
#         target = "spinal"
#         instruction = "Summarize the spinal load data."

#     # ==================================================================================
#     # STEP 2: CONFIGURE THE SPECIALIST AGENT (With Full Domain Knowledge)
#     # ==================================================================================
    
#     selected_df = dfs.get(target)
#     if selected_df is None or selected_df.empty:
#         return {"final_message": f"Error: Dataframe '{target}' is empty."}

#     # --- COMMON GUARDRAILS (The Fix for "I don't have data") ---
#     base_prefix = f"""
#     You are a Python Data Analyst. 
    
#     CRITICAL EXECUTION RULES:
#     1. The dataframe is ALREADY LOADED in your environment as the variable `df`.
#     2. DO NOT ask the user for data. DO NOT say "I don't have access". USE `df` DIRECTLY.
#     3. You MUST execute Python code to answer. Do not just write the code; RUN IT.
    
#     PLOTTING RULES (IF ASKED):
#     - Use `plt.switch_backend('Agg')` at the start.
#     - `plt.clf()` to clear previous plots.
#     - Save strictly to: 'static/generated_plot.png'
#     - DO NOT use `plt.show()`.
#     """

#     # --- RESTORED DOMAIN KNOWLEDGE ---
#     if target == 'spinal':
#         system_prefix = base_prefix + """
        
#         DOMAIN KNOWLEDGE (SPINAL LOADS):
#         - `Load_Name` format: [Joint]_[Direction]
#         - COMPRESSION (Vertical): Look for suffix `_fy`. (e.g., 'L5_S1_fy')
#         - SHEAR (Horizontal): Look for suffixes `_fx` or `_fz`.
#         - JOINTS: 'T1_T2', ... 'L5_S1'.
#         - UNITS: Newtons.
        
#         TIP: To find 'L5 Compression', filter where `Load_Name` contains 'L5' AND ends with '_fy'.
#         """
        
#     elif target in ['forces', 'activations']:
#         system_prefix = base_prefix + f"""
        
#         DOMAIN KNOWLEDGE (MUSCLES):
#         - COLUMN: 'Muscle_Name'
#         - SUFFIXES: `_r` or `_R` (Right Side), `_l` or `_L` (Left Side).
#           (If no suffix is specified by user, assume Both or Right).
        
#         MUSCLE MAPPING (Prefix -> Muscle Group):
#         * 'Ps_'    -> Psoas Major
#         * 'IL_'    -> Iliocostalis (e.g., IL_L1_L2_r)
#         * 'LTpT_'  -> Longissimus Thoracis
#         * 'LTpL_'  -> Longissimus Lumborum
#         * 'QL_'    -> Quadratus Lumborum
#         * 'MF_'    -> Multifidus
#         * 'rect_abd' -> Rectus Abdominis
#         * 'IO_'    -> Internal Oblique
#         * 'EO_'    -> External Oblique
        
#         VALUE DEFINITION:
#         - If 'forces' agent: Units are Newtons.
#         - If 'activations' agent: Units are 0.0 to 1.0 (Normalized).
#         """

#     # ==================================================================================
#     # STEP 3: EXECUTE
#     # ==================================================================================
#     try:
#         agent = create_pandas_dataframe_agent(
#             llm=llm,
#             df=selected_df,
#             verbose=True,
#             allow_dangerous_code=True,
#             agent_type="openai-tools",  
#             handle_parsing_errors=True,
#             prefix=system_prefix
#         )

#         # Force the input to remind the agent AGAIN that df exists
#         final_prompt = f"{instruction} (Recall: Data is in variable `df`)"
        
#         response = agent.invoke({"input": final_prompt})
#         output_text = response["output"] if isinstance(response, dict) else str(response)

#         # Check for plot success in the text
#         if "plot" in instruction.lower() and "saved" not in output_text.lower():
#             output_text += "\n(System Note: A plot was requested. Check the dashboard.)"
            
#         return {"final_message": output_text}

#     except Exception as e:
#         return {"final_message": f"Agent Execution Error: {e}"}
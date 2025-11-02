"""
AI-Powered EDA Chat Agent - MVP with Chart Generation and Logging
Single file implementation with LangGraph and Chainlit using Google Gemini
"""
import traceback
import os
import io
import sys
import logging
from uuid import uuid4
from typing import TypedDict, Literal
from contextlib import redirect_stdout
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import chainlit as cl
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# ============================================================================
# Logging Configuration
# ============================================================================

# Create logs directory if it doesn't exist
os.makedirs("./logs", exist_ok=True)

# Configure logging
log_filename = f"./logs/eda_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create logger
logger = logging.getLogger("EDA_Agent")
logger.setLevel(logging.DEBUG)


GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("⚠️ GOOGLE_API_KEY not found in environment; LLM calls may fail")

# Custom formatter to remove emojis for console
class NoEmojiFormatter(logging.Formatter):
    """Formatter that strips emojis from log messages for console output."""
    def format(self, record):
        # Get the original message
        original_msg = record.getMessage()
        # Remove emojis (basic removal - strips non-ASCII characters)
        record.msg = ''.join(char for char in original_msg if ord(char) < 128)
        record.args = ()
        return super().format(record)

# Create formatters
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

simple_formatter = NoEmojiFormatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# File handler - detailed logs with emojis
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(detailed_formatter)

# Console handler - no emojis (Windows safe)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(simple_formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("="*80)
logger.info("EDA Agent Started")
logger.info(f"Log file: {log_filename}")
logger.info("="*80)

# Set style for better-looking charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# LangGraph Agent State & Nodes
# ============================================================================

class AgentState(TypedDict):
    question: str
    dataset_path: str
    code_history: list[str]
    reflection: str
    final_answer: str
    iteration: int
    chart_path: str


def planner_node(state: AgentState) -> AgentState:
    """Analyzes the question and previous attempts to plan next action."""
    logger.info("🧠 PLANNER NODE: Starting planning phase")
    logger.debug(f"Input state: question='{state['question']}', iteration={state.get('iteration', 0)}")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
    system_prompt = """You are a data analysis planner. Your job is to create a plan for answering the user's question about their dataset.

The dataset is loaded as a pandas DataFrame called 'df'.

IMPORTANT: Break down complex EDA requests into SMALL, FOCUSED steps. Each iteration should accomplish ONE specific task:
- For "do EDA": Start with basic overview (shape, info, describe), then move to specific analyses in later iterations
- Focus on the MOST IMPORTANT insight first
- Keep each step simple and fast to execute

Previous code history:
{code_history}

Previous reflection:
{reflection}

User question: {question}

Current iteration: This is attempt #{iteration}

Provide a concise, FOCUSED plan (1-2 sentences) for the NEXT SINGLE analysis step. Don't try to do everything at once."""

    prompt = system_prompt.format(
        code_history="\n".join(state.get("code_history", [])) if state.get("code_history") else "None",
        reflection=state.get("reflection", "None"),
        question=state["question"],
        iteration=state.get("iteration", 0) + 1
    )
    
    logger.debug(f"Sending prompt to LLM (length: {len(prompt)} chars)")
    response = llm.invoke([HumanMessage(content=prompt)])
    
    logger.info(f"📋 Plan generated: {response.content[:100]}...")
    logger.debug(f"Full plan: {response.content}")
    
    return {
        **state,
        "reflection": response.content
    }


def coder_node(state: AgentState) -> AgentState:
    """Generates and executes Python code based on the plan."""
    logger.info("💻 CODER NODE: Starting code generation and execution")
    logger.debug(f"Plan to implement: {state.get('reflection', '')[:100]}...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
    
    system_prompt = """You are a Python data analysis expert. Generate Python code to answer the user's question.

Rules:
1. The dataset is already loaded as a pandas DataFrame called 'df'
2. Use pandas, matplotlib (plt), and seaborn (sns) for analysis and visualization
3. For visualizations, ALWAYS save the plot using: plt.savefig(chart_path, bbox_inches='tight', dpi=100)
4. After saving, ALWAYS call plt.close() to free memory
5. Print results using print() statements
6. Keep code concise and focused - DO ONE THING WELL
7. Handle potential errors gracefully
8. Do NOT include import statements (libraries are already imported)
9. Do NOT reload the data (df is already available)
10. Variable 'chart_path' is available for saving charts
11. IMPORTANT: Keep execution time under 10 seconds - don't generate too many plots or complex computations

For broad EDA requests, focus on ONE aspect at a time:
- Iteration 1: Basic info (df.info(), df.describe())
- Iteration 2: Missing values and data types
- Iteration 3: One key visualization (distribution of target variable, correlation heatmap, etc.)

Plan: {reflection}

User question: {question}

Generate ONLY the Python code, no explanations. The code will be executed directly."""

    prompt = system_prompt.format(
        reflection=state.get("reflection", ""),
        question=state["question"]
    )
    
    logger.debug("Requesting code generation from LLM")
    response = llm.invoke([HumanMessage(content=prompt)])
    code = response.content.strip()
    
    # Remove markdown code blocks if present
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()
    
    logger.info("📝 Generated code:")
    logger.info("-" * 60)
    for i, line in enumerate(code.split('\n'), 1):
        logger.info(f"{i:3d} | {line}")
    logger.info("-" * 60)
    
    # Execute the code
    output = ""
    error = ""
    chart_path = ""
    
    try:
        logger.info("⚙️  Loading dataset...")
        df = pd.read_csv(state["dataset_path"])
        logger.debug(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Create charts directory if it doesn't exist
        os.makedirs("./temp_charts", exist_ok=True)
        
        # Generate unique chart path
        chart_filename = f"{uuid4()}.png"
        chart_path = f"./temp_charts/{chart_filename}"
        logger.debug(f"Chart path prepared: {chart_path}")
        
        logger.info("▶️  Executing code...")
        # Capture stdout
        f = io.StringIO()
        with redirect_stdout(f):
            # Execute the code with chart_path available
            exec(code, {
                "pd": pd,
                "df": df,
                "plt": plt,
                "sns": sns,
                "print": print,
                "chart_path": chart_path
            })
        
        output = f.getvalue()
        logger.info("✅ Code executed successfully")
        
        if output:
            logger.info("📤 Output:")
            logger.info("-" * 60)
            for line in output.split('\n'):
                if line.strip():
                    logger.info(f"   {line}")
            logger.info("-" * 60)
        
        # Check if chart was actually created
        if os.path.exists(chart_path):
            chart_size = os.path.getsize(chart_path)
            logger.info(f"📊 Chart generated: {chart_path} ({chart_size} bytes)")
        else:
            logger.debug("No chart file was created")
            chart_path = ""
        
    except Exception as e:
        error = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
        output = error
        chart_path = ""
        logger.error("❌ Code execution failed:")
        logger.error(error)
    
    # Update code history
    code_history = state.get("code_history", [])
    code_history.append(f"Code:\n{code}\n\nOutput:\n{output}")
    
    logger.debug(f"Code history updated (total entries: {len(code_history)})")
    
    return {
        **state,
        "code_history": code_history,
        "reflection": output,
        "chart_path": chart_path if chart_path else state.get("chart_path", "")
    }


def reflector_node(state: AgentState) -> AgentState:
    """Reflects on the code output and decides if the question is answered."""
    logger.info("🤔 REFLECTOR NODE: Analyzing results")
    logger.debug(f"Iteration: {state.get('iteration', 0)}")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
    
    system_prompt = """You are analyzing the results of a data analysis to determine if the user's question has been answered.

User question: {question}

Latest analysis output:
{reflection}

Chart generated: {has_chart}

Iteration: {iteration} of 5

All previous attempts:
{code_history}

Task:
1. Determine if the question is SUFFICIENTLY answered (don't demand perfection)
2. For broad EDA requests: Mark as COMPLETE after 3-4 good iterations showing key insights
3. If YES: Provide a clear, concise final answer summarizing what was discovered
4. If NO and iterations < 4: Explain what ONE additional thing to analyze next

If a chart was generated, mention it in your answer (e.g., "The visualization shows...")

Format your response as:
DECISION: [COMPLETE or CONTINUE]
ANSWER: [Your response to the user]"""

    prompt = system_prompt.format(
        question=state["question"],
        reflection=state.get("reflection", ""),
        has_chart="Yes" if state.get("chart_path") else "No",
        iteration=state.get("iteration", 0) + 1,
        code_history="\n\n".join(state.get("code_history", [])) if state.get("code_history") else "None"
    )
    
    logger.debug("Requesting reflection from LLM")
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    
    # Parse the decision
    decision_line = [line for line in content.split("\n") if line.startswith("DECISION:")][0]
    is_complete = "COMPLETE" in decision_line
    
    logger.info(f"🎯 Decision: {'COMPLETE' if is_complete else 'CONTINUE'}")
    
    # Extract answer
    answer_start = content.find("ANSWER:")
    if answer_start != -1:
        answer = content[answer_start + 7:].strip()
    else:
        answer = content
    
    logger.debug(f"Answer preview: {answer[:100]}...")
    
    iteration = state.get("iteration", 0) + 1
    logger.info(f"📍 Completed iteration {iteration}")
    
    return {
        **state,
        "final_answer": answer if is_complete else "",
        "reflection": answer,
        "iteration": iteration
    }


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """Determines if the agent should continue or end."""
    iteration = state.get("iteration", 0)
    has_answer = bool(state.get("final_answer"))
    
    if has_answer:
        logger.info("✅ Agent flow: ENDING (answer found)")
        return "end"
    elif iteration >= 5:
        logger.warning(f"⚠️  Agent flow: ENDING (max iterations {iteration} reached)")
        return "end"
    else:
        logger.info(f"🔄 Agent flow: CONTINUING (iteration {iteration}/5)")
        return "continue"


# Build the LangGraph
logger.info("🏗️  Building LangGraph workflow...")
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("coder", coder_node)
workflow.add_node("reflector", reflector_node)

# Set entry point
workflow.set_entry_point("planner")

# Add edges
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "reflector")

# Add conditional edges
workflow.add_conditional_edges(
    "reflector",
    should_continue,
    {
        "continue": "planner",
        "end": END
    }
)

# Compile the graph
eda_agent = workflow.compile()
logger.info("✅ LangGraph workflow compiled successfully")


# ============================================================================
# Chainlit Application
# ============================================================================

@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    session_id = str(uuid4())[:8]
    logger.info(f"🚀 New chat session started: {session_id}")
    cl.user_session.set("session_id", session_id)
    
    await cl.Message(
        content="👋 Welcome to the EDA Agent with Chart Generation! Please upload a CSV file to get started."
    ).send()
    
    # Request file upload
    logger.info(f"📁 [{session_id}] Requesting file upload...")
    files = await cl.AskFileMessage(
        content="Please upload your CSV file (max 100MB):",
        accept=["text/csv"],
        max_size_mb=100,
        timeout=180
    ).send()
    
    if not files:
        logger.warning(f"❌ [{session_id}] No file uploaded")
        await cl.Message(content="❌ No file uploaded. Please refresh to try again.").send()
        return
    
    file = files[0]
    logger.info(f"📥 [{session_id}] File received: {file.name} ({file.size} bytes)")
    
    # Create temp_data directory if it doesn't exist
    os.makedirs("./temp_data", exist_ok=True)
    
    # Save file with unique name
    unique_filename = f"{uuid4()}.csv"
    dataset_path = f"./temp_data/{unique_filename}"
    
    logger.debug(f"💾 [{session_id}] Saving file to: {dataset_path}")
    
    # Save the uploaded file
    with open(file.path, "rb") as src:
        with open(dataset_path, "wb") as dst:
            dst.write(src.read())
    
    logger.info(f"✅ [{session_id}] File saved successfully")
    
    # Validate the file
    try:
        logger.info(f"🔍 [{session_id}] Validating CSV file...")
        df = pd.read_csv(dataset_path)
        rows, cols = df.shape
        
        logger.info(f"📊 [{session_id}] Dataset loaded: {rows} rows × {cols} columns")
        logger.debug(f"Columns: {', '.join(df.columns.tolist())}")
        logger.debug(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        await cl.Message(
            content=f"✅ Dataset loaded successfully!\n\n📊 **Shape:** {rows} rows × {cols} columns\n\n💬 You can now ask questions about your data or request visualizations!"
        ).send()
        
        # Store dataset path in user session
        cl.user_session.set("dataset_path", dataset_path)
        logger.info(f"✅ [{session_id}] Session initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ [{session_id}] Error loading CSV: {str(e)}")
        logger.debug(traceback.format_exc())
        
        await cl.Message(
            content=f"❌ Error loading CSV file: {str(e)}\n\nPlease refresh and upload a valid CSV file."
        ).send()
        
        # Clean up invalid file
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
            logger.debug(f"🗑️  [{session_id}] Cleaned up invalid file")


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and run the EDA agent."""
    session_id = cl.user_session.get("session_id", "unknown")
    dataset_path = cl.user_session.get("dataset_path")
    
    logger.info("="*80)
    logger.info(f"💬 [{session_id}] New message received: '{message.content}'")
    logger.info("="*80)
    
    if not dataset_path:
        logger.warning(f"⚠️  [{session_id}] No dataset loaded")
        await cl.Message(
            content="⚠️ No dataset loaded. Please refresh the page and upload a CSV file first."
        ).send()
        return
    
    # Prepare initial state
    initial_state = {
        "question": message.content,
        "dataset_path": dataset_path,
        "code_history": [],
        "reflection": "",
        "final_answer": "",
        "iteration": 0,
        "chart_path": ""
    }
    
    logger.info(f"🎬 [{session_id}] Starting agent workflow...")
    start_time = datetime.now()
    
    # Create a step message for progress updates
    step_msg = cl.Message(content="")
    await step_msg.send()
    
    # Run the agent step by step
    try:
        current_state = initial_state
        
        while True:
            iteration = current_state.get("iteration", 0) + 1
            
            # Update status: Planning
            step_msg.content = f"🧠 **Step {iteration}/5: Planning...**"
            await step_msg.update()
            current_state = planner_node(current_state)
            
            # Show the plan
            plan_preview = current_state.get("reflection", "")[:150]
            step_msg.content = f"🧠 **Step {iteration}/5: Planning Complete**\n\n📋 *{plan_preview}...*\n\n💻 Generating code..."
            await step_msg.update()
            
            # Update status: Coding
            current_state = coder_node(current_state)
            
            # Show code execution
            step_msg.content = f"💻 **Step {iteration}/5: Executing Code**\n\n⚙️ Running analysis...\n\n🤔 Evaluating results..."
            await step_msg.update()
            
            # Update status: Reflecting
            current_state = reflector_node(current_state)
            
            # Check if we should continue
            decision = should_continue(current_state)
            
            if decision == "end":
                step_msg.content = f"✅ **Step {iteration} Complete** - Analysis finished!"
                await step_msg.update()
                break
            else:
                # Show we're continuing
                step_msg.content = f"✅ **Step {iteration} Complete**\n\n🔄 Moving to next step..."
                await step_msg.update()
                await cl.sleep(0.5)  # Brief pause so user can see progress
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"⏱️  [{session_id}] Agent workflow completed in {elapsed_time:.2f}s")
        
        # Remove step message
        await step_msg.remove()
        
        # Send the final results
        chart_path = current_state.get("chart_path", "")
        answer = current_state.get("final_answer", current_state.get("reflection", "I couldn't generate an answer."))
        
        if chart_path and os.path.exists(chart_path):
            chart_size = os.path.getsize(chart_path)
            logger.info(f"📤 [{session_id}] Sending chart to user ({chart_size} bytes)")
            
            # Create image element
            image = cl.Image(path=chart_path, name="chart", display="inline")
            
            # Send the final answer with the chart
            await cl.Message(
                content=f"✅ **Analysis Complete** _(took {elapsed_time:.1f}s)_\n\n{answer}",
                elements=[image]
            ).send()
            
            # Clean up chart file after sending
            try:
                os.remove(chart_path)
                logger.debug(f"🗑️  [{session_id}] Chart file cleaned up")
            except Exception as cleanup_error:
                logger.warning(f"⚠️  [{session_id}] Could not clean up chart: {cleanup_error}")
        else:
            logger.info(f"📤 [{session_id}] Sending text response to user")
            # Send just the text answer
            await cl.Message(
                content=f"✅ **Analysis Complete** _(took {elapsed_time:.1f}s)_\n\n{answer}"
            ).send()
        
        logger.info(f"✅ [{session_id}] Response sent successfully")
        
    except Exception as e:
        logger.error(f"❌ [{session_id}] Error in agent workflow:")
        logger.error(traceback.format_exc())
        
        try:
            await step_msg.remove()
        except:
            pass
        
        await cl.Message(
            content=f"❌ An error occurred: {str(e)}\n\nPlease try rephrasing your question."
        ).send()
    finally:
        logger.info(f"🏁 [{session_id}] Message processing completed")
        logger.info("="*80)


if __name__ == "__main__":
    logger.info("🎯 Application ready to accept connections")
    logger.info(f"📝 Logs are being written to: {log_filename}")
    # This allows running with: chainlit run app.py
    pass
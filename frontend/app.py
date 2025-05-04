import streamlit as st
from autogen_ext.models.openai import OpenAIChatCompletionClient
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.progress_bar import progress_manager
import asyncio
from openai import OpenAI
import logging
import warnings
from autogen_agentchat.messages import TextMessage
import threading
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import concurrent.futures

# --- Configuration ---
MAX_HISTORY_TURNS = 20  # Keep last 20 pairs (user+assistant) for context

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add this near the top of your script
warnings.filterwarnings("ignore", message=r"Model .* is not found. The cost will be 0.*")

# --- Streamlit App ---
st.set_page_config(page_title="SwarmCents Chat", page_icon="💰", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: gold;'>
        💰 SwarmCents Chat
    </h1>
    <p style='text-align: center; color: lightgray; font-size: 16px;'>
        Your AI guide to Polymarket predictions 📊🔍
    </p>
""", unsafe_allow_html=True)

# API Keys
OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-2024-08-06")
OPEN_AI_URL = os.environ.get("OPEN_AI_URL", "https://api.openai.com/v1")

# Sidebar
with st.sidebar:
    st.markdown("## 🛠️ How to use")
    st.markdown("1. Choose your Model 🤖\n"
                "2. Ask a question through natural language. 💬")

    st.markdown("---")
    st.markdown("### ⚙️ API Configuration")
    selected_model = st.selectbox("Choose Model", ["gpt-4o", "grok 2", "o3-mini"], index=0)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("💰 <b>SwarmCents Chat</b> helps you explore and analyze predictions on Polymarket using AI.", unsafe_allow_html=True)

# Check for valid API Key & Model
if not OPEN_AI_KEY or not selected_model:
    st.warning("You must provide a valid OpenAI API key and select a model!", icon="⚠️")
    st.stop()

client = OpenAI(
    base_url=OPEN_AI_URL,
    api_key=OPEN_AI_KEY
)

client1 = OpenAIChatCompletionClient(
    model=MODEL_NAME,
    base_url=OPEN_AI_URL,
    api_key=OPEN_AI_KEY,
    model_info={
        "family": "gpt-4o",
        "json_output": True,
        "vision": True,
        "function_calling": True,
    }
)

from backend.Agent import run_prediction_analysis

# --- Initialize Chat History in Session State ---
INITIAL_MESSAGE = [
    {
        "role": "assistant",
        "content": 
"""Hey there, I'm SwarmCents Chat! I'm here to help you analyze predictions about Polymarket topics made on Twitter.
Please select one of these specific options:
1. Find predictions on [specific topic], also give account names of users who made them. 
2. Build profile for [@predictor_handle]. Show me their all prediction history and analysis.
3. Verify prediction: "[exact prediction text]"
4. Calculate the credibility score for [@predictor_handle].
Which option would you like to proceed with?""",
    },
]

# Add a reset button
if st.sidebar.button("🔄 Reset Chat"):
    for key in st.session_state.keys():
        del st.session_state[key]
    # Re-initialize after deleting
    st.session_state.messages = list(INITIAL_MESSAGE)  # Ensure it's a mutable list copy
    st.rerun()  # Force rerun after reset

# Display the chat messages
if "messages" not in st.session_state:
    st.session_state.messages = INITIAL_MESSAGE

# --- Async Utils ---
async def run_async_task(coro):
    """Run an async coroutine and return its result."""
    return await coro

def run_async_function(coro):
    """Execute an async function from a synchronous context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# Initialize chat messages in session state    
if "messages" not in st.session_state:
    st.session_state.messages = list(INITIAL_MESSAGE)  # Use list() to ensure mutable copy

# --- Display Chat History ---
avatar_assistant = None
avatar_user = None
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatar_assistant if message["role"] == "assistant" else avatar_user):
        st.markdown(message["content"])

# --- Add a flag in session state to prevent double input ---
if "is_waiting" not in st.session_state:
    st.session_state.is_waiting = False

# --- Handle User Input ---
prompt_disabled = st.session_state.is_waiting  # Disable input if waiting for a response
prompt = st.chat_input("Type your Query", disabled=prompt_disabled)

# --- Improved Progress Bar Manager ---
class AsyncProgressManager:
    def __init__(self, progress_bar, status_text):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.should_stop = False
        self.current_progress = 0
        self.ctx = get_script_run_ctx()
        self.status_messages = {
            5: "🔄 Initializing...",
            15: "🔍 Searching for predictions...",
            40: "📊 Analyzing market data...",
            65: "💡 Generating insights...",
            85: "✍️ Crafting final response..."
        }
        
    async def update_progress(self):
        add_script_run_ctx(self.ctx)
        while not self.should_stop and self.current_progress <= 100:
            self.progress_bar.progress(self.current_progress)
            
            if self.current_progress in self.status_messages:
                self.status_text.text(self.status_messages[self.current_progress])
                
            await asyncio.sleep(0.9)  # Non-blocking sleep
            self.current_progress += 1
            
            # Cap at 95% while waiting for actual completion
            if self.current_progress > 95:
                self.current_progress = 95
                
    def complete(self):
        self.should_stop = True
        self.progress_bar.progress(100)
        self.status_text.text("✅ Done!")
        
    def error(self):
        self.should_stop = True
        self.status_text.text("❌ Error occurred")

# --- Combined Analysis and Progress Function ---
async def run_analysis_with_progress(text_messages, progress_bar, status_text):
    """Run the analysis and update progress simultaneously."""
    # Create progress manager
    progress_mgr = AsyncProgressManager(progress_bar, status_text)
    
    try:
        # Start progress updates in a separate task
        progress_task = asyncio.create_task(progress_mgr.update_progress())
        
        # Run the actual analysis in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # Run the potentially blocking analysis in a thread pool
            response = await loop.run_in_executor(
                pool, 
                lambda: run_async_function(run_prediction_analysis(text_messages))
            )
        
        # Mark progress as complete
        progress_mgr.complete()
        
        # Cancel progress task
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
            
        return response
        
    except Exception as e:
        progress_mgr.error()
        if 'progress_task' in locals():
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass
        raise e

# --- Handle User Input ---
if prompt:
    # --- Set waiting flag to True ---
    st.session_state.is_waiting = True

    # 1. Append user message to FULL history (for display)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatar_user):
        st.markdown(prompt)

    # 2. Prepare the LIMITED history to send to the agent
    # Keep only the last N messages (user + assistant pairs)
    history_limit = MAX_HISTORY_TURNS * 2
    limited_history = st.session_state.messages[-history_limit:]

    # Convert the LIMITED history to the format expected by your agent
    text_messages_for_agent = [
        TextMessage(content=m["content"], source=m["role"])
        for m in limited_history
    ]

    # 3. Call the agent with the LIMITED history
    with st.chat_message("assistant", avatar=avatar_assistant):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")
        status_container = st.container()

        with status_container:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("🔄 Initializing...")

            response = None  # Initialize response variable
            
            try:
                # Use the improved combined function
                response = run_async_function(
                    run_analysis_with_progress(
                        text_messages_for_agent, 
                        progress_bar, 
                        status_text
                    )
                )
                
                # Small delay before removing progress elements
                time.sleep(0.5)
                
                # Replace placeholder with actual response
                placeholder.markdown(response)
                
                # Clear the progress elements
                progress_bar.empty()
                status_text.empty()

            except Exception as e:
                st.error(f"An error occurred: {e}")
                response = "Sorry, I encountered an error." # Provide a fallback response
                placeholder.markdown(response)
                print("AN EXCEPTION OCCURRED", e)
                
                # Clear the progress elements
                progress_bar.empty()
                status_text.empty()

    # 4. Append assistant response to FULL history (for display)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # --- Reset waiting flag ---
    st.session_state.is_waiting = False
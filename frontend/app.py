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
import time

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

# Initialize chat messages in session state    
if "messages" not in st.session_state:
    st.session_state.messages = INITIAL_MESSAGE

# --- Helper functions ---
def run_async_function(coro):
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        return loop.run_until_complete(task)
    except RuntimeError:
        # No running loop, so create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Clean up any lingering async generators
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

# --- Process state tracking ---
if "processing_state" not in st.session_state:
    st.session_state.processing_state = {
        "is_processing": False,
        "current_step": 0,
        "total_steps": 5,
        "user_input": None,
        "response": None,
        "error": None,
        "start_time": None
    }

# --- Display Chat History ---
avatar_assistant = None
avatar_user = None
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatar_assistant if message["role"] == "assistant" else avatar_user):
        st.markdown(message["content"])

# --- Define status messages for each step ---
status_messages = [
    "🔄 Initializing...",
    "🔍 Searching for predictions...",
    "📊 Analyzing market data...",
    "💡 Generating insights...",
    "✍️ Crafting final response..."
]

# --- Handle processing state ---
if st.session_state.processing_state["is_processing"]:
    # Display the user's message
    if st.session_state.processing_state["user_input"]:
        # Only show if we haven't added it to messages yet
        if len(st.session_state.messages) == 0 or st.session_state.messages[-1]["role"] != "user" or st.session_state.messages[-1]["content"] != st.session_state.processing_state["user_input"]:
            st.session_state.messages.append({"role": "user", "content": st.session_state.processing_state["user_input"]})
            with st.chat_message("user", avatar=avatar_user):
                st.markdown(st.session_state.processing_state["user_input"])
    
    # Show the assistant is responding
    with st.chat_message("assistant", avatar=avatar_assistant):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")
        status_container = st.container()
        
        with status_container:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create text messages for the agent
            if st.session_state.processing_state["current_step"] == 0:
                # First step - prepare the messages
                history_limit = MAX_HISTORY_TURNS * 2
                limited_history = st.session_state.messages[-history_limit:]
                
                text_messages_for_agent = [
                    TextMessage(content=m["content"], source=m["role"])
                    for m in limited_history
                ]
                
                # Store in session state for next steps
                st.session_state.processing_state["agent_messages"] = text_messages_for_agent
            
            # Update progress based on current step
            current_step = st.session_state.processing_state["current_step"]
            total_steps = st.session_state.processing_state["total_steps"]
            
            # Calculate progress percentage
            progress_pct = int((current_step / total_steps) * 100)
            progress_bar.progress(progress_pct)
            
            # Update status message
            if current_step < len(status_messages):
                status_text.text(status_messages[current_step])
            else:
                status_text.text("Almost there...")
            
            # Execute the appropriate step based on current_step
            if current_step == 0:
                # Initialization step - just advance to next step
                st.session_state.processing_state["current_step"] = 1
                st.rerun()
                
            elif current_step < 4:
                # Intermediate steps - simulate processing
                time.sleep(1)  # Simulate work
                st.session_state.processing_state["current_step"] += 1
                st.rerun()
                
            elif current_step == 4:
                # Final step - actually call the agent
                try:
                    response = run_async_function(run_prediction_analysis(
                        st.session_state.processing_state["agent_messages"]
                    ))
                    
                    # Store the response
                    st.session_state.processing_state["response"] = response
                    st.session_state.processing_state["current_step"] += 1
                    
                    # Set final progress
                    progress_bar.progress(100)
                    status_text.text("✅ Done!")
                    
                    # Replace placeholder with actual response
                    placeholder.markdown(response)
                    
                    # Add to messages
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Reset processing state
                    st.session_state.processing_state = {
                        "is_processing": False,
                        "current_step": 0,
                        "total_steps": 5,
                        "user_input": None,
                        "response": None,
                        "error": None,
                        "start_time": None
                    }
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    error_response = "Sorry, I encountered an error."
                    
                    # Store the error
                    st.session_state.processing_state["error"] = str(e)
                    st.session_state.processing_state["response"] = error_response
                    
                    # Replace placeholder with error message
                    placeholder.markdown(error_response)
                    
                    # Add to messages
                    st.session_state.messages.append({"role": "assistant", "content": error_response})
                    
                    # Reset processing state
                    st.session_state.processing_state = {
                        "is_processing": False,
                        "current_step": 0,
                        "total_steps": 5,
                        "user_input": None,
                        "response": None,
                        "error": None,
                        "start_time": None
                    }
                
                # Clear the progress elements
                progress_bar.empty()
                status_text.empty()

# --- Handle User Input ---
prompt_disabled = st.session_state.processing_state["is_processing"]
prompt = st.chat_input("Type your Query", disabled=prompt_disabled)

if prompt and not st.session_state.processing_state["is_processing"]:
    # Start processing
    st.session_state.processing_state = {
        "is_processing": True,
        "current_step": 0,
        "total_steps": 5,
        "user_input": prompt,
        "response": None,
        "error": None,
        "start_time": time.time()
    }
    st.rerun()
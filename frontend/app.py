import streamlit as st
from autogen_ext.models.openai import OpenAIChatCompletionClient
import sys
import os
# Removed the duplicate sys.path.append, assuming the first one is correct
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.progress_bar import progress_manager # Keep if used elsewhere, removed in the thread function
import asyncio
from openai import OpenAI
import logging
import warnings
# from autogen_agentchat.messages import TextMessage # Use the one from the backend if needed
import threading
import time
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
import traceback # Added for better error logging

# --- Configuration ---
MAX_HISTORY_TURNS = 20  # Keep last 20 pairs (user+assistant) for context

# Add the project root to Python path (Ensure this path is correct relative to THIS file)
# Assuming this file is in a subdirectory like 'streamlit_app' or 'frontend'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Add this near the top of your script
warnings.filterwarnings("ignore", message=r"Model .* is not found. The cost will be 0.*")
logging.basicConfig(level=logging.INFO) # Added basic logging

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
# Use the selected model from sidebar dynamically
MODEL_NAME = st.session_state.get("selected_model", os.environ.get("MODEL_NAME", "gpt-4o-2024-08-06")) # Default model
OPEN_AI_URL = os.environ.get("OPEN_AI_URL", "https://api.openai.com/v1")

# Sidebar
with st.sidebar:
    st.markdown("## 🛠️ How to use")
    st.markdown("1. Choose your Model 🤖\n"
                "2. Ask a question through natural language. 💬")

    st.markdown("---")
    st.markdown("### ⚙️ API Configuration")
    # Store selected model in session state so it persists across reruns
    st.session_state.selected_model = st.selectbox("Choose Model", ["gpt-4o", "grok 2", "o3-mini"], index=["gpt-4o", "grok 2", "o3-mini"].index(MODEL_NAME) if MODEL_NAME in ["gpt-4o", "grok 2", "o3-mini"] else 0)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("💰 <b>SwarmCents Chat</b> helps you explore and analyze predictions on Polymarket using AI.", unsafe_allow_html=True)

# Check for valid API Key
if not OPEN_AI_KEY:
    st.warning("You must provide a valid OpenAI API key!", icon="⚠️")
    st.stop()

# Initialize OpenAI client (can be done here as it's not model-dependent in this example)
client = OpenAI(
    base_url=OPEN_AI_URL,
    api_key=OPEN_AI_KEY
)

# Initialize Autogen client (Use selected model)
client1 = OpenAIChatCompletionClient(
    model=st.session_state.selected_model, # Use the selected model
    base_url=OPEN_AI_URL,
    api_key=OPEN_AI_KEY,
    model_info={
        "family": "gpt-4o", # Keep family consistent if using gpt-4o features
        "json_output": True,
        "vision": True,
        "function_calling": True,
    }
)

# Import the backend agent function
try:
    # Assuming backend is in a 'backend' folder at project root
    from backend.Agent import run_prediction_analysis
    from autogen_agentchat.messages import TextMessage # Assuming TextMessage is here
except ImportError as e:
    st.error(f"Could not import backend components: {e}")
    st.info(f"Please ensure your project structure includes a 'backend' folder at {PROJECT_ROOT} with an 'Agent.py' file containing 'run_prediction_analysis' and 'autogen_agentchat.messages' is accessible.")
    st.stop()


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

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = list(INITIAL_MESSAGE)
if "is_waiting" not in st.session_state:
    st.session_state.is_waiting = False
if "worker_thread" not in st.session_state:
    st.session_state.worker_thread = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "job_status" not in st.session_state:
    st.session_state.job_status = "idle" # "idle", "running", "finished", "error"


# Add a reset button
if st.sidebar.button("🔄 Reset Chat"):
    # Clear session state except model selection
    st.session_state.messages = list(INITIAL_MESSAGE)
    st.session_state.is_waiting = False
    st.session_state.worker_thread = None
    st.session_state.last_response = None
    st.session_state.job_status = "idle"
    st.rerun() # Force rerun after reset

# --- Display Chat History ---
avatar_assistant = None # Define avatars if you have image files
avatar_user = None

# Display messages up to the point a job might be running
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatar_assistant if message["role"] == "assistant" else avatar_user):
        st.markdown(message["content"])

# --- Handle displaying running job status or final response ---
if st.session_state.job_status == "running":
    with st.chat_message("assistant", avatar=avatar_assistant):
        # These placeholders need to exist in the main script run
        # They will be updated by the background thread using the context
        st.session_state.placeholder = st.empty()
        st.session_state.status_container = st.container()
        with st.session_state.status_container:
             # Recreate the progress bar and status text placeholders on rerun
            st.session_state.progress_bar = st.progress(0)
            st.session_state.status_text = st.empty()

        st.session_state.placeholder.markdown("Thinking...") # Initial state


elif st.session_state.job_status == "finished":
    # The worker thread already updated session_state.messages and session_state.last_response
    # The final message will be displayed by the history loop on the next rerun
    # Clean up state after displaying
    logging.info("Job finished, cleaning up state.")
    st.session_state.last_response = None
    st.session_state.job_status = "idle"
    st.session_state.is_waiting = False # Should already be False from worker thread, but good to ensure
    st.session_state.worker_thread = None # Clean up thread reference

elif st.session_state.job_status == "error":
    # Display error message
     with st.chat_message("assistant", avatar=avatar_assistant):
        st.error(st.session_state.last_response if st.session_state.last_response else "An error occurred.")
        # Clean up state
        st.session_state.last_response = None
        st.session_state.job_status = "idle"
        st.session_state.is_waiting = False
        st.session_state.worker_thread = None


# --- Worker Function to Run in Thread ---
def worker_target(messages_for_agent, client_autogen, client_openai,
                  placeholder_obj, progress_bar_obj, status_text_obj):
    """
    Target function for the background thread.
    Runs the agent, updates UI elements, and triggers rerun.
    """
    try:
        # Get and add Streamlit script run context to this thread
        ctx = get_script_run_ctx()
        if ctx is None:
            logging.error("Failed to get Streamlit script run context in worker thread.")
            # Attempt to set error state even without context for logging
            if 'st' in locals():
                 st.session_state.job_status = "error"
                 st.session_state.last_response = "Internal error: Could not get Streamlit context in thread."
                 # Cannot rerun without context
            return # Exit thread

        add_script_run_ctx(ctx=ctx)
        logging.info("Streamlit context added to worker thread.")

        # --- Mock Progress Updates (Runs in the thread) ---
        status_messages = {
            5: "🔄 Initializing...",
            15: "🔍 Searching for predictions...",
            40: "📊 Analyzing market data...",
            65: "💡 Generating insights...",
            85: "✍️ Crafting final response..."
        }
        
        # Simulate initial progress while waiting for agent startup
        for i in range(0, 90, 5): # Go up to 90% with mock progress
             if i in status_messages:
                 status_text_obj.text(status_messages[i])
             progress_bar_obj.progress(i)
             time.sleep(0.5) # Adjust speed

        # --- Run the actual async task (Blocks this thread) ---
        logging.info("Calling run_prediction_analysis...")
        # Pass necessary clients to the agent function if it needs them
        # Modify run_prediction_analysis to accept these if necessary
        # Assuming run_prediction_analysis takes text messages as input
        response_content = asyncio.run(run_prediction_analysis(messages_for_agent))
        logging.info("run_prediction_analysis completed.")

        # --- Update UI and State After Task Completes (using context) ---
        progress_bar_obj.progress(100)
        status_text_obj.text("✅ Done!")

        # Allow progress text to be seen briefly before removing
        time.sleep(1)

        placeholder_obj.markdown(response_content) # Update the placeholder
        status_text_obj.empty() # Clear status text
        progress_bar_obj.empty() # Clear progress bar

        # Update session state with the assistant's full response
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        st.session_state.last_response = response_content # Store for potential later use/debugging
        st.session_state.job_status = "finished"
        st.session_state.is_waiting = False # Reset the waiting flag

        logging.info("Worker thread finished, calling rerun.")
        # Trigger a rerun of the Streamlit script
        st.rerun()

    except Exception as e:
        logging.error(f"Error in worker thread: {e}")
        traceback.print_exc()
        try:
            # Attempt to update state and UI on error (using context)
            progress_bar_obj.progress(100) # Mark as complete/error state visually
            status_text_obj.text("❌ Error!")
            placeholder_obj.error(f"An error occurred during analysis: {e}")

            st.session_state.last_response = f"An error occurred during analysis: {e}"
            st.session_state.job_status = "error"
            st.session_state.is_waiting = False

            # Append an error message to history
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred during analysis: {e}"})

            st.rerun() # Trigger rerun to display error state
        except Exception as e_rerun:
            logging.error(f"Error during error handling/rerun in worker thread: {e_rerun}")
            traceback.print_exc()
            # If rerun fails, just log the error


# --- Handle User Input ---
prompt_disabled = st.session_state.is_waiting # Disable input if waiting for a response
prompt = st.chat_input("Type your Query", disabled=prompt_disabled)

if prompt and not st.session_state.is_waiting:
    # --- Set waiting flag to True and job status to running ---
    st.session_state.is_waiting = True
    st.session_state.job_status = "running"
    st.session_state.last_response = None # Clear previous response
    st.session_state.worker_thread = None # Clear previous thread reference

    # 1. Append user message to FULL history (for display)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # The history display loop at the top will render this

    # 2. Prepare the LIMITED history to send to the agent
    history_limit = MAX_HISTORY_TURNS * 2
    # Ensure we don't exceed the actual number of messages
    start_index = max(0, len(st.session_state.messages) - history_limit)
    limited_history = st.session_state.messages[start_index:]

    # Convert the LIMITED history to the format expected by your agent
    text_messages_for_agent = [
        TextMessage(content=m["content"], source=m["role"])
        for m in limited_history
    ]

    # 3. Create and start the worker thread
    # Pass the UI element objects created in the 'running' state display block
    # These objects persist across reruns in session_state because we created them
    # in the 'running' state block and store them in session state there.
    # However, a simpler way is to create them *just before* starting the thread
    # and pass the references directly. Let's do that.

    # Add UI elements for the running job *before* starting the thread
    # This ensures they exist for the worker thread to update
    with st.chat_message("assistant", avatar=avatar_assistant):
        # Store references in session state for the worker to find them easily
        st.session_state.placeholder = st.empty()
        st.session_state.status_container = st.container()
        with st.session_state.status_container:
            st.session_state.progress_bar = st.progress(0)
            st.session_state.status_text = st.empty()

        st.session_state.placeholder.markdown("Thinking...") # Initial state

        # Create and start the worker thread
        st.session_state.worker_thread = threading.Thread(
            target=worker_target,
            args=(
                text_messages_for_agent,
                client1, # Pass autogen client
                client, # Pass openai client if needed by agent
                st.session_state.placeholder,
                st.session_state.progress_bar,
                st.session_state.status_text
            )
        )
        st.session_state.worker_thread.daemon = True # Allow the app to exit even if thread is running
        st.session_state.worker_thread.start()

    # The main script now finishes here, allowing Streamlit to render
    # the current state (user message, thinking message, progress bar 0%).
    # The worker thread will update these elements and trigger a rerun when done.


# Optional: Add a check to re-join thread on rerun if necessary (often not needed with st.rerun)
# if st.session_state.worker_thread and st.session_state.worker_thread.is_alive():
#     # This block would execute on subsequent reruns while the thread is still running
#     # You might choose to display the progress bar and status text again here
#     # based on session state, but the worker thread is already updating the original objects.
#     pass

# The script ends here. Streamlit waits for user input or reruns triggered by the worker thread.
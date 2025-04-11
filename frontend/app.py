import streamlit as st
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
import os
import asyncio
from openai import OpenAI
import warnings
from autogen_agentchat.messages import TextMessage
import sys 
import time 

os.environ["AUTOGEN_DEBUG"] = "0"  # Basic debug info
os.environ["AUTOGEN_VERBOSE"] = "0"  # More detailed logging

# import toml
# import os

# # Load secrets from secrets.toml
# secrets = toml.load("C:\Amit_Laptop_backup\Imperial_essentials\AI Society\Hackathon Torus\secrets.toml")

# # Set environment variables
# for key, value in secrets.items():
#     os.environ[key] = value

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add this near the top of your script
warnings.filterwarnings("ignore", message=r"Model .* is not found. The cost will be 0.*")

st.set_page_config(page_title="SwarmCents Chat", page_icon="💰", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: gold;'>
        💰 SwarmCents Chat
    </h1>
    <p style='text-align: center; color: lightgray; font-size: 16px;'>
        Your AI guide to Polymarket predictions 📊🔍
    </p>
""", unsafe_allow_html=True)


# API Keys - Replace with your actual keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_KEY1 = os.environ.get("GROQ_API_KEY")
GROQ_API_KEY2 = os.environ.get("GROQ_API_KEY")

DATURA_API_KEY = os.environ.get("DATURA_API_KEY")
NEWS_API_TOKEN = os.environ.get("NEWS_API_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")

OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-2024-08-06")
DATURA_API_URL = "https://apis.datura.ai/twitter"

with st.sidebar:
    st.markdown("## 🛠️ How to use")
    st.markdown("1. Enter your Model 🤖\n"
                "2. Ask a question about a Polymarket Topic 💬")

    st.markdown("---")
    st.markdown("### ⚙️ API Configuration")
    selected_model = st.selectbox("Choose Model", [MODEL_NAME, "grok 2", "o3-mini"], index=0)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("💰 <b>SwarmCents Chat</b> helps you explore and analyze predictions on Polymarket using AI.", unsafe_allow_html=True)

# Check for valid API Key & Model
if not OPEN_AI_KEY or not selected_model:
    st.warning("You must provide a valid OpenAI API key and select a model!", icon="⚠️")
    st.stop()

client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=OPEN_AI_KEY
)
client1  = OpenAIChatCompletionClient(
    model = MODEL_NAME,
    # base_url="https://api.openai.com/v1",
    api_key=OPEN_AI_KEY,
)

from backend.Agent import run_prediction_analysis

INITIAL_MESSAGE = [
    {
        "role": "assistant",
        "content": 
"Hey there, I'm SwarmCents Chat! I'm here to help you analyze predictions on Polymarket."
"Please select one of these specific options:"
"1. Find predictions on [specific topic], also give account names of users who made them. Try your best to show atleast 10 predictions."
"2. Build profile for [@predictor_handle]. Show me their all prediction history and analysis."
"3. Verify prediction: [exact prediction text]"
"4. Calculate credibility score for [@predictor_handle]"
"Which option would you like to proceed with?",
    },
]

# Add a reset button
if st.sidebar.button("🔄 Reset Chat"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state["messages"] = INITIAL_MESSAGE

# Display the chat messages
if "messages" not in st.session_state:
    st.session_state.messages = INITIAL_MESSAGE

def run_async_function(coro):
    try:
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop in current thread: create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

avatar_assistant = None 
avatar_user = None
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatar_assistant if message["role"] == "assistant" else avatar_user):
        st.markdown(message["content"])

# Streamlit Input Box
if prompt := st.chat_input("Type your Query"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatar_user):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=avatar_assistant):
        placeholder = st.empty()
        placeholder.markdown("*Thinking...*")
        text_messages = [
            TextMessage(content=m["content"], source=m["role"])
            for m in st.session_state.messages
        ]
        # Run the async function safely
        response = run_async_function(run_prediction_analysis(text_messages))
        
        placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
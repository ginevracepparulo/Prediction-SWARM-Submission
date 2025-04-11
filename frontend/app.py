import streamlit as st
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
import json
import requests
import re
from typing import List, Dict, Tuple
import os
import asyncio
from openai import OpenAI
os.environ["AUTOGEN_DEBUG"] = "0"  # Basic debug info
os.environ["AUTOGEN_VERBOSE"] = "0"  # More detailed logging
import warnings
import re
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Add this near the top of your script
warnings.filterwarnings("ignore", message=r"Model .* is not found. The cost will be 0.*")

# Streamlit UI
st.title("Polymarket Prediction Chatbot")

# API Keys - Replace with your actual keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_KEY1 = os.environ.get("GROQ_API_KEY")
GROQ_API_KEY2 = os.environ.get("GROQ_API_KEY")

DATURA_API_KEY = os.environ.get("DATURA_API_KEY")
NEWS_API_TOKEN = os.environ.get("NEWS_API_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")

OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
DATURA_API_URL = "https://apis.datura.ai/twitter"

# Sidebar for API Key and Model Selection
with st.sidebar:
    st.header("API Configuration")
    selected_model = st.selectbox("Model", [MODEL_NAME], index=0)

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

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Streamlit Input Box
if prompt := st.chat_input("Type your Query"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        text_messages = [
            TextMessage(content=m["content"], source=m["role"])
            for m in st.session_state.messages
        ]
        response = asyncio.run(run_prediction_analysis(text_messages=text_messages))

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
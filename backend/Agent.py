from .AutogenWrappers import find_predictions_wrapper, build_profiles_wrapper, verify_prediction_wrapper, calculate_credibility_scores_batch_wrapper
from utils.progress_bar import  progress_manager
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
from autogen_agentchat.messages import ToolCallRequestEvent, ToolCallExecutionEvent, BaseMessage
import os 
import logging
from dotenv import load_dotenv
dotenv_path = "C:\Amit_Laptop_backup\Imperial_essentials\AI Society\Hackathon Torus\.env"
loaded = load_dotenv(dotenv_path=dotenv_path)
if not loaded:
     # Fallback in case it's mounted at root instead
     load_dotenv()

logger = logging.getLogger("app")

# Configure the logging system
logging.basicConfig(level=logging.INFO)

# Initialize the API keys and URLs
OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-2024-08-06")
OPEN_AI_URL = os.environ.get("OPEN_AI_URL","https://api.openai.com/v1")

client1  = OpenAIChatCompletionClient(
    model = MODEL_NAME,
    #base_url = OPEN_AI_URL,
    api_key= OPEN_AI_KEY,
    #model_info={
    #    "family": "unknown",
    #    "json_output": True,
    #    "vision": True,  # This can be kept
    #   "function_calling": True,  # This can be kept
    #}
)

assistant = AssistantAgent(
    name="SwarmcentsHelper",
    # llm_config=llm_config,
    model_client=client1,
    tools=[find_predictions_wrapper, build_profiles_wrapper, verify_prediction_wrapper, calculate_credibility_scores_batch_wrapper],
    system_message="""You are a prediction analysis expert that helps users find, profile, and verify predictions.
    
STRICT RULES YOU MUST FOLLOW:
1. You MUST ONLY use the provided functions - never make up data or predictions
2. You MUST ask clarifying questions if the user request is unclear
3. You MUST verify all predictions before making claims about accuracy
4. You MUST direct the user to choose one of the four function options

AVAILABLE FUNCTIONS:
1. find_predictions(user_prompt) - Finds posts containing predictions on a topic 
2. build_profile(handle) - Builds a profile of a predictor based on their history
3. verify_prediction(prediction_query) - Verifies if a prediction came true
4. calculate_credibility(handle) - Calculates a credibility score for a predictor

Respond ONLY with one of these approaches:
- Ask clarifying questions if needed
- Propose which function to use based on the user's need
- Execute one of the functions with appropriate parameters
- Present results from function calls (never make up data)
""",
reflect_on_tool_use=True,
model_client_stream=True,  # Enable streaming tokens from the model client.
)

"""
async def run_prediction_analysis(text_messages):

    response = await assistant.on_messages(text_messages,
        cancellation_token=CancellationToken(),
    )
    logger.info(response.inner_messages)
    logger.info(response.chat_message)
    return response.chat_message.content
"""
"""
async def run_prediction_analysis(text_messages):
    current_messages = text_messages[:]
    cancellation_token = CancellationToken()

    while True:
        response = await assistant.on_messages(current_messages, cancellation_token=cancellation_token)

        # Add the full output messages from this round
        current_messages.extend(response.inner_messages)

        for m in response.inner_messages:
            role = getattr(m, "role", m.__class__.__name__)
            content = getattr(m, "content", "")
            tool_call_id = getattr(m, "tool_call_id", "")
            logger.info(f"{role}: {content} | Tool Call ID: {tool_call_id}")

        # If assistant responds with tool calls, the next loop handles their execution and response
        if response.chat_message.tool_calls:
            # Keep looping; tool call was made, but not resolved yet
            continue

        # No more tool calls pending - we're done
        return response.chat_message.content
"""

async def run_prediction_analysis(text_messages):
    current_messages = text_messages[:]
    cancellation_token = CancellationToken()

    # if progress_callback:
    #     progress_callback(5, "🔄 Starting analysis...")

    # Track progress stages
    # total_steps = 5  # Adjust based on typical execution flow
    # current_step = 0

    while True:
        try:
           # Update progress for model thinking step          
            # if progress_callback:
            #     progress_value = min(95, (1 / total_steps) * 100)
            #     progress_callback(int(progress_value), "🔍 Analyzing your request...")
            # if progress_callback:
            #     progress_value = min(95, (current_step / total_steps) * 100)
                
            #     if current_step == 1:
            #         progress_callback(int(progress_value), "🔍 Analyzing your request...")
            #     elif current_step == 2:
            #         progress_callback(int(progress_value), "📚 Retrieving information...")
            #     elif current_step == 3:
            #         progress_callback(int(progress_value), "🧮 Processing data...")
            #     elif current_step == 4:
            #         progress_callback(int(progress_value), "💭 Formulating response...")
            #     else:
            #         progress_callback(int(progress_value), "📝 Finalizing results...")

            response = await assistant.on_messages(current_messages, cancellation_token=cancellation_token)

            # Debug/log
            for m in response.inner_messages:
                role = getattr(m, "role", m.__class__.__name__)
                content = getattr(m, "content", str(m))
                tool_call_id = getattr(m, "tool_call_id", "")
                logger.info(f"{role} | Tool Call ID: {tool_call_id}")

            valid_messages = [m for m in response.inner_messages if isinstance(m, BaseMessage)]

            current_messages.extend(valid_messages)

            # Check if tool_calls exist on the assistant's last message
            if hasattr(response.chat_message, "tool_calls") and response.chat_message.tool_calls:
                continue
            
            # Update to 100% when complete
            logging.info(f"run prediction analysis callback: {progress_manager.get_callback()}")
            if progress_manager.get_callback():
                progress_manager.update_progress(100, "✅ Analysis complete!")

            # Safe return
            if isinstance(response.chat_message, BaseMessage):
                return response.chat_message.content
            else:
                logger.info("Invalid chat_message type:", type(response.chat_message))
                return "Sorry, something went wrong while processing the assistant's response."


        except Exception as e:
            logger.info("An exception occurred:", str(e))
            return "Sorry, I encountered an error while processing your request."

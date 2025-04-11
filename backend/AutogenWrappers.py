
from .PredictionFinder import PredictionFinder
from .PredictionVerifier import PredictionVerifier
from .PredictionProfiler import PredictionProfiler
import asyncio
from typing import List
import os 
from openai import OpenAI

# Initialize the API keys and URLs
DATURA_API_KEY = os.environ.get("DATURA_API_KEY")
NEWS_API_TOKEN = os.environ.get("NEWS_API_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")

OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
DATURA_API_URL = "https://apis.datura.ai/twitter"

client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=OPEN_AI_KEY
)

# ============ AUTOGEN INTEGRATION ============
prediction_finder = PredictionFinder(client, DATURA_API_KEY, DATURA_API_URL)
predictor_profiler = PredictionProfiler(client, DATURA_API_KEY, DATURA_API_URL)
prediction_verifier = PredictionVerifier(client, NEWS_API_TOKEN, GOOGLE_API_KEY, GOOGLE_CSE_ID)

# Register the functions with the agents
def find_predictions_wrapper(user_prompt: str):
    """Wrapper for the find_predictions function"""
    print("Finding predictions...")
    return prediction_finder.find_predictions(user_prompt)
    
def build_profiles_wrapper(handles: List[str]):
    """Wrapper for the build_profiles function"""
    print("Building profiles...")
    return asyncio.run(predictor_profiler.build_profiles(handles))

def  calculate_credibility_scores_batch_wrapper(handles: List[str]):
    print("Calculating credibility scores for batch...")
    """Wrapper for the calculate_credibility_scores_batch function"""
    return asyncio.run(predictor_profiler.calculate_credibility_scores_batch(handles, prediction_verifier))

def verify_prediction_wrapper(prediction: str):
    print("Verifying prediction...")    
    """Wrapper for the verify_prediction function"""
    return prediction_verifier.verify_prediction(prediction)

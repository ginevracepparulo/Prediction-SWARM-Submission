import json
from typing import List, Dict
import requests
import re
import os 
from datura_py import Datura
import time
from dotenv import load_dotenv
import logging
dotenv_path = "C:\Amit_Laptop_backup\Imperial_essentials\AI Society\Hackathon Torus\.env"
loaded = load_dotenv(dotenv_path=dotenv_path)
if not loaded:
     # Fallback in case it's mounted at root instead
     load_dotenv()

# Initialise environment variables
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-2024-08-06")

DATURA_API_KEY = os.environ.get("DATURA_API_KEY")

# ============ COMPONENT 3: PREDICTOR VERIFIER ============
class PredictionVerifier:
    """Verifies whether predictions have come true or proven false."""
    
    def __init__(self, groq_client, news_api_token, google_api_key, google_cse_id):
        self.groq_client = groq_client
        self.news_api_token = news_api_token
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.datura = Datura(api_key=DATURA_API_KEY)
    
    def fetch_google_results(self, query: str) -> List[Dict]:
        """Fetch search results from Google Custom Search API."""
        google_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={self.google_api_key}&cx={self.google_cse_id}&num=3"
        
        response = requests.get(google_url)
        # print("Google URL", response)
        if response.status_code == 200:
            data = response.json()
            # print("Google data", data)
            print("Capturing the data", data.get("data", []))
            return data.get("items", [])[:2]
        
        return []

    def generate_search_query2(self, prediction_query: str) -> str:
        """Generate a concise question-style search query from a multi-paragraph prediction tweet."""
        context = """
        You are an expert at analyzing long prediction tweets (2-3 paragraphs) and extracting the core prediction to create concise, question-style search queries for Perplexica.

        Guidelines:
        1. Read the entire tweet carefully, focusing on the main prediction
        2. Identify the key subject, event, and timeframe
        3. Ignore supporting arguments or explanations
        4. Convert the core prediction into a natural-sounding question
        5. Keep it under 15 words when possible

        Examples:
        1. Prediction tweet: 'After analyzing market trends and political indicators, I believe there's a 52% chance that the UK will vote to leave the European Union in the 2016 referendum. This accounts for... [2 more paragraphs]'
        Query: What were the chances of Brexit happening in 2016?

        2. Prediction tweet: 'Considering current polling data and historical trends, my model shows a 30% probability that Donald Trump could win the 2016 US Presidential Election. Factors include... [3 paragraphs]'
        Query: Was Trump likely to win the 2016 election?

        3. Prediction tweet: 'Based on early adoption rates and technology reviews, there's an 80% probability that Apple's iPhone will revolutionize the smartphone industry when it launches in 2007. [2 more paragraphs explaining]'
        Query: Did experts predict iPhone's success in 2007?

        4. Prediction tweet: 'After evaluating team performance and tournament statistics, I estimate India has a 52% chance of winning the T20 Cricket World Cup Final in 2024. The analysis shows... [3 paragraphs]'
        Query: Were India favorites for the 2024 T20 World Cup?

        5. Prediction tweet: 'Cryptocurrency volatility patterns suggest a 40% probability Bitcoin could reach $100,000 by December 2021. My model accounts for... [2 paragraphs of technical analysis]'
        Query: Could Bitcoin hit $100k in 2021?

        Now generate a concise question query (only the question, no extra text) for this prediction tweet:
        """
        
        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prediction_query},
            ],
        )
        
        return completion.choices[0].message.content.strip()

    def generate_polymarket_topic(self, prediction_query: str) -> str:
        """Generate a Polymarket topic from the predictions."""
        context = """You are an expert in identifying Polymarket topics from tweets containing predictions.
        Your task is to extract the Polymarket topic from the prediction.
        Here are some examples of Polymarket topics:
        1. "Hamas will be defeated by end of 2025, Judea and Samaria annexed by end of 2026, minister says"
            Topic: Israel-Hamas conflict
        2. "I predict that the US will not default on its debt in 2024."
            Topic: US debt default
        3. "I believe that the 2024 US Presidential Election will be a  landslide victory for the Democrats."
            Topic: 2024 US Presidential Election
        4. "I predict that the price of Bitcoin will reach $100,000 by the end of 2025."
            Topic: Bitcoin price
        5. "I believe that the 2024 Summer Olympics will be held in Paris."
            Topic: 2024 Summer Olympics
        6. "I predict that US will leave NATO by 2025."
            Topic: US leaving NATO

        Now, given the following user prediction, generate a Polymarket topic. (Just the topic, no additional text or explanation.)"""

        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prediction_query}
            ]
        )
        return completion.choices[0].message.content.strip()

    def generate_polymarket_topic2(self, prediction_query: str) -> str:
        """Generate a Polymarket topic from the predictions."""

        context = """You are an expert in identifying Polymarket topics from tweets containing predictions.
Your task is to extract the Polymarket topic from the prediction. Try to suggest 3 different formulations of the same topic. Each formulation should have a different granularity level of the topic. 

Format your response as a JSON object with each topic formulation as a key-value pair, where the key is an integer starting from 1 and the value is the topic formulation.

Here are some examples:
1.
    Input: 
    "Hamas will be defeated by end of 2025, Judea and Samaria annexed by end of 2026, minister says"
    
    Expected output: 
{
    "1": "Israel-Hamas conflict"
    "2": "Hamas defeat by end of 2025"
    "3": "Israel-Hamas ceasefire by end of 2025"
}

2.
    Input: 
    "I predict that the US will not default on its debt in 2024."

    Expected output:
{   
    "1": "US debt default",
    "2": "US debt ceiling crisis in 2024",
    "3": "US government default risk in 2024"
}
3.  
    Input:
    "I believe that the 2024 US Presidential Election will be a landslide victory for the Democrats."

    Expected output:
{   
    "1": "2024 US Presidential Election",
    "2": "US Presidential Election 2024",
    "3": "US elections 2024"
}
4.
    Input:
    "I predict that the price of Bitcoin will reach $100,000 by the end of 2025."

    Expected output:
{   
    1: "Bitcoin price",
    2: "Bitcoin price prediction",
    3: "Bitcoin price forecast"
}

5. 
    Input:
    "I believe that the 2024 Summer Olympics will be held in Paris."

    Expected output:
{   
    "1": "2024 Summer Olympics",
    "2": "Summer Olympics 2024",
    "3": "Paris Summer Olympics"
}
6. 
    Input:
    "I predict that US will leave NATO by 2025."

    Expected output:
{       
    "1": "US leaving NATO",
    "2": "US NATO membership",
    "3": "US NATO withdrawal"
}
        
Now, analyze following prediction. 
Ensure the response is **valid JSON** with no additional text."""

        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prediction_query}
            ]
        )

        # print("Completion generate polymarket topic", completion.choices[0].message.content)

        return completion.choices[0].message.content.strip()

    def fetch_multiple_news_articles(self, search_queries) -> List[Dict]:
        """Fetch news articles related to the prediction."""
        print("Fetching news articles...")
        raw_output = search_queries
        raw_output = re.sub(r"^```(json)?|```$", "", raw_output).strip()
        # Extract JSON from response
        match = re.search(r"\{(.*)\}", raw_output, re.DOTALL)
        if match:
            raw_output = match.group(0)  # Extract only the JSON content
        # json_content = "{" + match.group(1) + "}"
        
        try:
            analysis = json.loads(raw_output)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse search queries JSON in fetch_multiple_google_results: {e}")
        
        # search_queries_dict = json.loads(search_queries) 
        search_queries_dict = analysis
        articles = []  # Initialize an empty list to store articles
        for query in search_queries_dict.values():
            print(f"Fetching news articles for query: {query}")
            articles = self.fetch_news_articles(query)
            if len(articles) > 0:
                return articles
        # If no articles found, return an empty list
        if len(articles) == 0:
            print("No articles found for any of the search queries in news articles.")
            return []
    
    def fetch_multiple_google_results(self, search_queries) -> List[Dict]:
        """Fetch google results related to the prediction."""
        print("Fetching google results...")

        # try:
        #     search_queries_dict = json.loads(search_queries)
        # except json.JSONDecodeError as e:
        #     raise ValueError(f"Failed to parse search queries JSON in fetch_multiple_google_results: {e}")
        raw_output = search_queries
        raw_output = re.sub(r"^```(json)?|```$", "", raw_output).strip()
        # Extract JSON from response
        match = re.search(r"\{(.*)\}", raw_output, re.DOTALL)
        if match:
            raw_output = match.group(0)  # Extract only the JSON content
        # json_content = "{" + match.group(1) + "}"
        else:
            print("No match found in fetch_multiple_google_results")
            print("raw_output fetch_multiple_google_results::", raw_output)
            return []
        
        try:                
            analysis = json.loads(raw_output)
        except json.JSONDecodeError as e:
            print("raw_output fetch_multiple_google_results Error", raw_output)
            raise ValueError(f"Failed to parse search queries JSON in fetch_multiple_google_results: {e}")
        
        # search_queries_dict = json.loads(search_queries) 
        search_queries_dict = analysis
        articles = []  # Initialize an empty list to store articles
        for query in search_queries_dict.values():
            print("Fetching google results for query:", query)
            articles = self.fetch_google_results(query)
            if len(articles) > 0:
                return articles
        # If no articles found, return an empty list
        if len(articles) == 0:
            print("No articles found for any of the search queries in google results.")
            return []

    def fetch_news_articles(self, search_query: str) -> List[Dict]:
        """Fetch news articles related to the prediction, with up to 5 retries."""

        max_retries = 5
        delay = 1  # Start with 1 second delay

        for attempt in range(1, max_retries + 1):
            try:
                result = self.datura.basic_web_search(
                    query=search_query,
                    num=5,
                    start=1
                )

                data = result.get("data", [])

                print(f"Attempt {attempt}: Captured data ->", data)

                if data:  # If we got results, return them
                    return data

            except Exception as e:
                print(f"Attempt {attempt} failed with error: {e}")

            # If we're not on the last attempt, wait and retry
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2  # Optional: exponential backoff

        # After all retries fail
        print("All attempts failed. Returning empty list.")
        return []

    def analyze_verification(self, prediction_query: str, all_sources: List[Dict]) -> Dict:
        """Analyze the sources to determine if the prediction was accurate."""
        article_summaries = "\n".join(
            [f"Title: {src['title']}, Source: {src['source']}, Description: {src['description']}" for src in all_sources]
        )

        print("Okay analyze_verification")
        system_prompt = """
        You are an AI analyst verifying predictions for Polymarket, a prediction market where users bet on real-world outcomes. Your task is to classify claims as TRUE, FALSE, or UNCERTAIN *only when evidence is insufficient*.

        ### Rules:
        1. *Classification Criteria*:
        - ⁠ TRUE ⁠: The news articles *conclusively confirm* the prediction happened (e.g., "Bill passed" → voting records show it passed).
        - ⁠ FALSE ⁠: The news articles *conclusively disprove* the prediction (e.g., "Company will move HQ" → CEO denies it).
        - ⁠ UNCERTAIN ⁠: *Only if* evidence is missing, conflicting, or outdated (e.g., no articles after the predicted event date).

        2. *Evidence Standards*:
        - Prioritize *recent articles* (within 7 days of prediction date).
        - Trust *primary sources* (government releases, official statements) over opinion pieces.
        - Ignore irrelevant or off-topic articles.

        3. *Conflict Handling*:
        - If sources conflict, weigh authoritative sources (e.g., Reuters) higher than fringe outlets.
        - If timing is unclear (e.g., "will happen next week" but no update), default to ⁠ UNCERTAIN ⁠.
        
        """       

        analysis_prompt = f"""
        The prediction is: "{prediction_query}". 

        Here are some recent news articles about this topic:
        {article_summaries}

        Based on this data, determine if the prediction was accurate. 
        Summarize the key evidence and provide the output in *JSON format* with the following structure:

        {{
          "result": "TRUE/FALSE/UNCERTAIN",
          "summary": "Brief explanation of why the claim is classified as TRUE, FALSE, or UNCERTAIN based on the news articles."
        }}

        Ensure the response is *valid JSON* with no additional text.
        """
        
        ai_verification = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_prompt},
            ],
        )
        
        match = re.search(r"\{(.*)\}", ai_verification.choices[0].message.content, re.DOTALL)
        if match:
            print("Match found")
            ai_verification_result = "{" + match.group(1) + "}"
            try:
                return json.loads(ai_verification_result)
            except json.JSONDecodeError:
                return {
                    "result": "UNCERTAIN",
                    "summary": "Could not analyze the prediction due to formatting issues."
                }
        else:
            return {
                "result": "UNCERTAIN",
                "summary": "Could not analyze the prediction due to formatting issues."
            }
    
    def verify_prediction(self, prediction_query: str) -> Dict:
        """Main method to verify a prediction."""
        # Generate search query
        # search_query = self.generate_search_query(prediction_query)
        # search_query = self.generate_polymarket_topic(prediction_query)
        search_queries = self.generate_polymarket_topic2(prediction_query)

        print(f"search_queries", search_queries)
        # print(f"Generated Search Query: {search_query}")
        
        # Fetch news articles
        # articles = self.fetch_news_articles(search_query)
        articles = self.fetch_multiple_news_articles(search_queries)
        # Fetch Google search results
        # google_results = self.fetch_google_results(search_query)
        google_results = self.fetch_multiple_google_results(search_queries)

        # Prepare sources from both APIs
        all_sources = [
            {"title": a['title'], "source": a['link'], "description": a['snippet']} for a in articles
        ] + [
            {"title": g['title'], "source": g['link'], "description": g['snippet']} for g in google_results
        ]

        # all_sources = [
        #     {"title": g['title'], "source": g['link'], "snippet": g['snippet']} for g in google_results
        # ]
        # print("all_sources", all_sources)
        if not all_sources:
            return {
                "result": "UNCERTAIN",
                "summary": "No relevant information found to verify this prediction.",
                "sources": []
            }
        # print("articles", articles)
        print("all_sources", len(all_sources))
        # Analyze verification
        verification_data = self.analyze_verification(prediction_query, all_sources)
        print("Final result")
        # Final result
        final_result = {
            "result": verification_data["result"],
            "summary": verification_data["summary"],
            "sources": all_sources
        }
        
        return final_result
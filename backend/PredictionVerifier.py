import json
from typing import List, Dict
import requests
import re
import os 

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-2024-08-06")
# ============ COMPONENT 3: PREDICTOR VERIFIER ============

class PredictionVerifier:
    """Verifies whether predictions have come true or proven false."""
    
    def __init__(self, groq_client, news_api_token, google_api_key, google_cse_id):
        self.groq_client = groq_client
        self.news_api_token = news_api_token
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
    
    def fetch_google_results(self, query: str) -> List[Dict]:
        """Fetch search results from Google Custom Search API."""
        google_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={self.google_api_key}&cx={self.google_cse_id}&num=3"
        
        response = requests.get(google_url)
        if response.status_code == 200:
            data = response.json()
            return data.get("items", [])
        return []
    
    def generate_search_query(self, prediction_query: str) -> str:
        """Generate a search query for news APIs based on the prediction."""
        context = """
        You are an expert in constructing search queries for news APIs to find relevant articles related to political predictions.
        Your task is to generate a properly formatted query for searching news related to a given prediction.

        Examples:
        1. Prediction: 'Chances of UK leaving the European Union in 2016 was 52%'
           Query: Brexit, UK, European Union, 2016

        2. Prediction: 'Chances of Donald Trump winning the 2016 US Presidential Election was 30%'
           Query: Donald Trump, elections, 2024

        3. Prediction: 'Chances of Apple's iPhone revolutionizing the smartphone industry in 2007 was 80%'
           Query: Apple, iPhone, smartphone, 2007

        4. Prediction: 'Chances of India winning T20 Cricket WorldCup Final in 2024 was 52%'
           Query: India, T20, Cricket World Cup, Winner, 2024

        5. Prediction: 'Chances of Bitcoin reaching $100,000 in 2021 was 40%'
           Query: Bitcoin, price, cryptocurrency, $100,000, 2021
        Now, generate a query for the following prediction: (Only generate query and no additional text or explanation.)
        """
        
        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prediction_query},
            ],
        )
        
        return completion.choices[0].message.content.strip()
    
    def fetch_news_articles(self, search_query: str) -> List[Dict]:
        """Fetch news articles related to the prediction."""
        encoded_keywords = re.sub(r'[^\w\s]', '', search_query).replace(' ', '+')
        
        news_url = (
            f"https://api.thenewsapi.com/v1/news/all?"
            f"api_token={self.news_api_token}"
            f"&search={encoded_keywords}"
            f"&search_fields=title,main_text,description,keywords"
            f"&language=en"
            f"&published_after=2024-01-01"
            f"&sort=relevance_score"
        )
        
        news_response = requests.get(news_url)
        if news_response.status_code == 200:
            news_data = news_response.json()
            return news_data.get("data", [])
        return []
    
    def analyze_verification(self, prediction_query: str, all_sources: List[Dict]) -> Dict:
        """Analyze the sources to determine if the prediction was accurate."""
        article_summaries = "\n".join(
            [f"Title: {src['title']}, Source: {src['source']}, Description: {src['description']}, Snippet: {src['snippet']}" for src in all_sources]
        )

        system_prompt = """
        You are an AI analyst verifying predictions for Polymarket, a prediction market where users bet on real-world outcomes. Your task is to classify claims as TRUE, FALSE, or UNCERTAIN **only when evidence is insufficient**.

        ### Rules:
        1. **Classification Criteria**:
        - `TRUE`: The news articles **conclusively confirm** the prediction happened (e.g., "Bill passed" → voting records show it passed).
        - `FALSE`: The news articles **conclusively disprove** the prediction (e.g., "Company will move HQ" → CEO denies it).
        - `UNCERTAIN`: **Only if** evidence is missing, conflicting, or outdated (e.g., no articles after the predicted event date).

        2. **Evidence Standards**:
        - Prioritize **recent articles** (within 7 days of prediction date).
        - Trust **primary sources** (government releases, official statements) over opinion pieces.
        - Ignore irrelevant or off-topic articles.

        3. **Conflict Handling**:
        - If sources conflict, weigh authoritative sources (e.g., Reuters) higher than fringe outlets.
        - If timing is unclear (e.g., "will happen next week" but no update), default to `UNCERTAIN`.
        
        """       

        analysis_prompt = f"""
        The prediction is: "{prediction_query}". 

        Here are some recent news articles about this topic:
        {article_summaries}

        Based on this data, determine if the prediction was accurate. 
        Summarize the key evidence and provide the output in **JSON format** with the following structure:

        {{
          "result": "TRUE/FALSE/UNCERTAIN",
          "summary": "Brief explanation of why the claim is classified as TRUE, FALSE, or UNCERTAIN based on the news articles."
        }}

        Ensure the response is **valid JSON** with no additional text.
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
        search_query = self.generate_search_query(prediction_query)
        print(f"Generated Search Query: {search_query}")
        
        # Fetch news articles
        articles = self.fetch_news_articles(search_query)
        
        # Fetch Google search results
        google_results = self.fetch_google_results(prediction_query)
        
        # Prepare sources from both APIs
        all_sources = [
            {"title": a['title'], "source": a['source'], "published": a['published_at'], "description": a['description'], "snippet": a['snippet']} for a in articles
        ] + [
            {"title": g['title'], "source": g['link'], "snippet": g['snippet'], "description": g.get('pagemap', {}).get('metatags', [{}])[0].get('og:description', '') if 'pagemap' in g else "", "published": "N/A"} for g in google_results
        ]
        
        if not all_sources:
            return {
                "result": "UNCERTAIN",
                "summary": "No relevant information found to verify this prediction.",
                "sources": []
            }
        
        # Analyze verification
        verification_data = self.analyze_verification(prediction_query, all_sources)
        
        # Final result
        final_result = {
            "result": verification_data["result"],
            "summary": verification_data["summary"],
            "sources": all_sources
        }
        
        return final_result
    
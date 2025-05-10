import json
from typing import List, Dict, Tuple
import requests
import re
import os
import asyncio
import logging

logger = logging.getLogger("prediction_finder")

# Configure the logging system
logging.basicConfig(level=logging.INFO)

# Initialize environment variables
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-2024-08-06")

class PredictionFinder:
    """Finds tweets containing predictions about specified topics."""
    
    def __init__(self, groq_client, datura_api_key, datura_api_url):
        self.groq_client = groq_client
        self.datura_api_key = datura_api_key
        self.datura_api_url = datura_api_url
    
    def generate_search_query(self, user_prompt: str) -> str:
        """Generate a properly formatted search query from user prompt."""
        context = """You are an expert in constructing search queries for the Datura API to find relevant tweets related to Polymarket predictions.
        Your task is to generate properly formatted queries based on user prompts.

        Here are some examples of well-structured Datura API queries:

        1. I want mentions of predictions about the 2024 US Presidential Election, excluding Retweets, with at least 20 likes.  
        Query: (President) (elections) (USA) min_faves:20  

        2. I want tweets from @Polymarket users discussing cryptocurrency price predictions, excluding tweets without links.  
        Query: (Bitcoin) (Ethereum) (crypto) 

        3. I want tweets predicting the outcome of the Wisconsin Supreme Court election between Susan Crawford and Brad Schimel.  
        Query: (Wisconsin) (SupremeCourt) (Crawford) (Schimel) (election)

        4. I want tweets discussing AI stock price predictions in 2025  
        Query: (AI) (tech) (stock)

        5. I want mentions of predictions about the winner of the 2025 NCAA Tournament.  
        Query: (NCAA) (MarchMadness) (2025) (winner) 

        6. I want tweets discussing whether Yoon will be out as president of South Korea before May.  
        Query: (Yoon) (SouthKorea) (president) (resign) (before May) 

        Now, given the following user prompt, generate a properly formatted Datura API query. (Just the query, no additional text or explanation.)"""

        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_prompt}
            ]
        )
        return completion.choices[0].message.content.strip()

    def generate_polymarket_topic(self, user_prompt: str) -> str:
        """Generate a Polymarket topic from the user prompt."""
        context = """You are an expert in identifying Polymarket topics from user prompts.
        Your task is to extract the Polymarket topic from the user prompt.
        Here are some examples of Polymarket topics:
        1. I want mentions of predictions about the 2024 US Presidential Election, excluding Retweets, with at least 20 likes.
            Topic: 2024 US Presidential Election
        2. I want tweets from @Polymarket users discussing cryptocurrency price predictions, excluding tweets without links.
            Topic: cryptocurrency price predictions
        3. I want tweets predicting the outcome of the Wisconsin Supreme Court election between Susan Crawford and Brad Schimel.
            Topic: Wisconsin Supreme Court election
        4. I want tweets discussing AI stock price predictions in 2025
            Topic: AI stock price predictions
        5. I want mentions of predictions about the winner of the 2025 NCAA Tournament.
            Topic: 2025 NCAA Tournament
        6. I want tweets discussing whether Yoon will be out as president of South Korea before May.
            Topic: Yoon out as president of South Korea before May

        Now, given the following user prompt, generate a Polymarket topic. (Just the topic, no additional text or explanation.)"""

        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    
    async def get_tweets(self, user_prompt: str, min_likes: int = 0, count: int = 100, max_retries = 5) -> List[Dict]:
        #Fetch tweets from Datura API based on the generated query.

        payload = {
            "prompt": user_prompt,
            "model": "HORIZON",
            "start_date": "2024-04-10",
            "lang": "en",
            "verified": False,
            "blue_verified": False,
            "is_quote": False,
            "is_video": False,
            "is_image": False,
            "min_retweets": 0,
            "min_replies": 0,
            "min_likes": min_likes,
            "count": count
        }
        
        headers = {
            "Authorization": self.datura_api_key,
            "Content-Type": "application/json"
        }
        
        for attempt in range(max_retries):
            try:
                #print(f"🔁 Attempt {attempt + 1} to fetch tweets...")
                response = await asyncio.to_thread(requests.post, url=self.datura_api_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                tweets_ls = data.get("miner_tweets", [])
                #print(tweets_ls)
                #print(len(tweets_ls), "tweets found")
                if len(tweets_ls) > 0:
                    return tweets_ls
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries-1:
                    print("🚫 Max retries reached. Returning error.")
                    return []
            
            print(f"Attempt {attempt + 1} failed. Retrying...")
            await asyncio.sleep(2)
        
        print("🚫 Max retries reached. Returning error.")
        return {"error": "Invalid Username. No tweets found after 5 attempts.", "tweets": []}

    def process_tweets(self, tweets: List[Dict]) -> Tuple[Dict, Dict]:
        """Process tweets to create structured data."""
        hash_dict = {}
        username_to_tweet = {}
        
        for tweet in tweets:
            user_id = tweet["user"]["id"]
            
            hash_dict[user_id] = {
                "username": tweet["user"]["username"],
                "favourites_count": tweet["user"]["favourites_count"],
                "is_blue_verified": tweet["user"]["is_blue_verified"],
                "tweet_text": tweet["text"],
                "like_count": tweet["like_count"],
                "created_at": tweet["created_at"],
                "tweet url": tweet["url"],
            }
            
            username_to_tweet[tweet["user"]["username"]] = tweet["text"]
        
        return hash_dict, username_to_tweet
    
    def analyze_predictionsold(self, username_to_tweet: Dict) -> str:
        """Analyze tweets to identify predictions."""
        json_string = json.dumps(username_to_tweet, indent=4)
        
        context = """You are an expert in identifying explicit and implicit predictions in tweets related to Polymarket topics.

Here is a JSON object containing tweets, where each key represents a unique tweet ID and the value is the tweet text.

Your task:  
For each tweet, determine if it contains an **explicit or implicit prediction** about a future event **related to a Polymarket topic**.  
- If it **does**, return "Yes".  
- If it **does not**, return "No".  

Format your response as a JSON object with each tweet ID mapped to "Yes" or "No".

Example:

Input JSON:
{
    "username1": "Bitcoin will hit $100K by the end of 2025!",
    "username2": "The economy is in trouble. People are struggling.",
    "username3": "I bet Trump wins the next election."
}

Expected Output:
{
    "username1": "Yes",
    "username2": "No",
    "username3": "Yes"
}

Now, analyze the following tweets and generate the output: 
Ensure the response is **valid JSON** with no additional text.
"""

        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": json_string}
            ]
        )
        
        return completion.choices[0].message.content

    def analyze_predictions(self, username_to_tweet: Dict, poly_topic: str) -> str:
        """Analyze tweets to identify predictions."""

        input_dict = {"Polymarket Topic": poly_topic, "tweets": username_to_tweet}

        print("input_dict", input_dict)

        json_string = json.dumps(input_dict, indent=4)
        
        context = """You are an expert in identifying explicit and implicit predictions in tweets related to a Polymarket topic.

Here is a JSON object containing a polymarket topic and tweets that should be related to it, where the first key is the string "Polymarket Topic" and the value is the Polymarket topic description, subsequently each key represents a unique tweet ID and the value is the tweet text.

Your task:  
For each tweet, determine if it contains an **explicit or implicit prediction** about a future event **related to the specified Polymarket topic**.  
- If it **does**, return "Yes".  
- If it **does not**, return "No".  

Consider:
- that the tweet may be related to the Polymarket topic but not contain a prediction, in which case you should return "No"
- that the tweet may contain a prediction but not be related to the Polymarket topic, in which case you should also return "No"
- that the tweet may contain a prediction and be related to the Polymarket topic, in which case you should return "Yes"
- that the tweet may not be related to the Polymarket topic and not contain a prediction, in which case you should also return "No"

Format your response as a JSON object with each tweet ID mapped to "Yes" or "No".

Example:

Input JSON:
{"Polymarket Topic": "2026 CONCACAF World Cup Qualifiers", tweets:

{
"username1": "Miami FC trio Gerald Diaz, Nico Cardona, and Ricardo Rivera have been called up to the Puerto Rico national team for the 2026 CONCACAF World Cup Qualifiers.",
"username2": "Another @FPFPuertoRico ☎️🆙 for Gerald Diaz, Nico Cardona, and Ricardo Rivera as Puerto Rico competes in the CONCACAF Qualifiers for 2026 🇵🇷 🙌 #vamosmiami",
"username3": "I bet Canada will be the group winner in the CONCACAF Championship"
}

}

Expected Output:
{
"username1": "No",
"username2": "No",
"username3": "Yes"
}

Now, analyze the following tweets.
Ensure the response is **valid JSON** with no additional text.
""" 
        
        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": json_string}
            ]
        )
        
        return completion.choices[0].message.content

    def filter_tweets_by_prediction(self, yes_no: str, hash_dict: Dict) -> str:
        """Filter tweets to only include those with predictions."""
        match_yes_no = re.search(r"\{(.*)\}", yes_no, re.DOTALL)
        json_content_yes_no = "{" + match_yes_no.group(1) + "}"
        
        yes_no_dict = json.loads(json_content_yes_no)
        
        filtered_tweets = {
            tweet_id: details
            for tweet_id, details in hash_dict.items()
            if details["username"] in yes_no_dict and yes_no_dict[details["username"]] == "Yes"
        }
        
        return json.dumps(filtered_tweets, indent=4)
    
    async def find_predictionsold(self, user_prompt: str) -> Dict:
        """Main method to find predictions based on user prompt."""

        print(f"Generated Search Query: {user_prompt}")
        
        # Get tweets
        tweets = await self.get_tweets(user_prompt)
        
        if not tweets:
            return {"error": "No tweets found matching the criteria"}

        # Process tweets
        hash_dict, username_to_tweet = self.process_tweets(tweets)
        
        print("DEBUGGING hash_dict")
        print(f"Fetched {len(hash_dict)} tweets")

        # Analyze predictions
        prediction_analysis = self.analyze_predictions(username_to_tweet)

        # Filter tweets
        filtered_predictions = self.filter_tweets_by_prediction(prediction_analysis, hash_dict)

        # Return as dictionary
        return json.loads(filtered_predictions)

    async def find_predictions(self, user_prompt: str) -> Dict:
        """Main method to find predictions based on user prompt."""

        print(f"Generated Search Query: {user_prompt}")
        
        # Get Polymarket topic
        poly_topic = self.generate_polymarket_topic(user_prompt)
        print(f"Generated Polymarket Topic: {poly_topic}")
        
        # Get tweets
        tweets = await self.get_tweets(user_prompt)
        
        if not tweets:
            return {"error": "No tweets found matching the criteria"}
        
        # Process tweets
        hash_dict, username_to_tweet = self.process_tweets(tweets)

        print(f"Fetched {len(hash_dict)} tweets")

        # Analyze predictions
        prediction_analysis = self.analyze_predictions(username_to_tweet, poly_topic)

        print("Prediction Analysis:", prediction_analysis)

        # Filter tweets
        filtered_predictions = self.filter_tweets_by_prediction(prediction_analysis, hash_dict)
        
        print("Filtered Predictions:", filtered_predictions)

        # Return as dictionary
        return json.loads(filtered_predictions)
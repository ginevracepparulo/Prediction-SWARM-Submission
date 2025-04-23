from supabase import create_client, Client
import os
from dotenv import load_dotenv
load_dotenv()  

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

class Database():

    def __init__(self):
        self.supabase = create_client(url, key)

    def insert_profile(self,profile):
        handle = profile["handle"]
        total_tweets_analyzed = profile["total_tweets_analyzed"]
        prediction_tweets = profile["prediction_tweets"]
        analysis = profile["analysis"]
        row = {
            "handle": handle,
            "total_tweets_analyzed": total_tweets_analyzed,
            "prediction_tweets": prediction_tweets,
            "analysis": analysis
        }
        response = self.supabase.table("Predictor Profiles").insert(row).execute()
        return response

    def select_profile(self,handle):
        response = self.supabase.table("Predictor Profiles").select("*").eq("handle", handle).execute()
        print("Response:", response)
        return response

    def fetch_profiles(self):
        response = self.supabase.table("Predictor Profiles").select("*").execute()
        return response.data

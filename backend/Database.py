# from supabase import create_client, Client
import os
from dotenv import load_dotenv
import pymongo
import sys
import logging
dotenv_path = "C:\Amit_Laptop_backup\Imperial_essentials\AI Society\Hackathon Torus\.env"
loaded = load_dotenv(dotenv_path=dotenv_path)
if not loaded:
     # Fallback in case it's mounted at root instead
     load_dotenv()
logger = logging.getLogger("app")
# url = os.environ.get("SUPABASE_URL")
# key = os.environ.get("SUPABASE_KEY")

MongodbClient = os.environ.get("MONGODB_CLIENT")
class Database():

    def __init__(self):
        # self.supabase = create_client(url, key)
        self.mongodb = pymongo.MongoClient(MongodbClient)
        self.db = self.mongodb["UserProfileDB"]

        # Default collection name
        self.collection_name = "UserProfile"

        self.mongo_collection = self.db[self.collection_name]


    def insert_profile(self,profile):
        handle = profile["handle"]
        total_tweets_analyzed = profile["total_tweets_analyzed"]
        prediction_tweets = profile["prediction_tweets"]
        prediction_count = profile["prediction_count"]
        prediction_rate = profile["prediction_rate"]
        analysis = profile["analysis"]
        row = {
            "handle": handle,
            "total_tweets_analyzed": total_tweets_analyzed,
            "prediction_tweets": prediction_tweets,
            "prediction_count": prediction_count,
            "prediction_rate": prediction_rate,
            "analysis": analysis
        }

        # The document to insert will have key = handle, value = row
        document = {
            handle: row
        }


        # Check if collection exists, if not create a default collection
        if self.collection_name not in self.db.list_collection_names():
            print(f"Collection {self.collection_name} does not exist. Creating it now.")
            self.mongo_collection = self.db[self.collection_name]  # Create the collection on demand
        
        # Step 1: Check collection size
        stats = self.db.command("collstats", self.mongo_collection.name)
        size_in_mb = stats['size'] / (1024 * 1024)
        print(f"Current collection size: {size_in_mb:.2f} MB")

        if size_in_mb > 500:  # Approaching 512 MB
            print("Storage limit nearing! Deleting oldest 5 documents...")

            # Find the oldest documents (sorted by _id) and delete them
            oldest_docs = self.mongo_collection.find().sort("_id", 1).limit(5)
            ids_to_delete = [doc["_id"] for doc in oldest_docs]

            print(f"IDs to delete: {ids_to_delete}")
            # Delete the oldest documents
            self.mongo_collection.delete_many({"_id": {"$in": ids_to_delete}})
        
        # Step 2: Insert the new document
        result = self.mongo_collection.insert_one(document)
        print("Inserted document with ID:", result.inserted_id)

        return result

    def select_profile(self, handle):
        # Query MongoDB using the handle to find the profile
        logger.info("Runninhg select_profile")
        result = self.mongo_collection.find_one({"handle": handle})

        logger.info(f"result {result}")
        if result:
            # Extract the relevant data and return the specified structure
            profile_data = result.get(handle)  # Since handle is used as the key
            if profile_data:
                # Return only the fields you're interested in
                return {
                    "handle": profile_data["handle"],
                    "total_tweets_analyzed": profile_data["total_tweets_analyzed"],
                    "prediction_tweets": profile_data["prediction_tweets"],
                    "prediction_count": profile_data["prediction_count"],
                    "prediction_rate": profile_data["prediction_rate"],
                    "analysis": profile_data["analysis"]
                }
            else:
                print(f"No profile data found for handle: {handle}")
                return None
        else:
            print(f"No profile found for handle: {handle}")
            return None

      

    # def fetch_profiles(self):
    #     response = self.supabase.table("Predictor Profiles").select("*").execute()
    #     return response.data

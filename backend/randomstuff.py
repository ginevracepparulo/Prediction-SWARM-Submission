def analyze_predictions(username_to_tweet, poly_topic) -> str:
    """Analyze tweets to identify predictions."""
    

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

Now, analyze the following tweets related to the following Polymarket topic: %s and generate the output: 
Ensure the response is **valid JSON** with no additional text.
""" % poly_topic
    try:
        print("context", context)
    except Exception as e:
        print("Error in context", e)
    return 

prediction_analysis = analyze_predictions({"elon":"USA will win 50 medals"}, "2030 olympics")
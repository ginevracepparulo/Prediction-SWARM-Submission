# import requests
# from substrateinterface import Keypair

# url = "https://memory.sension.torus.directory/api/auth/challenge"
# WALLET_ADDRESS = "5DetSJZ3mSCk5bpaP98NCAVN8FqU7aB4aqFXtJMc5PFbuUzk"
# WALLET_SEED_PHRASE = "ranch grant sunset body purse elite top furnace develop observe hobby license"  # Replace with your actual seed phrase

# # 1. Get challenge
# challenge_resp = requests.post(
#     "https://memory.sension.torus.directory/api/auth/challenge",
#     json={"wallet_address": WALLET_ADDRESS},
#     headers={"Content-Type": "application/json"}
# )
# challenge_data = challenge_resp.json()
# message = challenge_data["message"]
# challenge_token = challenge_data["challenge_token"]

# print("Challenge request: ", challenge_resp.json())

# # 2. Sign the message (replace with your wallet's signing method)
# wallet = Keypair.create_from_mnemonic(WALLET_SEED_PHRASE)
# signature = wallet.sign(message.encode("utf-8")).hex()

# # 3. Verify signature
# verify_resp = requests.post(
#     "https://memory.sension.torus.directory/api/auth/verify",
#     json={
#         "challenge_token": challenge_token,
#         "signature": signature
#     },
#     headers={"Content-Type": "application/json"}
# )

# print("Verify response: ", verify_resp.json())
# session_token = verify_resp.json().get("session_token")

# print("Session token: ", session_token)
# # 4. Use session token for authenticated requests
# headers = {
#     "Authorization": f"Bearer {session_token}",
#     "Content-Type": "application/json"
# }
# # response = requests.get("YOUR_API_ENDPOINT", headers=headers)

# url_insert = "https://memory.sension.torus.directory/api/predictions/insert"

# payload = {
#     "content": "Bitcoin will reach $100k by 2026",
#     "prediction_timestamp": "2025-06-20T12:00:00Z",
#     "predictor_twitter_username": "your_twitter",
#     "topic": "crypto",
#     "url": "https://twitter.com/your_twitter/status/123456789"
# }


# response = requests.post(url_insert, json=payload, headers=headers)

# print(response)
import requests

url = "https://api.desearch.ai/desearch/ai/search/links/twitter"

payload = {
    "model": "NOVA",
    "prompt": "Israel Hamas ceasefire before June 2025"
}
payload = {
    "prompt": "Israel Hamas ceasefire before June 2025",
    "model": "NOVA",
    "start_date": "2024-04-10",
    "lang": "en",
    "verified": False,
    "blue_verified": False,
    "is_quote": False,
    "is_video": False,
    "is_image": False,
    "min_retweets": 0,
    "min_replies": 0,
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": "dt_$X6oACKtNOE_2RL984Dg-C8Ds6HZmsQLA4N7ez3NysVg"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)

# headers = {
#     "Authorization": "dt_$X6oACKtNOE_2RL984Dg-C8Ds6HZmsQLA4N7ez3NysVg",
#     "Content-Type": "application/json",
# }

# params = {
#     "user": 'elonmusk',
#     "query": "until:2024-9-28",
#     "count": 50
# }

# response = requests.get(
#     "https://api.desearch.ai/twitter",
#     headers=headers,
#     params=params
# )
# response.raise_for_status()
# tweets_ls = response.json()

# print(tweets_ls)
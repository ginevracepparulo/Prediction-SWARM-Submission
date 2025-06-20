import requests
from substrateinterface import Keypair

url = "https://memory.sension.torus.directory/api/auth/challenge"
WALLET_ADDRESS = "5DetSJZ3mSCk5bpaP98NCAVN8FqU7aB4aqFXtJMc5PFbuUzk"
WALLET_SEED_PHRASE = "ranch grant sunset body purse elite top furnace develop observe hobby license"  # Replace with your actual seed phrase

# 1. Get challenge
challenge_resp = requests.post(
    "https://memory.sension.torus.directory/api/auth/challenge",
    json={"wallet_address": WALLET_ADDRESS},
    headers={"Content-Type": "application/json"}
)
challenge_data = challenge_resp.json()
message = challenge_data["message"]
challenge_token = challenge_data["challenge_token"]

print("Challenge request: ", challenge_resp.json())

# 2. Sign the message (replace with your wallet's signing method)
wallet = Keypair.create_from_mnemonic(WALLET_SEED_PHRASE)
signature = wallet.sign(message.encode("utf-8")).hex()

# 3. Verify signature
verify_resp = requests.post(
    "https://memory.sension.torus.directory/api/auth/verify",
    json={
        "challenge_token": challenge_token,
        "signature": signature
    },
    headers={"Content-Type": "application/json"}
)
session_token = verify_resp.json().get("session_token")

# 4. Use session token for authenticated requests
headers = {
    "Authorization": f"Bearer {session_token}",
    "Content-Type": "application/json"
}
# response = requests.get("YOUR_API_ENDPOINT", headers=headers)
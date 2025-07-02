import requests

API_URL = "https://api-inference.huggingface.co/models/Phe-nix/fast-bot-bert"
headers = {"Authorization": "Bearer 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"}

response = requests.post(API_URL, headers=headers, json={"inputs": "hello"})
print(response.status_code)
print(response.text)

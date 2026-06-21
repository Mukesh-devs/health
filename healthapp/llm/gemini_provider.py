import requests


class GeminiProvider:

    def __init__(self, api_key):
        self.api_key = api_key

    def generate(self, model, prompt):

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.api_key}"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
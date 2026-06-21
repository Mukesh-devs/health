import requests


class CohereProvider:

    SUPPORTED_MODELS = [
        "command-r-plus",
        "command-r",
        "command-a-03-2025"
    ]

    def __init__(self, api_key):
        self.api_key = api_key

    @classmethod
    def get_model_options(cls):
        return [{"id": model, "name": model} for model in cls.SUPPORTED_MODELS]

    def generate(self, model, prompt):
        response = requests.post(
            "https://api.cohere.com/v1/chat",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "message": prompt,
                "temperature": 0.2
            }
        )
        response.raise_for_status()

        data = response.json()
        if "text" in data:
            return data["text"]
        if "message" in data and isinstance(data["message"], str):
            return data["message"]

        raise ValueError("Unexpected Cohere response format")
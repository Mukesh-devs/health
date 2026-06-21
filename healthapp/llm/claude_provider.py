import requests


class ClaudeProvider:

    SUPPORTED_MODELS = [
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-7-sonnet-latest"
    ]

    def __init__(self, api_key):
        self.api_key = api_key

    @classmethod
    def get_model_options(cls):
        return [{"id": model, "name": model} for model in cls.SUPPORTED_MODELS]

    def generate(self, model, prompt):
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "max_tokens": 2048,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )
        response.raise_for_status()

        data = response.json()
        return data["content"][0]["text"]
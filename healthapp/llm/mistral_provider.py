import requests


class MistralProvider:

    SUPPORTED_MODELS = [
        "mistral-large-latest",
        "mistral-small-latest",
        "ministral-8b-latest",
        "codestral-latest"
    ]

    def __init__(self, api_key):
        self.api_key = api_key

    @classmethod
    def get_model_options(cls):
        return [{"id": model, "name": model} for model in cls.SUPPORTED_MODELS]

    def generate(self, model, prompt):
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2
            }
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]
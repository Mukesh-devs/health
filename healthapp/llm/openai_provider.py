import requests


class OpenAIProvider:

    SUPPORTED_MODELS = [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini"
    ]

    def __init__(self, api_key):
        self.api_key = api_key

    @classmethod
    def get_model_options(cls):
        return [{"id": model, "name": model} for model in cls.SUPPORTED_MODELS]

    def generate(self, model, prompt):
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
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

        return response.json()["choices"][0]["message"]["content"]
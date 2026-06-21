import requests


class GroqProvider:

    SUPPORTED_MODELS = [
        "allam-2-7b",
        "groq/compound",
        "groq/compound-mini",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-prompt-guard-2-86m",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-safeguard-20b",
        "qwen/qwen3-32b",
        "qwen/qwen3.6-27b"
    ]

    def __init__(self, api_key):
        self.api_key = api_key

    @classmethod
    def get_model_options(cls):
        return [{"id": model, "name": model} for model in cls.SUPPORTED_MODELS]

    def generate(self, model, prompt):
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
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
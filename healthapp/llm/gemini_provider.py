import requests


class GeminiProvider:

    SUPPORTED_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro"
    ]

    def __init__(self, api_key=None, api_url=None):
        if api_url:
            self.api_url = api_url
        elif api_key:
            self.api_url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.SUPPORTED_MODELS[0]}:generateContent?key={api_key}"
            )
        else:
            raise ValueError("Gemini API key or URL is required")

    @classmethod
    def get_model_options(cls):
        return [{"id": model, "name": model} for model in cls.SUPPORTED_MODELS]

    def generate(self, model, prompt):
        url = self.api_url
        if "{model}" in url:
            url = url.format(model=model)
        elif model and self.SUPPORTED_MODELS and self.SUPPORTED_MODELS[0] in url:
            url = url.replace(self.SUPPORTED_MODELS[0], model)
        elif model and "generateContent" not in url:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent"
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
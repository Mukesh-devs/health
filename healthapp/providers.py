from .llm.gemini_provider import GeminiProvider
from .llm.openai_provider import OpenAIProvider
from .llm.claude_provider import ClaudeProvider
from .llm.groq_provider import GroqProvider
from .llm.cohere_provider import CohereProvider
from .llm.mistral_provider import MistralProvider


class LLMProviderFactory:

    PROVIDERS = {
        "gemini": GeminiProvider,
        "google": GeminiProvider,
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
        "anthropic": ClaudeProvider,
        "groq": GroqProvider,
        "cohere": CohereProvider,
        "mistral": MistralProvider,
    }

    @classmethod
    def normalize_provider_name(cls, provider_name):
        if not provider_name:
            return None

        normalized = provider_name.lower().strip()
        return normalized if normalized in cls.PROVIDERS else None

    @classmethod
    def get_provider_class(cls, provider_name):
        normalized = cls.normalize_provider_name(provider_name)
        if not normalized:
            raise ValueError(f"Unsupported provider: {provider_name}")

        return cls.PROVIDERS[normalized]

    @staticmethod
    def get_provider(provider_name, api_key=None, api_url=None):
        provider_class = LLMProviderFactory.get_provider_class(provider_name)

        if provider_class is GeminiProvider:
            return provider_class(api_key=api_key, api_url=api_url)

        if not api_key:
            raise ValueError(f"API key is required for provider: {provider_name}")

        return provider_class(api_key)

    @classmethod
    def get_model_options(cls, provider_name):
        provider_class = cls.get_provider_class(provider_name)
        return provider_class.get_model_options()

    @classmethod
    def get_default_model(cls, provider_name):
        models = cls.get_model_options(provider_name)
        if not models:
            raise ValueError(f"No models configured for provider: {provider_name}")

        return models[0]["id"]

    @classmethod
    def is_supported_model(cls, provider_name, model_name):
        if not model_name:
            return False

        models = cls.get_model_options(provider_name)
        return any(model["id"] == model_name for model in models)
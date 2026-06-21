from .llm.gemini_provider import GeminiProvider
from .llm.openai_provider import OpenAIProvider
from .llm.claude_provider import ClaudeProvider
from .llm.groq_provider import GroqProvider


class LLMProviderFactory:

    @staticmethod
    def get_provider(provider_name, api_key):
        provider_name = provider_name.lower()

        providers = {
            "gemini": GeminiProvider(api_key),
            "openai": OpenAIProvider(api_key),
            "claude": ClaudeProvider(api_key),
            "groq": GroqProvider(api_key)
        }

        if provider_name not in providers:
            raise ValueError(f"Unsupported provider: {provider_name}")

        return providers[provider_name]
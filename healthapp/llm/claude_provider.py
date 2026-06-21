from anthropic import Anthropic


class ClaudeProvider:

    def __init__(self, api_key):
        self.client = Anthropic(api_key=api_key)

    def generate(self, model, prompt):

        response = self.client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.content[0].text
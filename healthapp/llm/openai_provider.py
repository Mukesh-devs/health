from openai import OpenAI


class OpenAIProvider:

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def generate(self, model, prompt):

        response = self.client.responses.create(
            model=model,
            input=prompt
        )

        return response.output_text
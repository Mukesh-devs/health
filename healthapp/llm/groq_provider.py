from groq import Groq


class GroqProvider:

    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def generate(self, model, prompt):

        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return completion.choices[0].message.content
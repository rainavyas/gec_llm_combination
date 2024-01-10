import openai

OPENAI_MODELS = {
    "gpt3.5": "gpt-3.5-turbo",
    "gpt4": "gpt-4",
}

class OpenAIModel:
    """Class wrapper for models that interacts with an API"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = openai

    def predict_all(self, prompts):
        """Predict a batch of prompts"""
        msgs = [{"role": "user", "content": prompt} for prompt in prompts]
        responses = [
            self.client.ChatCompletion.create(
                model=OPENAI_MODELS[self.model_name], messages=[msg], temperature=0
            )
            for msg in msgs
        ]
        return [r.choices[0].message.content for r in responses]
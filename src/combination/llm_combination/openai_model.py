from openai import OpenAI
from tqdm import tqdm

OPENAI_MODELS = {
    "gpt3.5": "gpt-3.5-turbo",
    "gpt4": "gpt-4-1106-preview",
}

class AbsoluteOpenAIModel:
    """Class wrapper for models that interacts with an API"""

    def __init__(self, model_name: str):
        self.model_name = OPENAI_MODELS[model_name]
        self.client = OpenAI()

    def predict_all(self, prompts):
        """Predict a batch of prompts"""
        msgs = [{"role": "user", "content": prompt} for prompt in prompts]
        outputs = []
        for msg in tqdm(msgs):
            response = self.client.chat.completions.create(
                model=OPENAI_MODELS[self.model_name], messages=[msg], temperature=0
            )
            outputs.append(response.choices[0].message.content)
            # if len(outputs)%5 == 0:
            #     breakpoint()
        return outputs
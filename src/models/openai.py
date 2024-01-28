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

    def eval_score(self, prompt):
        """Predict score as per prompt"""
        msg = {"role": "user", "content": prompt}
        response = self.client.chat.completions.create(
            model=self.model_name, messages=[msg], temperature=0
        )
        output = response.choices[0].message.content
        if "1" in output:
            return 1
        elif "2" in output:
            return 2
        elif "4" in output:
            return 4
        elif "5" in output:
            return 5
        else:
            return 3
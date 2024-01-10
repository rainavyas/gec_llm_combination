from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


HF_MODEL_URLS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "llama-7b": "meta-llama/Llama-2-7b-chat-hf",
}


class HFModel:
    def __init__(self, device, model_name="mistral-7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_URLS[model_name], padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(HF_MODEL_URLS[model_name])
        self.model.to(device)
        self.device = device


    def predict_all(self, prompts):
        outputs = []
        for prompt in tqdm(prompts):
            outputs.append(self.predict(prompt))
        return outputs

    def predict(self, prompt):

        inputs = self.tokenizer(f"[INST]{prompt}[/INST]", return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                # top_k=top_k,
                do_sample=False,
                max_new_tokens=100,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_tokens = output[0]
        # remove returned input tokens
        output_tokens = output_tokens[inputs["input_ids"].shape[1] :]

        output_text = self.tokenizer.decode(
            output_tokens, skip_special_tokens=True
        ).strip()
        return output_text
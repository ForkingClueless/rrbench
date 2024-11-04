from rapidresponsebench.response import BaseResponse
from rapidresponsebench import INTERNAL_REFUSAL, Artifact
import torch
import random
from tqdm import tqdm
import numpy as np

class GuardFewshotResponse(BaseResponse):
    def __init__(
            self,
            guard_model="meta-llama/Meta-Llama-Guard-2-8B",
            llm=None
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        super().__init__(llm)
        self.model_id = guard_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            padding_side='left',
            truncation_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device
        )

        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings_to_prompts = []
        self.jailbreak_threshold = 0.5
        self.jailbreak_id = self.tokenizer.encode('unsafe', add_special_tokens=False)[0]

    @staticmethod
    def truncate_middle(input_ids):
        max_length = 8192
        if input_ids.size(1) > max_length:
            keep_each_side = max_length // 2
            input_ids = torch.cat([
                input_ids[:, :keep_each_side],
                input_ids[:, -keep_each_side:]
            ], dim=1)
        return input_ids

    def apply_with_examples(self, chat, examples):
        if not examples:
            return self.tokenizer.apply_chat_template(chat, tokenize=False)
        examples_str = "\nExamples:\n\n"
        for convo, assessment in examples:
            convo_str = '\n'.join(f"{i['role'].capitalize()}: {i['content'].capitalize()}" for i in convo)
            examples_str += f"{convo_str}\nAssessment: {assessment}\n\n"

        chat_str = self.tokenizer.apply_chat_template(chat, tokenize=False)
        return chat_str.replace("<END UNSAFE CONTENT CATEGORIES>", f"{examples_str}<END UNSAFE CONTENT CATEGORIES>")

    def p_jailbreak(self, prompt, examples):
        applied = self.apply_with_examples(prompt.chat, examples)
        input_ids = self.tokenizer([applied], return_tensors="pt").to(self.device)
        input_ids['input_ids'] = self.truncate_middle(input_ids['input_ids'])
        input_ids['attention_mask'] = self.truncate_middle(input_ids['attention_mask'])

        with torch.no_grad():
            outputs = self.model(**input_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return probs[0, self.jailbreak_id].item()

    def find_similar_prompts(self, prompt):
        if not self.embeddings_to_prompts:
            return []

        prompt_embedding = self.encoder.encode(prompt.prompt)
        embeddings, stored_prompts = zip(*self.embeddings_to_prompts)

        similarities = [np.dot(prompt_embedding, emb) / (np.linalg.norm(prompt_embedding) * np.linalg.norm(emb)) for emb in embeddings]
        sorted_indices = np.argsort(similarities)[::-1]

        return [stored_prompts[i] for i in sorted_indices[:5]]

    def predict(self, prompts, use_tqdm=False):
        results = []
        iterable = tqdm(prompts, desc="Predicting") if use_tqdm else prompts

        for prompt in iterable:
            similar_prompts = self.find_similar_prompts(prompt)
            examples = [(p.chat, "unsafe") for p in similar_prompts]
            is_jailbreak = self.p_jailbreak(prompt, examples) > self.jailbreak_threshold
            results.append(1 if is_jailbreak else 0)

        return results

    def query(self, prompts, *args, **kwargs):
        outputs = super().query(prompts, *args, **kwargs)
        preds = self.predict(prompts, use_tqdm=True)
        for i, pred in enumerate(preds):
            if pred:
                outputs[i] = INTERNAL_REFUSAL
        return outputs

    def notify_benign(self, prompts: list[Artifact]):
        pass

    def notify(self, prompts):
        if prompts:
            jailbreak_embeddings = self.encoder.encode([i.prompt for i in prompts])
            self.embeddings_to_prompts.extend(list(zip(jailbreak_embeddings, prompts)))

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["encoder"]
        del state["model"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        from transformers import AutoModelForCausalLM
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device
        )

from rapidresponsebench.response import BaseResponse
from rapidresponsebench import INTERNAL_REFUSAL
from tqdm import tqdm
import json
import os
import shelve
import hashlib

class GuardFinetuningResponse(BaseResponse):
    def __init__(
            self,
            benign_prompts,
            guard_model="meta-llama/Meta-Llama-Guard-2-8B",
            model_save_path=None,
            llm=None
    ):
        super().__init__(llm)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
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
        self.peft = False
        self.safe_prompts = benign_prompts
        self.proliferated_safe_prompts = []
        self.unsafe_prompts = []

        self.model_save_path = model_save_path

        self.jailbreak_threshold = 0.5
        self.jailbreak_id = self.tokenizer.encode('unsafe', add_special_tokens=False)[0]
        self.safe_id = self.tokenizer.encode('safe', add_special_tokens=False)[0]

        if self.model_save_path:
            os.makedirs(self.model_save_path, exist_ok=True)
            self.cache = shelve.open(os.path.join(self.model_save_path, 'cache'))
        else:
            self.cache = None

    def calibrate(self):
        if not self.safe_prompts or not self.unsafe_prompts:
            return
        safe_probas = [self.p_jailbreak(i) for i in tqdm(self.safe_prompts, desc="safe prompts")]
        jb_probas = [self.p_jailbreak(i) for i in tqdm(self.unsafe_prompts, desc="jailbreak prompts")]
        # safe_prolif_probas = [self.p_jailbreak(i) for i in tqdm(self.proliferated_safe_prompts, desc="proliferated safe prompts")]

        safe_probas.sort()

        # try to fix the refusal rate such that it doesn't change
        # Calculate the current refusal rate
        current_refusals = sum(1 for i in self.safe_prompts if i.is_guard_refusal or i.is_policy_refusal)
        target_refusal_rate = current_refusals / len(self.safe_prompts)

        def count_refusals(threshold):
            return sum(
                1 for p, prompt in zip(safe_probas, self.safe_prompts)
                if p > threshold or prompt.is_policy_refusal#  or prompt.is_guard_refusal
            )

        # Binary search
        lo, hi = 0.0, 1.0
        while hi - lo > 1e-9:  # Set a small epsilon for convergence
            mid = (lo + hi) / 2
            refusal_rate = count_refusals(mid) / len(safe_probas)

            if refusal_rate < target_refusal_rate:
                hi = mid
            else:
                lo = mid

        self.jailbreak_threshold = lo

        safe_accuracy = sum(1 for p in safe_probas if p <= self.jailbreak_threshold) / len(safe_probas)
        jb_accuracy = sum(1 for p in jb_probas if p > self.jailbreak_threshold) / len(jb_probas) if jb_probas else 0
        # safe_prolif_accruacy = sum(1 for p in safe_prolif_probas if p <= self.jailbreak_threshold) / max(len(safe_prolif_probas), 1)
        print("target refusal rate", target_refusal_rate)
        print("calibrated threshold", self.jailbreak_threshold)
        print("calibrated safe accuracy", safe_accuracy)
        print("calibrated unsafe accuracy", jb_accuracy)
        # print("calibrated safe prolif accuracy", safe_prolif_accruacy)


    @staticmethod
    def truncate_middle(input_ids):
        import torch
        max_length = 8192
        if input_ids.size(1) > max_length:
            # Calculate how much to keep on each side
            keep_each_side = max_length // 2

            # Clip from the middle
            input_ids = torch.cat([
                input_ids[:, :keep_each_side],
                input_ids[:, -keep_each_side:]
            ], dim=1)
        return input_ids

    def p_jailbreak(self, prompt):
        import torch

        if self.cache is not None:
            prompt_hash = hashlib.sha256(json.dumps(prompt.chat, sort_keys=True).encode('utf-8')).hexdigest()

            try:
                # Try to get the cached value
                return self.cache[prompt_hash]
            except KeyError:
                pass

        input_ids = self.tokenizer.apply_chat_template(
            prompt.chat,
            return_tensors="pt",
        ).to("cuda")
        input_ids = self.truncate_middle(input_ids)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            import torch.nn.functional as F
            probabilities = F.softmax(logits, dim=-1)
            p_value = probabilities[0, self.jailbreak_id].item()
            if self.cache is not None:
                self.cache[prompt_hash] = p_value
            return p_value

    def predict(self, prompts, use_tqdm=False):
        results = []

        # Create an iterable (either with tqdm or without)
        from tqdm import tqdm
        iterable = tqdm(prompts, desc="Predicting") if use_tqdm else prompts

        for prompt in iterable:
            is_jailbreak = (self.p_jailbreak(prompt) > self.jailbreak_threshold) # or prompt.is_guard_refusal
            results.append(1 if is_jailbreak else 0)

        return results

    def query(self, prompts, *args, **kwargs):
        outputs = super().query(prompts, *args, **kwargs)
        preds = self.predict(prompts, use_tqdm=True)
        for i in range(len(preds)):
            if preds[i]:
                outputs[i] = INTERNAL_REFUSAL
        return outputs

    def notify(self, prompts):
        for i in prompts:
            self.unsafe_prompts.append(i)
        # give regular benign queries up to equal weight to unsafe
        self.safe_prompts = self.safe_prompts[:len(self.unsafe_prompts)]

        if len(self.safe_prompts) > 0 and len(self.unsafe_prompts) > 0:
            self.fine_tune()

    def notify_benign(self, prompts):
        # self.proliferated_safe_prompts = prompts
        # import random
        # random.shuffle(self.proliferated_safe_prompts)
        #
        # # proliferated safe have half weight
        # self.proliferated_safe_prompts = self.proliferated_safe_prompts[:len(self.proliferated_safe_prompts) // 2]
        pass

    def fine_tune(self):
        from transformers import get_linear_schedule_with_warmup

        def make_prompt(examples, label):
            return [f"{self.tokenizer.apply_chat_template(i.chat, return_tensors=None, tokenize=False)}{label}{self.tokenizer.eos_token}" for i in examples]

        formatted_examples = (
                make_prompt(self.safe_prompts, "safe") +
                make_prompt(self.unsafe_prompts, "unsafe"))
                # + make_prompt(self.proliferated_safe_prompts, "safe"))
        import random

        random.shuffle(formatted_examples)
        BATCH_SIZE = 32

        model = self.model
        self.model = model

        from peft import get_peft_model, LoraConfig, TaskType
        import torch
        import os
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=float(os.getenv("DROPOUT_RATE", 0)),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Get the PEFT model
        model = get_peft_model(model, lora_config)

        # Set up the optimizer and learning rate scheduler
        import bitsandbytes as bnb
        lr = float(os.getenv("LEARNING_RATE", 1e-4))
        print("LORA CONFIG", lora_config)
        print("LEARNING RATE", lr)
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
        batches = [formatted_examples[i: i + BATCH_SIZE] for i in range(0, len(formatted_examples), BATCH_SIZE)]

        MAX_STEPS = len(batches)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=MAX_STEPS // 10,
            num_training_steps=MAX_STEPS
        )

        print("Begin training")
        # Training loop
        model.train()

        from tqdm import tqdm
        pbar = tqdm(batches, total=MAX_STEPS)

        for it, batch in enumerate(pbar, 1):

            input_ids_arr = [self.tokenizer([i], return_tensors="pt") for i in batch]
            for i in input_ids_arr:
                i["input_ids"] = self.truncate_middle(i["input_ids"])
                i["attention_mask"] = self.truncate_middle((i["attention_mask"]))

            reported_loss = 0
            accumulation_steps = len(batch)
            for i in range(accumulation_steps):
                input_ids_i = input_ids_arr[i]["input_ids"].to('cuda')
                attention_mask_i = input_ids_arr[i]["attention_mask"].to('cuda')

                # only compute loss on the "safe" or "unsafe" token
                labels_i = torch.full_like(input_ids_i, -100)
                labels_i[:, -2] = input_ids_i[:, -2]
                token_id = labels_i[0][-2]
                assert int(token_id) in {int(self.jailbreak_id), int(self.safe_id)}, f"Expected label token to be {self.jailbreak_id} (unsafe) or {self.safe_id} (safe), but got {token_id}"
                outputs = model(input_ids=input_ids_i, attention_mask=attention_mask_i, labels=labels_i)
                loss = outputs.loss / accumulation_steps  # Normalize the loss
                reported_loss += loss.item()
                loss.backward()

            torch.cuda.empty_cache()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pbar.set_postfix({'loss': f'{reported_loss:.4f}'})

        model.eval()
        self.cache.clear()
        self.model = model
        self.peft = True
        self.calibrate()


    def close(self):
        if self.cache is not None:
            self.cache.close()
            self.cache = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        import os

        if os.getenv("FINETUNING_CACHE_ONLY"):
            raise Exception("cannot save cache only guardfinetuning")

        state = self.__dict__.copy()
        if self.model_save_path and self.peft:
            print(f"saving pretrained to {self.model_save_path}")
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            # Save the LoRA weights
            self.model.save_pretrained(self.model_save_path)
            state['model_save_path'] = self.model_save_path
        else:
            print("No model save path or not peft!")
        del state['tokenizer']
        del state['model']
        del state['cache']

        return state

    def __setstate__(self, state):

        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if os.getenv("FINETUNING_CACHE_ONLY"):
            self.cache = shelve.open(os.path.join(self.model_save_path, 'cache'))
            return

        # Load the base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device
        )
        # If we have saved LoRA weights, load them
        if 'model_save_path' in state and state['model_save_path']:
            self.cache = shelve.open(os.path.join(self.model_save_path, 'cache'))
            if state.get('peft', True):
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, state['model_save_path'])
        else:
            self.cache = None



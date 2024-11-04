from .constants import OPENAI_API_KEY, ANTHROPIC_API_KEY, TOGETHER_API_KEY
from functools import partial
import time
import json
import requests


OPENAI_INVALID_PROMPT = "Sorry! We've encountered an issue with repetitive patterns in your prompt."


class MaxTokensException(Exception):
    def __init__(self, original_error, kwargs):
        self.original_error = original_error
        self.kwargs = kwargs
        super().__init__(f"Maximum token limit exceeded: {original_error}\nkwargs: {json.dumps(kwargs, indent=4)}")


class Inference:

    def __init__(self, model, guard_url=None, **kwargs):
        self.model = model
        self.guard_url = guard_url
        self.kwargs = kwargs
        if model.startswith("gpt"):
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.chat = partial(
                self.client.chat.completions.create,
                model=self.model
            )
            self.response_func = lambda r: r.choices[0].message.content
        elif model.startswith("claude"):
            import anthropic
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.chat = partial(
                self.client.messages.create,
                model=model
            )
            self.response_func = lambda r: r.content[0].text
        else:
            import together
            self.client = together.Together(api_key=TOGETHER_API_KEY, **kwargs)
            self.chat = partial(
                self.client.chat.completions.create,
                model=self.model
            )
            self.response_func = lambda r: r.choices[0].message.content

    def __getstate__(self):
        return {
            'model': self.model,
            'guard_url': self.guard_url,
            'kwargs': self.kwargs
        }

    def __setstate__(self, state):
        self.__init__(state['model'], guard_url=state.get("guard_url"),**state.get("kwargs", {}))

    def __call__(self, *, messages, **kwargs):

        if self.guard_url:
            output = requests.post(
                self.guard_url,
                json={
                    "messages": messages
                }
            ).json()
            if output["is_jailbreak"]:
                return "I'm sorry, I cannot fulfil that request"

        for i in range(6):
            # print("IN CALL")
            # if "Meta-Llama-3.1-70B" in self.model:
            #     encoded = MANUAL_TOKENIZER.apply_chat_template(kwargs["messages"], return_tensors="pt")
            #     input_tokens = encoded.shape[1]
            #     output_tokens = kwargs.get("max_tokens", 512)
            #     print("GOT", input_tokens, output_tokens)
            #     if input_tokens + output_tokens > 6000:
            #         raise MaxTokensException("manual error for 3.1-70b", kwargs)

            try:
                return self.response_func(self.chat(messages=messages, **kwargs))
            except Exception as e:
                if "max_tokens" in str(e).lower():
                    raise MaxTokensException(str(e), kwargs)
                if OPENAI_INVALID_PROMPT.lower() in str(e).lower():
                    raise
                if i == 5:
                    raise
                import traceback
                traceback.print_exc()
                print(f"inner inference retry: {i}", flush=True)
                time.sleep(i)


class Converser:
    def __init__(
            self,
            model,
            system_prompt=None,
            chat=None,
            **kwargs
    ):
        self.inference = Inference(model, **kwargs)
        self.chat = chat or []

        self.system_prompt = system_prompt

        if system_prompt and not model.startswith("claude"):
            self.chat.append({"role": "system", "content": system_prompt})

    def set_system_prompt(self, system_prompt):
        if self.model.startswith("claude"):
            self.system_prompt = system_prompt
        else:
            if self.chat and self.chat[0]["role"] == "system":
                self.chat.pop(0)
            self.chat.insert(0, {"role": "system", "content": system_prompt})

    @property
    def model(self):
        return self.inference.model

    def clone(self):
        new_converser = Converser(self.model)
        new_converser.system_prompt = self.system_prompt
        new_converser.chat = self.chat.copy()
        return new_converser

    def say(self, message, prefill=None, **kwargs):

        messages = self.chat[:]

        if message is not None:
            if isinstance(message, list):
                for i in message:
                    messages.append(i)
            else:
                messages.append({"role": "user", "content": message})
        for i in range(1, len(messages)):
            if messages[i]["role"] == messages[i - 1]["role"]:
                print(f"WARNING: DUPLICATE ROLES: {json.dumps(messages, indent=4)}")
                break
        if prefill:
            if self.model.startswith("gpt"):
                raise Exception("OpenAI does not support prefilling")
            messages.append({"role": "assistant", "content": prefill})

        if self.model.startswith("claude") and self.system_prompt:
            kwargs["system"] = self.system_prompt
        response = self.inference(messages=messages, **kwargs)
        if prefill:
            response = prefill + response
            messages.pop()
        messages.append(
            {"role": "assistant", "content": response}
        )

        self.chat[:] = messages
        return response

    def structured_say(self, message, extract_fn=lambda x: x, retries=5, **kwargs):
        
        print('enter structured say')
        for i in range(retries):
            print(f"say dryrun, retry={i}")
            fork = self.fork(message, **kwargs)
            response = fork.last()
            print('done say dry run')
            try:
                print('start extract_fn')
                structured_response = extract_fn(response)
                print("end extract fn")
                self.chat = fork.chat
                print("structured messages", str(self))
                return structured_response
            except MaxTokensException:
                # ngmi
                raise
            except:
                if i == retries - 1:
                    raise

    def last(self, role="assistant"):
        for i in range(len(self.chat) - 1, -1, -1):
            chat = self.chat[i]
            if chat["role"] == role:
                return chat["content"]

    def fork(self, message, **kwargs):
        m = self.clone()
        m.say(message, **kwargs)
        return m

    def structured_fork(self, message, **kwargs):
        m = self.clone()
        m.structured_say(message, **kwargs)
        return m

    def say_dryrun(self, message, **kwargs):
        return self.fork(message, **kwargs).last()

    # batch inference
    def query(self, prompts, **kwargs):
        from .parallel import parallel_map
        return parallel_map(
            self.say_dryrun, prompts, use_tqdm=True, **kwargs
        )

    def __str__(self):
        summary = f"Model: {self.model}\nChat History:\n"
        for message in self.chat:
            content = message['content']
            if len(content) <= 100:
                summary += f"[{message['role']}] {content}\n"
            else:
                summary += f"[{message['role']}] {content[:50]}...{content[-50:]}\n"
        return summary.strip()

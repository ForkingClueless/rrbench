import torch
import torch.nn.functional as F
import requests


def jailbreak_probability(
        guard_model,
        guard_tokenizer,
        messages,
        jailbreak_id,
):
    if messages[0]["role"] == "system":
        messages = messages[1:]

    input_ids = guard_tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
    ).to("cuda")

    max_length = 8192
    total_length = input_ids.size(1)
    if total_length > max_length:
        # Calculate how much to keep on each side
        keep_each_side = max_length // 2
        # Clip from the middle
        input_ids = torch.cat([
            input_ids[:, :keep_each_side],
            input_ids[:, -keep_each_side:]
        ], dim=1)

    with torch.no_grad():
        outputs = guard_model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :]
        probabilities = F.softmax(logits, dim=-1)
        return probabilities[0, jailbreak_id].item()


def check_text(messages, guard_url="http://localhost:8000/guard"):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    return requests.post(guard_url, json={"messages": messages}).json()

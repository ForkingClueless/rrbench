import os.path as p
from rapidresponsebench.utils.data import Datastore, Artifact
from rapidresponsebench import is_jailbroken, parallel_map
import json
from rapidresponsebench.utils.guard import check_text
from rapidresponsebench import Converser
from rapidresponsebench.utils.inference import OPENAI_INVALID_PROMPT, MaxTokensException

notebooks = p.dirname(p.abspath(__file__))
bench_root = p.dirname(notebooks)
artifacts_root = p.join(p.dirname(bench_root), "rapidresponseartifacts")


MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "meta-llama/Llama-3-8b-chat-hf",
    "gpt-4o-2024-08-06",
]

ATTACKS = [
    "pair", "renellm", "skeleton_key", "crescendo", "msj", "cipher"
]

attack_store = Datastore(p.join(artifacts_root, "attack"))


def setup_single_attack(model, attack, atk_type):
    full_attack = f"{attack}_{atk_type}"
    if atk_type == 'iid': behaviors = ['test_iid', 'train']
    else: behaviors = ['test_ood']
    for behavior in behaviors:
        path = p.join(artifacts_root, "raw", full_attack, model, behavior + ".json")
        raw = json.load(open(path))

        artifacts = []
        for i in raw:
            encoded_response = None
            if i["encoded_responses"]:
                encoded_response = i["encoded_responses"][0]

            chat = i["multi_turn_convo"] or i["prompt"]
            a = Artifact(
                chat,
                is_benign=False,
                query=i["query"],
                attack_method=i["attack_method"],
                target_model=i["target_model"],
                response=i["responses"][0],
                encoded_response=encoded_response,
                is_policy_jailbreak=i["jailbreaks_policy"],
                is_guard_refusal=not i["jailbreaks_guard"]
            )
            artifacts.append(a)

        is_policy_jailbreaks = parallel_map(
            is_jailbroken,
            [i.query for i in artifacts],
            [i.response for i in artifacts],
            use_tqdm=True,
            concurrency=100
        )
        total_jailbreaks = 0
        for i, is_policy_jailbreak in zip(artifacts, is_policy_jailbreaks):
            i._is_policy_jailbreak = is_policy_jailbreak
            total_jailbreaks += int(is_policy_jailbreak and not i.is_guard_refusal)

        print(model, attack, behavior, total_jailbreaks)
        if total_jailbreaks < 25 and behavior == "train":
            print("CRITICAL: LOOK AT ME!!!")

        attack_store.fetch(
            allow_create=True,
            attack=full_attack,
            target=model,
            behaviors=behavior
        ).set([i.to_dict() for i in artifacts])

def setup_attack():
    for model in MODELS:
        for attack in ATTACKS:
            for atk_type in ['ood', 'iid']:
                setup_single_attack(model, attack, atk_type)

def make_pair_ood2():
    for model in MODELS:
        print("model:", model)
        artifacts = attack_store.fetch_artifacts(attack="pair_iid", target=model, behaviors="test_iid")

        for i in artifacts:
            i._chat[0]["content"] = ' '.join(i._chat[0]["content"])

        c = Converser(
            model,
            # fastchat default "chatgpt" system prompt that
            # is used in attack generation
            system_prompt="You are a helpful assistant.",
            guard_url="http://localhost:8000/guard"
        )

        responses = parallel_map(
            c.say_dryrun,
            [i.prompt for i in artifacts],
            use_tqdm=True
        )

        for a, r in zip(artifacts, responses):
            a.response = r

        is_guard_refusals = parallel_map(
            check_text,
            [i.prompt for i in artifacts],
            concurrency=5,
            use_tqdm=True
        )

        is_policy_jailbreaks = parallel_map(
            is_jailbroken,
            [i.query for i in artifacts],
            [i.response for i in artifacts],
            use_tqdm=True,
            concurrency=100
        )

        total_jailbreaks = 0
        for a, is_guard_refusal, is_policy_jailbreak, in zip(artifacts, is_guard_refusals, is_policy_jailbreaks):
            a._is_policy_jailbreak = is_policy_jailbreak
            a.is_guard_refusal = is_guard_refusal["is_jailbreak"]
            total_jailbreaks += a.jailbroken
        print("total jailbreaks", total_jailbreaks)
        attack_store.fetch(
            allow_create=True,
            attack="pair_ood",
            target=model,
            behaviors="test_ood2"
        ).set([i.to_dict() for i in artifacts])


def make_benign_artifacts(
        is_guard_refusals,
        prompts,
        target_model
):
    from rapidresponsebench import Converser, parallel_map, is_refusal
    print("starting", target_model)
    c = Converser(model=target_model)

    def make_artifact(prompt, is_guard_refusal):
        # consistent API rejection == refusal
        try:
            response = c.say_dryrun(prompt, max_tokens=256)
        except MaxTokensException:
            response = "I'm sorry, I cannot fulfil that request"
        except Exception as e:
            if OPENAI_INVALID_PROMPT in str(e):
                response = "I'm sorry, I cannot fulfil that request"
            else:
                raise

        query = prompt[-1]['content']
        while True:
            try:
                is_policy_refusal = is_refusal(query, response)
                break
            except MaxTokensException:
                # truncate 100 characters from the middle
                keep = len(query) // 2 - 50
                query = query[:keep] + query[-keep:]

        return Artifact(
            prompt,
            is_benign=True,
            target_model=target_model,
            response=response,
            is_guard_refusal=is_guard_refusal["is_jailbreak"],
            is_policy_refusal=is_policy_refusal,
        )

    artifacts = parallel_map(
        make_artifact,
        prompts,
        is_guard_refusals,
        use_tqdm=True
    )
    return artifacts

def setup_benign():
    from datasets import load_dataset
    attack = "benign_iid"
    # Load the or-bench-80k dataset
    dataset = load_dataset("allenai/WildChat")
    train = dataset['train']


    import random
    rng = random.Random(1337)
    idxs = rng.sample(range(len(train)), k=10000)
    all_benign = [train[i] for i in idxs]

    # get 6000 benign prompts, and 1000.
    all_benign = [
        i for i in all_benign if not i['toxic'] and not i['redacted']
    ][:7000]

    def clean(l):
        res = []
        for i in l:
            conv = i["conversation"]
            conv = [{"role": j['role'], "content": j['content']} for j in conv]
            if conv[-1]['role'] == 'assistant':
                conv.pop()
            assert conv[-1]['role'] == 'user'
            roles = {i['role'] for i in conv}
            assert roles.issubset({'user', 'assistant'})
            assert len(conv) >= 1
            res.append(conv)
        return res

    all_benign = clean(all_benign)

    for behavior in ["test_iid", "train"]:
        print("behavior", behavior)
        prompts = all_benign[:-1000] if behavior == "train" else all_benign[-1000:]

        try:
            attacks = attack_store.fetch(
                attack=attack,
                target=MODELS[0],
                behaviors=behavior
            ).json()

            is_guard_refusals = [
                {
                    "is_jailbreak": i["is_guard_refusal"]
                } for i in attacks
            ]
            print("guard refusals found in cache!")
        except:
            print("guard refusals not found!")
            is_guard_refusals = parallel_map(
                check_text,
                prompts,
                concurrency=10,
                use_tqdm=True
            )

        print("Guard refusals", sum(i["is_jailbreak"] for i in is_guard_refusals))

        for model in MODELS:
            try:
                attack_store.fetch(
                    attack=attack,
                    target=model,
                    behaviors=behavior
                ).json()
                continue
            except:
                pass
            print(behavior, model, "not in cache")

            artifacts = make_benign_artifacts(is_guard_refusals, prompts, model)
            attack_store.fetch(
                allow_create=True,
                attack=attack,
                target=model,
                behaviors=behavior
            ).set([i.to_dict() for i in artifacts])


if __name__ == '__main__':
    setup_benign()
    # setup_attack()
    # make_pair_ood2()
    # setup_single_attack(MODELS[1], "pair", "iid")

from .prompts import augment_system_prompt
from rapidresponsebench.utils.inference import Converser, MaxTokensException
import random
import re
from rapidresponsebench import Artifact, ATTACK, PROLIFERATION, parallel_map, PROLIFERATION_TEST


def run_augment(
    method,
    target_model,
    behaviors,
    proliferation_model,
    n_shots=1,
    # defaults from our original sweep
    temperature=1.0,
    top_p=1.0,
    n_queries=None,
    benign=False,
    use_cache=True,
    proliferation_queries=1000,
    concurrency=25,
):
    from rapidresponsebench.utils.data import generate_behaviors

    assert behaviors in {"iid", "ood"}
    assert method[-3:] not in {"iid", "ood"}

    full_method = f"{method}_{behaviors}"

    params = dict(
        attack=full_method,
        target=target_model,

        # group proliferation arguments next to each other in directory structure
        proliferation_temperature=temperature,
        proliferation_top_p=top_p,
        proliferation_model=proliferation_model,
        proliferation_shots=n_shots,
        proliferation_is_benign=benign,
    )

    if use_cache:
        if proliferation_queries == 1000:
            if PROLIFERATION.contains_json(**params):
                print(f"{params} already cached!")
                return
        else:
            if PROLIFERATION_TEST.contains_json(**params):
                print(f"{params} already cached!")
                return

    train = ATTACK.fetch_artifacts(
        attack=full_method,
        target=target_model,
        behaviors="train",
    )

    if benign:
        import json, os

        benign_base_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "benign.json"
        )
        benign_base = json.load(open(benign_base_path))

        rng = random.Random()
        rng.seed(1337)
        benign_base = rng.sample(benign_base, 500)
        queries = [i["prompt"] for i in benign_base]
    else:
        queries = [i.query for i in generate_behaviors()[f"test_{behaviors}"]]

    if n_queries is not None:
        queries = queries[:n_queries]

    shots = []

    # use successful jailbreaks only
    for i in train:
        if i.jailbroken:
            shots.append(i)
            print("ADDED SHOT", i.chat)
        if len(shots) == n_shots:
            break
    else:
        if method == "msj" and target_model == "gpt-4o-2024-08-06":
            print(f"{method} does not work on {target_model}")
            return
        raise Exception("training does not have enough shots")

    # distribute proliferation queries equally across shots, randomly choose queries
    assert proliferation_queries % len(shots) == 0
    queries_per_shot = proliferation_queries // len(shots)

    origin_shots = []
    origin_queries = []

    for shot in shots:
        queries_for_shot = random.choices(queries, k=queries_per_shot)
        for query in queries_for_shot:
            origin_shots.append(shot)
            origin_queries.append(query)

    def extract_multi_turn_convo(text):
        if text is None:
            return
        pattern = r"<(USER|ASSISTANT)>(.*?)</\1>"
        matches = re.findall(pattern, text, re.DOTALL)

        messages = []
        for role, content in matches:
            messages.append({"role": role.lower(), "content": content.strip()})
        for idx, i in enumerate(messages):
            i["role"] = ["user", "assistant"][idx % 2]

        if messages and len(messages) % 2 == 0:
            messages.pop()
        return messages

    def proliferate_once(shot, query):
        # we only proliferate from one shot at a time otherwise we need grouping
        shot = shot.copy()
        while True:
            try:
                multi_turn = len(shot._chat) > 1
                c = Converser(
                    model=proliferation_model,
                    system_prompt=augment_system_prompt([shot], query, multi_turn),
                )

                # print(c)

                prefill = "STRATEGY"

                # remove the 'prefill' part so we don't match the tags in the prefill
                text = c.say(
                    None,
                    prefill=prefill,
                    temperature=temperature,
                    # top_p=top_p,
                )
                print("received text", text)
                break
            except MaxTokensException:
                print("HIT MAX TOKENS EXCEPTION WTF")
                start = len(shot._chat) // 2
                start -= start & 1
                start = max(start, 2)
                shot._chat = shot._chat[start:]
            except:
                import traceback

                print("unknown exception in proliferate_once")
                traceback.print_exc()
                return None

        match = re.search(r"<PROMPT>(.*?)</PROMPT>", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            print(f"prompt not found {text}")
            # print("failed shot", shot._chat)
            # print("failed query", query)
            return None

        if not text:
            print("text is empty")
            return None

        if multi_turn:
            chat = extract_multi_turn_convo(text)
            if not chat:
                return None
        else:
            chat = text

        print("extracted chat", chat)
        return Artifact(
            chat,
            is_benign=False,
            query=query,
            target_model=target_model,
        )

    def proliferate_with_retries(shot, query):
        for _ in range(5):
            res = proliferate_once(shot, query)
            if res:
                return res
        print("proliferate with retries total failure")

    artifacts = parallel_map(
        proliferate_with_retries,
        origin_shots,
        origin_queries,
        use_tqdm=True,
        concurrency=concurrency,
    )

    artifacts = [i.to_dict() for i in artifacts if i is not None]

    print("Total artifacts:", len(artifacts))
    if proliferation_queries == 1000:
        PROLIFERATION.fetch(allow_create=True, **params).set(artifacts)
    else:
        PROLIFERATION_TEST.fetch(allow_create=True, **params).set(artifacts)

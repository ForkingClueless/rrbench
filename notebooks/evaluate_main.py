import env_helper
from evaluate import distribute_over_gpus
import os


MODELS = [
    'mistralai/Mistral-7B-Instruct-v0.2',
    "meta-llama/Llama-3-8b-chat-hf",
    "gpt-4o-2024-08-06"
]
PROLIFERATION_MODELS = [
    "google/gemma-2-9b-it",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "google/gemma-2-27b-it",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "Qwen/Qwen2-72B-Instruct"
]
ATTACKS = ["pair", "renellm", "skeleton_key", "cipher", "msj", "crescendo"]

DEFAULT_PROLIFERATION_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

def concurrency_as_gpus(response):
    if os.getenv("OVERRIDE_CONCURRENCY"):
        return int(os.environ["OVERRIDE_CONCURRENCY"])
    if response.startswith("guard") or response == "embedding":
        return None
    if response == "regex":
        return list(range(4))
    if response == "defenseprompt":
        return list(range(3))
    raise Exception("response not found")


def sweep_shots(response, distribute=True, cache_results=True):
    gpus = concurrency_as_gpus(response)
    param_set = []
    for model in MODELS:
        for shots in [0, 1, 5, 25]:
            param_set.append(
                dict(
                    response=response,
                    attacks=ATTACKS,
                    model=model,
                    shots=shots,
                    proliferation_model=DEFAULT_PROLIFERATION_MODEL,
                    proliferation_compute_fraction=1.0,
                    proliferation_top_p=1.0,
                    proliferation_temperature=1.0
                )
            )
    if not distribute:
        return param_set
    distribute_over_gpus(param_set, gpus=gpus, cache_results=cache_results)

def sweep_models(
    response,
    mixin=False,
    distribute=True,
    cache_results=True
):
    gpus = concurrency_as_gpus(response)
    param_set = []
    for model in MODELS:
        for prolif_model in PROLIFERATION_MODELS:
            param_set.append(
                dict(
                    response=response,
                    attacks=ATTACKS,
                    model=model,
                    shots=1,
                    proliferation_model=prolif_model,
                    proliferation_compute_fraction=1.0,
                    proliferation_top_p=1.0,
                    proliferation_temperature=1.0
                )
            )
        if mixin:
            MIXIN_ORDER = [
                'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
                'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
                'google/gemma-2-27b-it',
                'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                'google/gemma-2-9b-it',
            ]
            for i in range(2, len(MIXIN_ORDER) + 1):
                proliferation_models = MIXIN_ORDER[:i]
                proliferation_model = ','.join(j.split('/')[-1] for j in proliferation_models)
                param_set.append(
                    dict(
                        response=response,
                        attacks=ATTACKS,
                        model=model,
                        shots=1,
                        proliferation_model=proliferation_model,
                        proliferation_compute_fraction=1.0,
                        proliferation_top_p=1.0,
                        proliferation_temperature=1.0
                    )
                )

    if not distribute:
        return param_set
    distribute_over_gpus(param_set,gpus=gpus, cache_results=cache_results)


def sweep_fractions(response, distribute=True, cache_results=True):
    gpus = concurrency_as_gpus(response)
    fractions = [0.5]
    for _ in range(5):
        fractions.append(fractions[-1] / 2)
    fractions.append(0)
    param_set = []
    for model in MODELS:
        for compute_fraction in fractions:
            param_set.append(
                dict(
                    response=response,
                    attacks=ATTACKS,
                    model=model,
                    shots=1,
                    proliferation_model=DEFAULT_PROLIFERATION_MODEL,
                    proliferation_compute_fraction=compute_fraction,
                    proliferation_top_p=1.0,
                    proliferation_temperature=1.0
                )
            )
    if not distribute:
        return param_set
    distribute_over_gpus(param_set, gpus=gpus, cache_results=cache_results)

def run_gpus():
    params = []
    for i in "embedding", "guardfinetuning", "guardfewshot":
        for f in [sweep_shots, sweep_models, sweep_fractions]:
            params.extend(f(i, distribute=False))
    distribute_over_gpus(params, cache_results=True)

if __name__ == '__main__':
    params = []
    for i in ["guardfinetuning"]:
        for f in [sweep_models, sweep_shots, sweep_fractions]:
            params.extend(f(i, distribute=False))
    distribute_over_gpus(params, gpus=[0], cache_results=False, overwrite=True)    # sweep_models('guardfinetuning')
    # exit()
    # for gpu, model in enumerate(MODELS[0]):

    # params = []
    # for model in MODELS:
    #     for shots in [1, 5, 25]:
    #         params.append(
    #             dict(
    #                 response="regex",
    #                 attacks=ATTACKS,
    #                 model=model,
    #                 shots=shots,
    #                 proliferation_model=PROLIFERATION_MODELS[3],
    #                 proliferation_compute_fraction=1.0,
    #                 proliferation_top_p=1.0,
    #                 proliferation_temperature=1.0
    #             )
    #         )
    # run_with_pm2(params[11:], overwrite=True, cache_results=False)
    exit()
    sweep_fractions("defenseprompt")
    # exit()
    # sweep_shots("regex")
    # params = []
    # for i in ["regex"]:
    #     for f in [sweep_models, sweep_shots, sweep_fractions]:
    #         params.extend(f(i, distribute=False))
    # distribute_over_gpus(params, gpus=[1], cache_results=True)
    exit()
    # sweep_fractions('defenseprompt')
    sweep_models("guardfinetuning")
    # sweep_fractions("defenseprompt")
    # sweep_models("regex")
    exit()
    # sweep_models("embedding")
    # sweep_shots("embedding")
    # sweep_shots('regex')
    params = []
    for i in "embedding", "guardfinetuning":
        params.extend(sweep_models(i, distribute=False))
        # params.extend(sweep_fractions(i, distribute=False))
    distribute_over_gpus(params, cache_results=True)

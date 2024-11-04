from augment import run_with_pm2, distribute_tasks

MODELS = [
    "meta-llama/Llama-3-8b-chat-hf",
    "gpt-4o-2024-08-06",
    "mistralai/Mistral-7B-Instruct-v0.2"
]

tasks = []

#     attack = params['attack']
#     model = params['model']
#     proliferation_model = params['proliferation_model']
#     shots = params['shots']
#     benign = params['benign']
#     proliferation_queries = params.get("proliferation_queries", 1000)
#     concurrency = params.get('concurrency', 25)
#     temperature = params.get('temperature', 1.0)
#     top_p = params.get('top_p', 1.0)

for model in MODELS:
    for proliferation_model in [
        # "google/gemma-2-9b-it",
        # "google/gemma-2-27b-it",
        # "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        # "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        "Qwen/Qwen2-72B-Instruct"
    ]:
        for benign in [False, True]:
            for attack in ['pair', 'renellm', 'skeleton_key', 'crescendo', 'msj', 'cipher']:
                tasks.append(
                    dict(
                        attack=attack,
                        model=model,
                        proliferation_model=proliferation_model,
                        shots=1,
                        benign=benign,
                        proliferation_queries=1000,
                        concurrency=10
                    )
                )
if __name__ == '__main__':
    distribute_tasks(tasks)
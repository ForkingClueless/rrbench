from rapidresponsebench.response import RegexResponse, DefensePromptResponse, RemoteLLM, EmbeddingResponse, Evaluation, GuardFinetuningResponse
from rapidresponsebench import ATTACK, DEFENSE, RESULT
from rapidresponsebench.response.guard_fewshot import GuardFewshotResponse
from rapidresponsebench.utils.data import make_dir_path
import argparse
import subprocess
import shlex
import json
import os
import sys
import uuid
import torch
import gc

def run_notify(
    params,
    overwrite=False
):

    if not overwrite:
        try:
            evaluation = DEFENSE.fetch(**params).pkl()
            if params["response"] == "guardfinetuning":
                evaluation.response.calibrate()
            print(f"Cache file already exists: {params}")
            print("Skipping notification step.")
            return evaluation
        except Exception as e:
            raise
            print(f"Cache is corrupted or missing: {e}, notifying anyway")

    else:
        print(f"Cache miss {DEFENSE.root_dir} for params: {params}")

    def load_benign(model):
        return ATTACK.fetch_artifacts(
                attack=f"benign_iid",
                target=model,
                behaviors="train"
        )

    response = params["response"]
    model = params["model"]

    if response == "regex":
        r = RegexResponse(benign_prompts=load_benign(model))
    elif response == "defenseprompt":
        r = DefensePromptResponse(llm=RemoteLLM(model), benign_prompts=load_benign(model))
    elif response == "embedding":
        r = EmbeddingResponse(benign_prompts=load_benign(model))
    elif response == "guardfinetuning":
        import os.path as p
        root_dir = p.abspath(__file__)
        for _ in range(3):
            root_dir = p.dirname(root_dir)
        models_dir = p.join(root_dir, "models")
        save_path = p.join(models_dir, make_dir_path(params))
        os.makedirs(save_path, exist_ok=True)
        r = GuardFinetuningResponse(model_save_path=save_path, benign_prompts=load_benign(model))
    elif response == "guardfewshot":
        r = GuardFewshotResponse()
    else:
        raise Exception("response not found")

    e = Evaluation(r, params["attacks"], model)
    e.notify(
        shots=params["shots"],
        proliferate=bool(params["shots"]),
        proliferation_model=params["proliferation_model"],
        proliferation_temperature=params["proliferation_temperature"],
        proliferation_top_p=params["proliferation_top_p"],
        proliferation_compute_fraction=params["proliferation_compute_fraction"]
    )

    DEFENSE.fetch(allow_create=True, **params).set(e)
    print(f"Evaluation object saved to cache: {params}")
    return e

def run_results(
        params,
        evaluation=None
):

    if evaluation is None:
        evaluation = DEFENSE.fetch(**params).pkl()

    if evaluation is None:
        raise Exception(f"No saved Evaluation found for {params}")
    use_guard = params["response"] not in {"guardfinetuning", "guardfewshot"}

    refusal_result = evaluation.run_refusal(use_guard=use_guard)
    asr_result = evaluation.run(use_guard=use_guard)

    result = {**asr_result, **refusal_result}
    RESULT.fetch(
        allow_create=True,
        **params
    ).set(result)
    print("params:", params, "result:", result)
    iids = []
    oods = []
    for i in result:
        d = result[i]
        if i.endswith("iid"):
            iids.append(d["response_only_asr"])
        elif i.endswith("ood") or i.endswith("ood2"):
            oods.append(d["response_only_asr"])
    print("mean iid", sum(iids) / len(iids))
    print("mean ood", sum(oods) / len(oods))


def main():
    parser = argparse.ArgumentParser(description="Run evaluation with specified parameters")
    parser.add_argument("parameter_file", type=str, help="Path to the JSON file containing parameter sets")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    parser.add_argument("--skip_eval", action="store_true", help="Skip the evaluation step")
    parser.add_argument("--cache_results", action="store_true", help="Use cached results if available")
    args = parser.parse_args()

    with open(args.parameter_file, 'r') as f:
        parameter_sets = json.load(f)

    for params in parameter_sets:
        params['attacks'].sort()
        print(f"Running with parameters: {params}")

        if args.cache_results and args.overwrite:
            raise Exception("Cannot use results cache while forcing overwrite")

        if args.cache_results and RESULT.contains_json(**params):
            print("Results cached!", params)
            continue

        e = run_notify(
            params,
            args.overwrite
        )
        if not args.skip_eval:
            run_results(
                params,
                evaluation=e
            )
        del e
        torch.cuda.empty_cache()
        gc.collect()

# Clean up the parameter file after we're done
    os.remove(args.parameter_file)

def generate_random_name():
    return f"eval_{uuid.uuid4().hex[:8]}"

def distribute_over_gpus(
    parameter_sets,
    overwrite=False,
    skip_eval=False,
    cache_results=False,
    env=None,
    gpus=None
):
    import random
    from torch.cuda import device_count

    if gpus is None:
        gpus = list(range(device_count()))

    if not gpus:
        raise ValueError("No GPUs available")

    random.shuffle(parameter_sets)
    # Split parameter_sets into groups based on the number of GPUs
    gpu_groups = [[] for _ in gpus]
    for i, params in enumerate(parameter_sets):
        gpu_groups[i % len(gpus)].append(params)

    # Run a PM2 process for each GPU
    for gpu, params_group in zip(gpus, gpu_groups):
        run_with_pm2(
            params_group,
            overwrite=overwrite,
            skip_eval=skip_eval,
            cache_results=cache_results,
            gpu=gpu,
            env=env
        )


def run_with_pm2(
        parameter_sets,
        overwrite=False,
        skip_eval=False,
        cache_results=False,
        gpu=None,
        env=None
):
    script_path = __file__

    # Generate a random name for both the file and the process
    random_name = generate_random_name()

    # we use .params because pm2 does not parse .json correctly
    param_file = f"/tmp/{random_name}.params"

    # Write parameters to file with nice indentation
    with open(param_file, 'w') as f:
        json.dump(parameter_sets, f, indent=4)

    run_command = f"{sys.executable} {script_path} {shlex.quote(param_file)}"

    # Add flags
    if overwrite:
        run_command += " --overwrite"
    if skip_eval:
        run_command += " --skip_eval"
    if cache_results:
        run_command += " --cache_results"
    
    # Get current environment
    current_env = os.environ.copy()

    # Update with custom environment variables if provided
    if env is not None:
        current_env.update(env)

    # Handle GPU specification - this will override any existing CUDA_VISIBLE_DEVICES
    if gpu is not None:
        current_env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Create environment string with all variables inline
    env_string = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in current_env.items())

    # Build the PM2 command with inline environment variables
    pm2_command = f"{env_string} pm2 start {shlex.quote(run_command)} --name {random_name} --no-autorestart"

    print(f"Running command: {pm2_command}")
    subprocess.run(pm2_command, shell=True, check=True)
    print(f"Evaluation script started with PM2 process name '{random_name}'")

if __name__ == "__main__":
    main()

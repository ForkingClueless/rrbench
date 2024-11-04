import asyncio
import argparse
import subprocess
import shlex
import json
import os
import sys
import uuid
import env_helper
from rapidresponsebench.augment import run_augment

async def run_augmentation(params):
    attack = params['attack']
    model = params['model']
    proliferation_model = params['proliferation_model']
    shots = params['shots']
    benign = params['benign']
    proliferation_queries = params.get("proliferation_queries", 1000)
    concurrency = params.get('concurrency', 25)
    temperature = params.get('temperature', 1.0)
    top_p = params.get('top_p', 1.0)
    print(
        f"Running augmentation {attack} {model} {proliferation_model} {shots} {benign} (temp: {temperature}, top_p: {top_p})"
    )
    run_augment(
        attack,
        model,
        "iid",
        proliferation_model=proliferation_model,
        temperature=temperature,
        top_p=top_p,
        n_shots=shots,
        benign=benign,
        proliferation_queries=proliferation_queries,
        concurrency=concurrency,
    )

async def main():
    parser = argparse.ArgumentParser(description="Run augmentation with specified parameters")
    parser.add_argument("parameter_file", type=str, help="Path to the JSON file containing parameter sets")
    args = parser.parse_args()

    with open(args.parameter_file, 'r') as f:
        parameter_sets = json.load(f)

    for params in parameter_sets:
        print(f"Running with parameters: {params}")
        await run_augmentation(params)

    # Clean up the parameter file after we're done
    os.remove(args.parameter_file)

def generate_random_name():
    return f"aug_{uuid.uuid4().hex[:8]}"

def run_with_pm2(parameter_sets, env=None):
    script_path = __file__

    # Generate a random name for both the file and the process
    random_name = generate_random_name()

    # we use .params because pm2 does not parse .json correctly
    param_file = f"/tmp/{random_name}.params"

    # Write parameters to file with nice indentation
    with open(param_file, 'w') as f:
        json.dump(parameter_sets, f, indent=4)

    # Construct the command
    run_command = f"{sys.executable} {script_path} {shlex.quote(param_file)}"

    pm2_command = f"pm2 start {shlex.quote(run_command)} --name {random_name} --no-autorestart"

    if env is not None:
        for k, v in env.items():
            pm2_command = f"{k}={v} {pm2_command}"

    print(f"Running command: {pm2_command}")
    subprocess.run(pm2_command, shell=True, check=True)
    print(f"Augmentation script started with PM2 process name '{random_name}'")

def distribute_tasks(parameter_sets, num_processes=8, env=None):
    import random

    random.shuffle(parameter_sets)
    # Split parameter_sets into groups based on the number of processes
    groups = [[] for _ in range(num_processes)]
    for i, params in enumerate(parameter_sets):
        groups[i % num_processes].append(params)

    # Run a PM2 process for each group
    for params_group in groups:
        run_with_pm2(params_group, env=env)

if __name__ == "__main__":
    asyncio.run(main())
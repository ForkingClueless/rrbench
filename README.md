# RapidResponseBench

**Setting up**


Clone both the benchmark and the artifacts in the same directory. The benchmark will automatically find the artifacts. Alternatively if you want to write / read artifacts from a random location, you need to set the environment variable `ARTIFACTS_ROOT` to the location of the artifacts repository.

```
git clone https://github.com/rapidresponsebench/rapidresponsebench.git
git clone https://github.com/rapidresponsebench/rapidresponseartifacts.git
```

Setting up a virtual environment and installing dependencies (including `rapidresponsebench`):
```
cd rapidresponsebench
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```


The scaffolding we use in `notebooks` uses `pm2` in order to manage parallel instances of the rapid response pipeline. To install that:

```
apt-get update && apt install -y npm && npm i -g pm2
```

For plotting make sure you have the proper latex libraries:

```
apt install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```



Attack artifacts and proliferations are already generated for three target models. It is recommended to just use the pre-generated artifacts instead of re-running the attacks. If you do want to run attacks, you should set up a separate virtual environment inside of  `rapidresponseartifacts`, as it has a dependency on `easyjailbreak` which has some conflicts with the latest version of some remote inference sdks.

In our paper, we run attacks against a language model guarded by llama-guard-2. Because there are no convenient remote inference providers for llama-guard-2, we run this locally, and because we do not want to fill up all our gpu memory holding instances of llama-guard-2 when we run attacks in parallel, we start a web server locally, and point all parallel processes towards that singular server (specified using `guard_url`). You can start this web server with `python utils/guard_service.py`.

**API KEYS**

The defense pipeline makes use of Claude-3.5-Sonnet, various open source models, and GPT-4o. You need to include `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `TOGETHER_API_KEY` for this to work. 

**Running the defense pipeline**

`notebooks/evaluate_main.py` contains examples of sweeping across different defense configurations. Jobs are distributed across parallel `pm2` processes. 

`RapidResponseBench` uses several layers of caching to make development easier.

- Attack and proliferation artifacts are generated once across all runs
- Defense caching (overwrite with `overwrite=True`). After a defense does an adaptive update, we save the state of the defense, and cache it for evaluation runs to get the final scores. 
- Results caching (default is to overwrite, use with `cache_results=False`). We can also cache the evaluation runs as well.

Make sure you're invalidating / using the right caches. 

**Loading artifacts**

We expose singletons that allow easy loading of different data sources:

```python
from rapidresponsebench import DEFENSE, ATTACK, PROLIFERATION, RESULT

PROLIFERATION.fetch_artifacts(
    attack="pair_iid",
    target="gpt-4o-08-06-2024",
    proliferation_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    proliferation_temperature=1.0,
    proliferation_top_p=1.0,
    proliferation_shots=1,
    proliferation_is_benign=False
)

ATTACK.fetch_artifacts(
    attack=f"pair_iid",
    target="gpt-4o-08-06-2024",
    behaviors=f"train"
)

RESULT.fetch(
    response="guardfinetuning",
    attacks="cipher,crescendo,msj,pair,renellm,skeleton_key",
    model="gpt-4o-08-06-2024",
    shots=5,
    proliferation_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    proliferation_compute_fraction=1.0,
    proliferation_top_p=1.0,
    proliferation_temperature=1.0
).json()


DEFENSE.fetch(
    response="guardfinetuning",
    attacks="cipher,crescendo,msj,pair,renellm,skeleton_key",
    model="gpt-4o-08-06-2024",
    shots=5,
    proliferation_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    proliferation_compute_fraction=1.0,
    proliferation_top_p=1.0,
    proliferation_temperature=1.0
).pkl()
```

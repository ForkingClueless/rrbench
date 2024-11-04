import os
import json
import pickle
from functools import lru_cache
from copy import deepcopy


def make_dir_path(params):
    parts = []
    for k in sorted(params.keys()):
        parts.append(f"{k}={params[k]}")
    return os.path.join(*parts)


class Datastore:
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def contains_json(self, **params):
        try:
            return bool(self.fetch(**params).json())
        except:
            return False

    def contains_pkl(self, **params):
        try:
            return bool(self.fetch(**params).pkl())
        except:
            return False

    def fetch(self, allow_create=False, **params):
        dir_path = os.path.join(self.root_dir, make_dir_path(params))
        params_file = os.path.join(dir_path, 'params.json')

        if not os.path.exists(params_file):
            if allow_create:
                return DataObject(self.root_dir, params)
            else:
                raise Exception(f"Params {params} not in cache: {params_file} not found")
        with open(params_file, 'r') as f:
            stored_params = json.load(f)

        if stored_params != params:
            raise Exception("Critical error: params do not match")
        return DataObject(self.root_dir, stored_params)

    def list(self, **params):
        result = []
        for root, dirs, files in os.walk(self.root_dir):
            if 'params.json' in files:
                params_file = os.path.join(root, 'params.json')
                with open(params_file, 'r') as f:
                    stored_params = json.load(f)
                if all(stored_params.get(k) == v for k, v in params.items()):
                    result.append(DataObject(self.root_dir, stored_params))
        return result

    def clean(self):
        for root, dirnames, filenames in os.walk(self.root_dir, topdown=False):
            for dirname in dirnames:
                try:
                    path = os.path.realpath(os.path.join(root, dirname))
                    os.rmdir(
                        path
                    )
                    print("cleaned", path)
                except OSError:
                    pass


    # overload some functions
    def fetch_artifacts(self, **params):
        try:
            return [Artifact(**i) for i in self.fetch(**params).json()]
        except TypeError:
            raise TypeError(f"Could not fetch artifacts with {params}")

    def set_artifacts(self, artifacts, **params):
        self.fetch(**params).set([i.to_dict() for i in artifacts])

    def cached(self):
        store = Datastore(self.root_dir)
        store.fetch_artifacts = lru_cache(None)(store.fetch_artifacts)
        return store


class DataObject:
    def __init__(self, root_dir, params):
        self.root_dir = root_dir
        self.params = params
        self.dir_path = os.path.join(root_dir, self.make_dir_path())
        params_file = os.path.join(self.dir_path, 'params.json')

        if not os.path.exists(params_file):
            os.makedirs(self.dir_path, exist_ok=True)
            with open(params_file, 'w') as f:
                json.dump(self.params, f)

    def make_dir_path(self):
        return make_dir_path(self.params)

    def set(self, data):
        try:
            # Attempt to serialize to JSON
            json.dumps(data)
            data_file = os.path.join(self.dir_path, 'data.json')
            with open(data_file, 'w') as f:
                json.dump(data, f)
        except TypeError:
            # If JSON serialization fails, use pickle
            print(f"JSON serialization failed for {self.params}, using pkl")
            data_file = os.path.join(self.dir_path, 'data.pkl')
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)

    def delete(self):
        import shutil
        shutil.rmtree(self.dir_path)

    @lru_cache(1)
    def pkl(self):
        pkl_file = os.path.join(self.dir_path, 'data.pkl')
        if os.path.exists(pkl_file):
            with open(pkl_file, 'rb') as f:
                return pickle.load(f)
        return None

    @lru_cache(1)
    def json(self):
        json_file = os.path.join(self.dir_path, 'data.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                return json.load(f)
        return None

p = os.path

def default_artifacts_root():
    root = p.abspath(__file__)
    for _ in range(4):
        root = p.dirname(root)
    return p.join(root, "rapidresponseartifacts")

ARTIFACTS_ROOT = os.getenv("ARTIFACTS_ROOT") or default_artifacts_root()
ATTACK = Datastore(p.join(ARTIFACTS_ROOT, "attack"))
DEFENSE = Datastore(p.join(ARTIFACTS_ROOT, "defense"))
PROLIFERATION = Datastore(p.join(ARTIFACTS_ROOT, "proliferation"))
PROLIFERATION_TEST = Datastore(p.join(ARTIFACTS_ROOT, "proliferation_test"))
RESULT = Datastore(p.join(ARTIFACTS_ROOT, "result"))


class Artifact:

    def __init__(
            self,
            chat,
            is_benign=None,
            query=None,
            attack_method=None,
            target_model=None,
            response=None,
            encoded_response=None, # if this is a cipher, this stores the raw response, while response
                                   # contains the decoded response
            is_guard_refusal=None,
            is_policy_jailbreak=None,
            is_policy_refusal=None,

            is_request_harmful=None,
    ):

        if isinstance(chat, str):
            chat = [
                {
                    "role": "user",
                    "content": chat
                }
            ]

        self._chat = chat
        self.is_benign = is_benign

        self.query = query
        self.attack_method = attack_method
        self.target_model = target_model
        self.response = response
        self.encoded_response = encoded_response

        self.is_guard_refusal = is_guard_refusal
        self._is_policy_jailbreak = is_policy_jailbreak
        self._is_policy_refusal = is_policy_refusal

        # for proliferations: determining intent preservation
        self.is_request_harmful = is_request_harmful

        if self._is_policy_refusal is not None:
            assert is_benign is not None
            assert is_benign
        if self._is_policy_jailbreak is not None:
            assert is_benign is not None
            assert not is_benign

    @property
    def is_policy_jailbreak(self):
        assert not self.is_benign
        assert self._is_policy_jailbreak is not None
        return self._is_policy_jailbreak

    @property
    def is_policy_refusal(self):
        assert self.is_benign
        assert self._is_policy_refusal is not None
        return self._is_policy_refusal

    @property
    def jailbroken(self):
        # if is_guard_refusal == None is falsy
        return self.is_policy_jailbreak and not self.is_guard_refusal

    @property
    def chat(self):
        return deepcopy(self._chat)

    @property
    def true_response(self):
        if self.encoded_response is not None:
            return self.encoded_response
        return self.response

    @property
    def prompt(self):
        if len(self._chat) == 1:
            return self.chat[-1]["content"]
        else:
            res = ""
            for i in self._chat:
                res += f"<{i['role'].upper()}>{i['content']}</{i['role'].upper()}>"
            return res

    def copy(self):
        return deepcopy(self)

    def to_dict(self):
        return {
            "chat": self.chat,
            "is_benign": self.is_benign,
            "query": self.query,
            "attack_method": self.attack_method,
            "target_model": self.target_model,
            "response": self.response,
            "encoded_response": self.encoded_response,
            "is_guard_refusal": self.is_guard_refusal,
            "is_policy_jailbreak": self._is_policy_jailbreak,
            "is_policy_refusal": self._is_policy_refusal,
            "is_request_harmful": self.is_request_harmful,
        }


def generate_behaviors():
    from datasets import load_dataset
    import random
    all_data = load_dataset("Lemhf14/EasyJailbreak_Datasets", "AdvBench")
    all_data = list(all_data['train'])
    rng = random.Random(1337)
    # 300 samples
    # 100 -> OOD test
    # 100 -> IID test
    # 100 -> IID train
    samples = rng.sample(all_data, k=300)
    rng.shuffle(samples)

    class Behavior:
        def __init__(self, query):
            self.query = query

    def make_behaviors(l):
        return [Behavior(i["query"]) for i in l]

    return {
        "train": make_behaviors(samples[:100]),
        "test_iid": make_behaviors(samples[100:200]),
        "test_ood": make_behaviors(samples[200:])
    }

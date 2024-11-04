from rapidresponsebench import INTERNAL_ACCEPT
from rapidresponsebench import Artifact


class MockLLM:
    def query(self, prompts, *args, **kwargs):
        return [INTERNAL_ACCEPT] * len(prompts)


class RemoteLLM:
    def __init__(self, model_name, concurrency=25):
        from rapidresponsebench.utils import Converser
        self.model_name = model_name
        self.model = Converser(model=model_name)
        self.concurrency = concurrency

    def query(self, prompts, *args, **kwargs):
        from rapidresponsebench.utils import parallel_map

        def run_artifact(artifact, **kw):
            clone = self.model.clone()
            clone.chat = artifact.chat
            return clone.say(None, **kw)

        use_tqdm = len(prompts) > 10

        return parallel_map(
                run_artifact,
                prompts,
                **kwargs,
                use_tqdm=use_tqdm,
                concurrency=self.concurrency,
            )


class BaseResponse:

    def __init__(self, llm=None):
        """

        :param llm: JailBreakBench LLM wrapper
        :param defense: JailBreakBench defense class
        """

        if llm is None:
            llm = MockLLM()

        self.llm = llm

    def query(
            self,
            prompts: list[Artifact],
            *args,
            **kwargs
    ) -> list[str]:
        """
        get output from the entire rapid response system

        :param prompts: potential jailbreak prompts
        :return:
        """
        return self.llm.query(prompts, *args, **kwargs)

    def notify(self, prompts: list[Artifact]):
        raise NotImplementedError

    def notify_benign(self, prompts: list[Artifact]):
        raise NotImplementedError


class NullResponse(BaseResponse):
    def notify(self, prompts: list[Artifact]):
        pass

    def notify_benign(self, prompts: list[str]):
        pass

from lm_eval.api.task import ConfigurableTask


def humaneval_nanochat(**kwargs):
    return ConfigurableTask.from_yaml("humaneval_nanochat", **kwargs)

from lm_eval.api.task import ConfigurableTask


def gsm8k_nanochat(**kwargs):
    return ConfigurableTask.from_yaml("gsm8k_nanochat", **kwargs)

import sys

from openprompt.plms import _MODEL_CLASSES, ModelClass
from transformers import (
    AutoTokenizer,
    MT5Config,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
)

from utils.tokenizer import MT5TokenizerWrapper


# some checkpoints use different tokenizers (e.g. directly PreTrainedTokenizerFast) and thus give an error if loaded from the model-specific class
# therefore we replace all model classes to use transformer's AutoTokenizer
class SlowDefaultAutoTokenizer(AutoTokenizer):
    # this class makes sure that always a slow tokenizer is created

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if "use_fast" not in kwargs:
            kwargs.update(use_fast=False)
        return super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)


def fix_openprompt_tokenizers():
    # add more models to OpenPrompt
    _MODEL_CLASSES.update(
        mt5=ModelClass(
            **{
                "config": MT5Config,
                "tokenizer": MT5Tokenizer,
                "model": MT5ForConditionalGeneration,
                "wrapper": MT5TokenizerWrapper,
            }
        ),
    )

    for model_class_name, model_class in _MODEL_CLASSES.items():
        _MODEL_CLASSES[model_class_name] = model_class._replace(
            tokenizer=SlowDefaultAutoTokenizer
        )


if sys.stdin is None or not sys.stdin.isatty():
    # replace input function
    safe_input = lambda *args: None
    safe_print = lambda *args: None
else:
    safe_input = input
    safe_print = print

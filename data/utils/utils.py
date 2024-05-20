import argparse
import collections.abc
import dataclasses
import errno
import logging
import os
import random
import sys
import time
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import datasets
import numpy
import psutil
import torch
import transformers
import yaml
from datasets import Dataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm
from transformers import TrainerCallback, integrations
from transformers.hf_argparser import DataClass, DataClassType

GLOBAL_SEED = None


def setup(log_level: str, pre_allocate_gpu: bool, use_tqdm: bool = True):
    # configure logging
    format_str = "[%(levelname)s - %(name)s - %(asctime)s] %(message)s"
    datefmt_str = "%d-%m-%Y %H:%M:%S"
    logging.basicConfig(
        format=format_str,
        datefmt=datefmt_str,
        level=(
            logging.INFO
            if log_level == "passive"
            else logging.getLevelName(log_level.upper())
        ),
        stream=TqdmStream(sys.stderr) if use_tqdm else sys.stderr,
    )
    if log_level == "debug":
        transformers.logging.set_verbosity_debug()
        datasets.logging.set_verbosity_debug()
        logging.debug("Enabled debug mode")
    else:
        transformers.logging.set_verbosity_info()
        datasets.logging.set_verbosity_info()

    if pre_allocate_gpu and torch.cuda.is_available():
        # allocate all available memory
        allocate_mem()


def separate_map_on_prefix(model_args, *prefixes, strict: bool = True):
    new_dicts = [{} for _ in prefixes]
    for key, value in model_args.items():
        for i, prefix in enumerate(prefixes):
            if key.startswith(prefix):
                new_dicts[i][key[len(prefix) :]] = value

    remaining_args = set(model_args.keys())
    for prefix, dict_ in zip(prefixes, new_dicts):
        remaining_args.difference_update(map(lambda x: prefix + x, dict_))

    if strict and remaining_args:
        raise ValueError(
            f"Not all model arguments have been consumed, the remaining are: {', '.join(remaining_args)}"
        )

    return *new_dicts, remaining_args


def dicts_to_feature_dict(dict_iter):
    output = defaultdict(list)
    for _dict in dict_iter:
        for key, value in _dict.items():
            output[key].append(value)
    return output


def timing(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(f"Execution of function {func.__name__} took {t2-t1:f}s")
        return res

    return wrapper


def monkeypatch():
    # we jut replace the callback for some integrations
    integrations.INTEGRATION_TO_CALLBACK["comet_ml"] = CometCallback
    integrations.INTEGRATION_TO_CALLBACK["wandb"] = WandbCallback


class CometCallback(integrations.CometCallback):
    """
    A :class:`~transformers.integrations.CometCallback` that may continue a previous experiment if environment variable `COMET_EXPERIMENT_KEY` is set.
    Taken from https://www.comet.ml/docs/python-sdk/continue-experiment/.
    Instantiate a callback from this class or use for monkeypatching.
    """

    # log_args = ['per_device_train_batch_size', 'gradient_accumulation_steps', 'learning_rate', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'max_grad_norm', 'lr_scheduler_type', 'warmup_steps', 'fp16']

    def setup(self, args, state, model):
        """
        Setup the optional Comet.ml integration.

        Environment:
            COMET_MODE (:obj:`str`, `optional`):
                "OFFLINE", "ONLINE", or "DISABLED"
            COMET_PROJECT_NAME (:obj:`str`, `optional`):
                Comet.ml project name for experiments
            COMET_OFFLINE_DIRECTORY (:obj:`str`, `optional`):
                Folder to use for saving offline experiments when :obj:`COMET_MODE` is "OFFLINE"

        For a number of configurable items in the environment, see `here
        <https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables>`__.
        """
        self._initialized = True
        if state.is_world_process_zero:
            # Check to see if there is a key in environment:
            EXPERIMENT_KEY = os.environ.get("COMET_EXPERIMENT_KEY", None)

            # First, let's see if we continue or start fresh:
            CONTINUE_RUN = False
            if EXPERIMENT_KEY is not None:
                # There is one, but the experiment might not exist yet:
                api = (
                    integrations.comet_ml.API()
                )  # Assumes API key is set in config/env
                try:
                    api_experiment = api.get_experiment_by_id(EXPERIMENT_KEY)
                except Exception:
                    api_experiment = None
                if api_experiment is not None:
                    CONTINUE_RUN = True
                    integrations.logger.info(
                        "Continuing Comet.ml experiment %s", api_experiment.get_name()
                    )

            if CONTINUE_RUN:
                # Setup the existing experiment to carry on:
                experiment = integrations.comet_ml.ExistingExperiment(
                    previous_experiment=EXPERIMENT_KEY,
                    log_env_details=True,  # to continue env logging
                    log_env_gpu=True,  # to continue GPU logging
                    log_env_cpu=True,  # to continue CPU logging
                )
            else:
                super().setup(args, state, model)

            # add info to experiment
            experiment = integrations.comet_ml.config.get_global_experiment()
            if experiment is not None:
                if getattr(args, "run_name", None) is not None:
                    experiment.set_name(args.run_name)
                if getattr(args, "tags", None) is not None:
                    for tag in args.tags:
                        experiment.add_tag(tag)


class WandbCallback(integrations.WandbCallback):
    """
    A [`TrainerCallback`] that sends the logs to [Weight and Biases](https://www.wandb.com/).
    """

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.

        One can subclass and override this method to customize the setup if needed. Find more information
        [here](https://docs.wandb.ai/integrations/huggingface). You can also override the following environment
        variables:

        Environment:
            WANDB_LOG_MODEL (`bool`, *optional*, defaults to `False`):
                Whether or not to log model as artifact at the end of training. Use along with
                *TrainingArguments.load_best_model_at_end* to upload best model.
            WANDB_WATCH (`str`, *optional* defaults to `"gradients"`):
                Can be `"gradients"`, `"all"` or `"false"`. Set to `"false"` to disable gradient logging or `"all"` to
                log gradients and parameters.
            WANDB_PROJECT (`str`, *optional*, defaults to `"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (`bool`, *optional*, defaults to `False`):
                Whether or not to disable wandb entirely. Set *WANDB_DISABLED=true* to disable.
        """

        super().setup(args, state, model, **kwargs)

        if not self._initialized:
            return

        # add some aggregate values to wandb
        for metric in [
            "em_answer",
            "em_ha_1",
            "em_ha_5",
            "f1",
            "f1_answer",
            "f1_ha_1",
            "f1_ha_5",
            "loss",
            "loss",
        ]:
            self._wandb.define_metric(f"eval/{metric}", summary="min")
            self._wandb.define_metric(f"eval/{metric}", summary="max")


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    train_prefix = "train_"
    train_prefix_len = len(train_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        elif k.startswith(train_prefix):
            # seems like train metrics already start with 'train_'
            new_d["train/" + k[train_prefix_len:]] = v
        else:
            # new_d["train/" + k] = v
            new_d[k] = v
    return new_d


class PyLoggerCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that logs to Python objects.
    Args:
    """

    def __init__(self):
        self._disabled = False
        self.init()

    def init(self):
        self._logs = defaultdict(list)

    def disable(self):
        self._disabled = True

    def enable(self):
        self._disabled = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

    def on_log(self, args, state, control, logs=None, **kwargs):
        # input(f"LOGGING using {self}: {'Disabled' if self._disabled else 'Enabled'}")
        # input(f"Logs: {logs}")
        if self._disabled or not state.is_world_process_zero:
            return

        # print("===== Log =====")
        # input(logs.keys())

        # self.logs['steps'].append(state.global_step)
        for k, v in rewrite_logs(logs).items():
            self._logs[k].append(v)

    def get_logs(self):
        return self._logs


@dataclass
class DataCollatorList:
    collators: Iterable

    def __call__(self, features):
        for collator in self.collators:
            features = collator(features)
        return features


@dataclass
class ProcessDataDataCollator:
    process_fn: Callable

    def __call__(self, features: Union[List[Dict], Dict], **kwargs):
        if isinstance(features, dict):
            # batched samples
            return self.process_fn(features)
        else:
            # several individual samples (single batches)
            return [self.process_fn(feature, batched=False) for feature in features]


class HfArgumentParser(transformers.HfArgumentParser):
    def __init__(
        self,
        dataclass_types: Union[DataClassType, Iterable[DataClassType]] = None,
        **kwargs,
    ):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will "fill" instances with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular way.
        """
        # call super constructor of HfArgumentParser
        argparse.ArgumentParser.__init__(self, **kwargs)

        if dataclass_types is not None:
            if dataclasses.is_dataclass(dataclass_types):
                dataclass_types = [dataclass_types]
            self.dataclass_types = dataclass_types
            for dtype in self.dataclass_types:
                self._add_dataclass_arguments(dtype)
        else:
            self.dataclass_types = []

    def parse_args_into_dataclasses(
        self,
        args=None,
        return_remaining_strings=False,
        look_for_args_file=True,
        args_filename=None,
    ) -> Tuple[DataClass, ...]:
        if args_filename or (look_for_args_file and len(sys.argv)):
            if args_filename:
                args_file = Path(args_filename)
            else:
                args_file = Path(sys.argv[0]).with_suffix(".args")

            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + args if args is not None else fargs + sys.argv[1:]
                # in case of duplicate arguments the first one has precedence
                # so we append rather than prepend.
        namespace, _ = self.parse_known_args(args=args)

        for action in self._get_positional_actions():
            # we only look for (sub)parsers
            if isinstance(action, argparse._SubParsersAction):
                parser = action.choices[getattr(namespace, action.dest)]
                if isinstance(parser, HfArgumentParser):
                    self.dataclass_types.extend(parser.dataclass_types)

        return super().parse_args_into_dataclasses(
            args=args,
            return_remaining_strings=return_remaining_strings,
            look_for_args_file=look_for_args_file,
            args_filename=args_filename,
        )


def check_positive(string, type=int):
    msg = "invalid value: %r (has to be a positive value of %s)" % (string, type)
    try:
        value = type(string)
    except:
        raise argparse.ArgumentTypeError(msg)
    if value <= 0:
        raise argparse.ArgumentTypeError(msg)
    return value


def is_remote_url(url: str):
    return urllib.parse.urlparse(url).scheme not in ["", "file"]


def expand_path(path: str):
    if not is_remote_url(path):
        path = os.path.abspath(os.path.expanduser(path))
    return path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class TqdmStream(object):
    """Dummy file-like that will write to tqdm
    https://github.com/tqdm/tqdm/issues/313
    """

    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file, end="")

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


# from https://stackoverflow.com/a/48774926
def tabulate_events(dirs):
    summary_iterators = [EventAccumulator(dir).Reload() for dir in dirs]

    tags = summary_iterators[0].Tags()["scalars"]

    out = defaultdict(list)
    steps = defaultdict(list)

    for tag in tags:
        for events in zip_longest(
            *[
                acc.Scalars(tag)
                for acc in summary_iterators
                if tag in acc.Tags()["scalars"]
            ]
        ):
            out[tag].append([e.value for e in events if e is not None])

    return out, steps


# from https://stackoverflow.com/a/48774926
def write_events_averaged(writer, d_combined, steps):
    tags, values = zip(*d_combined.items())

    for tag, _values in zip(tags, values):
        for i, mean in zip(steps[tag], numpy.array(_values).mean(axis=-1)):
            writer.add_scalar(tag, mean, global_step=i)

        writer.flush()


def get_best_std(d_combined, eval_steps=None):
    tags, values = zip(*d_combined.items())
    metrics = {}
    for tag, _values in zip(tags, values):
        # _values can have varying amount of columns per row since experiments can have a varying number of steps
        _values = [
            torch.tensor([_value for _value in __values if _value is not None])
            for __values in _values
        ]
        if eval_steps is None:
            max_values = torch.stack([_value.max(dim=0).values for _value in _values])
            min_values = torch.stack([_value.min(dim=0).values for _value in _values])
            last_values = torch.stack([_value[-1] for _value in _values])

            metrics[tag] = {
                "highest": {
                    "min": max_values.min().item(),
                    "max": max_values.max().item(),
                    "mean": max_values.mean(dtype=torch.float).item(),
                    "std": max_values.float().std(unbiased=False).item(),
                },
                "lowest": {
                    "min": min_values.min().item(),
                    "max": min_values.max().item(),
                    "mean": min_values.mean(dtype=torch.float).item(),
                    "std": min_values.float().std(unbiased=False).item(),
                },
                "last": {
                    "min": last_values.min().item(),
                    "max": last_values.max().item(),
                    "mean": last_values.mean(dtype=torch.float).item(),
                    "std": last_values.float().std(unbiased=False).item(),
                },
            }
        else:
            if not (tag.startswith("eval/") or tag.startswith("test/")):
                # logging steps for train and eval might differ hence we consider only evaluation for best model checkpoints
                continue
            steps_values = torch.stack(
                [__values[step] for __values, step in zip(_values, eval_steps)]
            )
            metrics[tag] = (
                {
                    "min": steps_values.min().item(),
                    "max": steps_values.max().item(),
                    "mean": steps_values.mean().item(),
                    "std": steps_values.std(unbiased=False).item(),
                },
            )
    return metrics


def init_random(seed: int = None):
    """
    Initializes the random generators to allow seeding.
    Args:
        seed (int): The seed used for all random generators.
    """
    global GLOBAL_SEED  # pylint: disable=global-statement

    if seed is None:
        random.Random(None)
        tmp_random = random.Random(None)
        GLOBAL_SEED = tmp_random.randint(0, 2**32 - 1)
    else:
        GLOBAL_SEED = seed

    # initialize random generators
    numpy.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)

    try:
        # try to load torch and initialize random generator if available
        import torch

        torch.cuda.manual_seed_all(GLOBAL_SEED)  # gpu
        torch.manual_seed(GLOBAL_SEED)  # cpu
    except ImportError:
        pass

    try:
        # try to load tensorflow and initialize random generator if available
        import tensorflow  # type: ignore

        tensorflow.random.set_seed(GLOBAL_SEED)
    except ImportError:
        pass

    logging.info("Seed is %d", GLOBAL_SEED)
    return GLOBAL_SEED


def print_mem_usage():
    process = psutil.Process(os.getpid())
    bytes_in_memory = process.memory_info().rss
    print(f"Current memory consumption: {bytes_in_memory/1024/1024:.0f} MiB")
    # print(f"Current memory consumption:\n  Bytes: {bytes_in_memory}\n  Kibibytes: {bytes_in_memory/1024}\n  Mebibytes: {bytes_in_memory/1024/1024}\n  Gibibytes: {bytes_in_memory/1024/1024/1024}")


def check_mem():
    mem = [
        gpu.split(",")
        for gpu in os.popen(
            '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
        )
        .read()
        .strip()
        .split("\n")
    ]
    return mem


def get_gpus():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        try:
            gpus = list(
                map(int, os.environ["CUDA_VISIBLE_DEVICES"].strip("[]").split(","))
            )
        except ValueError:
            # can only occur if it hasn't been checked for torch.cuda.is_available()
            gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = list(range(torch.cuda.device_count()))
    return gpus


def allocate_mem(percentage: float = 0.9):
    # this will allocate GPU memory
    # the allocator is cached until torch.cuda.empty_cache() is called (or the program ends)
    torch.cuda.init()
    _ = torch.empty(0)

    gpus = get_gpus()
    gpu_mem = check_mem()
    for gpu_id, gpu in enumerate(gpus):
        total, used = map(int, gpu_mem[gpu])

        max_mem = (
            total * percentage
        )  # somehow allocatable memory is always lower than total memory
        block_mem_floats = int(
            (max_mem - used) * 1024 * 1024 / 4
        )  # from float to MiB // one float is 4 byte
        if block_mem_floats >= 0:
            x = torch.empty(
                block_mem_floats,
                device=torch.device("cuda:%d" % gpu_id),
                dtype=torch.float32,
            )
            del x  # actually not necessary as pointer is removed once function returns (and gc runs)
        else:
            logging.warning("Cannot allocate memory on gpu %d: maximum exceeded" % gpu)


def select_unique(data: Dataset, column: str, seed=None, verbose: bool = False):
    if seed is not None:
        data = data.shuffle(seed=seed)
    unique_set = set()

    def filter_unique(sample):
        if sample[column] not in unique_set:
            unique_set.add(sample[column])
            return True
        if verbose:
            print(
                f"Value '{sample[column]}' appeared multiple times for column '{column}'"
            )
        return False

    return data.filter(filter_unique, num_proc=1)


def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def store_metrics(dir, filename, metrics, prefix=None):
    path = os.path.join(os.path.abspath(os.path.expanduser(dir)), filename + ".yaml")
    metrics = {prefix: metrics} if prefix is not None else {int(GLOBAL_SEED): metrics}

    try:
        with open(path, "r") as metric_file:
            tmp = yaml.full_load(metric_file)
            update_dict(tmp, metrics)
            metrics = tmp
    except:
        pass

    # create necessary dirs
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(path, "w") as metric_file:
        yaml.dump(metrics, metric_file, indent=4, sort_keys=True)


if __name__ == "__main__":
    allocate_mem(0.9659)
    while True:
        time.sleep(10)

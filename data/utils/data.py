import configparser
import logging
import os
import re
import string
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy
import stanza
import torch
from datasets import (
    Dataset,
    DatasetDict,
    DownloadConfig,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .utils import dicts_to_feature_dict, print_mem_usage


class EnvInterpolation(configparser.ExtendedInterpolation):
    """Interpolation which expands environment variables in values."""

    def before_get(self, parser, section, option, value, defaults):
        value = super().before_get(parser, section, option, value, defaults)
        return os.path.expandvars(value)


config = configparser.ConfigParser(
    inline_comment_prefixes="#", interpolation=EnvInterpolation()
)
config.read(os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets.ini"))
DATASETS = {}
for section in config.sections():
    DATASETS[section] = {}
    for option, value in config.items(section):
        if option == "loader":
            # value is treated as a generic dataser builder
            DATASETS[section]["path"] = value
        elif option == "builder_path":
            # value is treated as a path in data/datasets
            DATASETS[section]["path"] = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "datasets", value
            )
        elif option == "path":
            # value is treated as a path relative to the top-level package
            DATASETS[section]["path"] = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), value
            )
        elif option == "data_files":
            # value can be a relative path
            DATASETS[section]["data_files"] = os.path.abspath(value)
        else:
            DATASETS[section][option] = value

RE_PUNCTUATION_BASIC = re.compile(
    r'[,.\-;:_!?“”‘’\'"\[\]()]'
)
RE_PUNCTUATION = re.compile(r"[%s]" % re.escape(string.punctuation))
RE_PUNCTUATION_NO_DOLLAR = re.compile(
    r"[%s]" % string.punctuation.replace("$", "")
)  # excluding '$' (useful if it appears in the text as part of the answer)
RE_PUNCTUATION_NO_DOLLAR_QUOTE = re.compile(
    r"[%s]" % string.punctuation.replace("$", "").replace('"', "")
)  # excluding '$' and '"' (useful if both appear in the text as part of the answer)
RE_DASH = re.compile(r"[\u2013\u2014]")  # a dash '–' (not a hyphen '-')
RE_ARTICLES = re.compile(r"\b(a|an|the)\b")
RE_WHITESPACE = re.compile(r"[\s]+")


def normalize_answer(answer: str):
    def lower(answer: str):
        return answer.lower()

    def remove_punctuation(answer: str):
        return re.sub(RE_PUNCTUATION, "", answer)

    def remove_articles(answer: str):
        return re.sub(RE_ARTICLES, "", answer)

    def remove_whitespace(answer: str):
        return re.sub(RE_WHITESPACE, " ", answer.strip())

    if answer is None:
        return answer
    return remove_whitespace(remove_articles(remove_punctuation(lower(answer))))


def rectify_sample(sample, disable: bool = False):
    if disable:
        return sample
    if (
        "question" in sample
        and sample["question"] is not None
        and not sample["question"].endswith("?")
    ):
        sample["question"] += "?"
    return sample


def get_per_sample_indices(separator, *to_split):
    # NOTE this does expect that chunks of the same question only occur consecutively
    _, indices, counts = numpy.unique(separator, return_index=True, return_counts=True)
    counts = [counts[index] for index in numpy.argsort(indices)]
    split_indices = torch.arange(len(separator)).split(counts, dim=0)  # torch version
    # split_indices = numpy.split(range(len(separator)), numpy.sort(indices)[1:]) # numpy version
    return (
        [[item[i] for i in indices] for indices in split_indices] for item in to_split
    )  # does the same as torch.split but for any datatype
    return (torch.tensor(item).split(counts, dim=0) for item in to_split)


def process_samples_batched(
    batch_sample: Dict[str, Any], functions: Tuple[Callable], update: bool = False
):
    def apply_functions(samples):
        for function in functions:
            samples = [
                {**sample, **processed_sample} if update else processed_sample
                for sample in samples
                for processed_sample in function(**sample)
            ]
        return samples

    """ Method that can be used for datasets.map()

    This method takes a `dict` containing the batched items in the values,
    feeds one sample at a time to `process_sample` and transforms all
    output into a single dict to have the same structure as the input data.

    NOTE `process_sample` cannot be used directly for datasets.map() because it returns a generator
    """
    batch_size = len(next(iter(batch_sample.values())))
    samples = apply_functions(
        [
            {feature: values[i] for feature, values in batch_sample.items()}
            for i in range(batch_size)
        ]
    )
    return {
        feature: [sample[feature] for sample in samples]
        for feature in samples[0].keys()
    }


def unpack_samples(packed_samples, filter_fn=None):
    samples = {
        "id": [],
        "original_id": [],
        "context": [],
        "question": [],
        "answers": [],
        "score": [],
    }
    for gen_samples in tqdm(packed_samples, desc="Unpacking samples", unit="samples"):
        # gen_samples is a dict
        # apply filter function
        if filter_fn is not None:
            questions, answers, scores = filter_fn(
                context=gen_samples["context"],
                questions=gen_samples["questions"],
                answers=gen_samples["answers"],
                scores=gen_samples["scores"],
            )
        else:
            questions, answers, scores = (
                gen_samples["questions"],
                gen_samples["answers"],
                gen_samples["scores"],
            )
        for idx, (question, answer, score) in enumerate(
            zip(questions, answers, scores)
        ):
            # we append the counter to the original id to use as id for the new sample in order to have unique ids
            samples["id"].append(f"{gen_samples['id']}_{idx}")
            samples["original_id"].append(gen_samples["id"])
            samples["context"].append(gen_samples["context"])
            samples["question"].append(question)
            samples["answers"].append(answer)
            samples["score"].append(score)
    return Dataset.from_dict(samples)


def process_hf_dataset(
    data: Union[Dataset, DatasetDict],
    preprocess_function: Callable,
    remove_old_columns: bool = False,
    force_preprocess: bool = False,
    keep_in_memory: bool = False,
    num_processes: int = 1,
    fn_kwargs: dict = None,
    **map_kwargs,
):
    # print(data, preprocess_function, fn_kwargs)
    if not data or not preprocess_function:
        return None

    if remove_old_columns:
        if isinstance(data, DatasetDict):
            remove_columns = data["train"].column_names
        else:
            remove_columns = data.column_names
    else:
        remove_columns = None

    # process data in batch mode since output size is changed (multiple chunks per sample)
    processed_data = data.map(
        preprocess_function,
        fn_kwargs=fn_kwargs,
        batched=True,
        remove_columns=remove_columns,
        load_from_cache_file=not force_preprocess,
        keep_in_memory=keep_in_memory,
        num_proc=num_processes,
        **map_kwargs,
    )
    return processed_data


def expand_answers(
    data: Dataset,
    separate_answers,
    force_preprocess: bool = False,
    keep_in_memory: bool = False,
    num_processes: int = 1,
):
    if "answers" not in data.column_names or isinstance(
        data[0]["answers"]["text"][0], str
    ):
        logging.info("Skipping expanding answers")
        return data
    logging.info(
        f"Expanding answers: {'new instances' if separate_answers else 'simple'}"
    )
    num_samples = len(data)
    data = data.map(
        unpack_answers,
        fn_kwargs={"separate_answers": separate_answers},
        batched=True,
        load_from_cache_file=not force_preprocess,
        keep_in_memory=keep_in_memory,
        num_proc=num_processes,
    )
    logging.info(f"{len(data)} samples after expansion ({len(data)-num_samples} added)")
    return data


nlp = None


def d_truncate_context(
    samples: Dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    consider_question: bool,
    consider_answer: bool,
    truncate_sentences_only: bool,
    remove_instance_if_context_does_not_contain_answer: bool,
):
    samples["num_truncated_tokens"] = [None] * len(samples["context"])
    ignore_indices = []
    for idx in range(len(samples["context"])):
        max_context_length = max_length
        if consider_question:
            max_context_length -= len(tokenizer.tokenize(samples["question"][idx]))
        if consider_answer:
            max_context_length -= len(
                tokenizer.tokenize(samples["answers"][idx]["text"][0])
            )

        num_context_tokens = len(tokenizer.tokenize(samples["context"][idx]))

        if truncate_sentences_only:
            raise ValueError("Not implemented")
        else:
            # context could be shorter than max_context length hence we slice first and use the last token afterwards
            end_idx = tokenizer(
                samples["context"][idx],
                add_special_tokens=False,
                return_offsets_mapping=True,
            )["offset_mapping"][:max_context_length][-1][1]
            samples["context"][idx] = samples["context"][idx][:end_idx]
            # consider only first answer span
            if (
                remove_instance_if_context_does_not_contain_answer
                and (
                    samples["answers"][idx]["answer_start"][0]
                    + len(samples["answers"][idx]["text"][0])
                )
                >= end_idx
            ):
                # answer not within context
                ignore_indices.append(idx)
            samples["num_truncated_tokens"][idx] = max(
                0, num_context_tokens - max_context_length
            )

    if not ignore_indices:
        return samples
    return {
        key: [value for _idx, value in enumerate(values) if _idx not in ignore_indices]
        for key, values in samples.items()
    }


def truncate_context(
    data: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    consider_question: bool,
    consider_answer: bool,
    truncate_sentences_only: bool,
    additional_length: int = 0,
    remove_instance_if_context_does_not_contain_answer: bool = None,
    force_preprocess: bool = False,
    keep_in_memory: bool = False,
    num_processes: int = 1,
):
    global nlp

    logging.info(
        f"Truncating contexts so that model input has at most {max_length} tokens"
    )
    if truncate_sentences_only and nlp is None:
        # initialize tokenizer (for sentence segmentation)
        nlp = stanza.Pipeline(lang="en", processors="tokenize", verbose=False)
    num_samples = len(data)
    data = data.map(
        d_truncate_context,
        fn_kwargs=dict(
            tokenizer=tokenizer,
            max_length=max_length - additional_length,
            consider_question=consider_question,
            consider_answer=consider_answer,
            truncate_sentences_only=truncate_sentences_only,
            remove_instance_if_context_does_not_contain_answer=remove_instance_if_context_does_not_contain_answer,
        ),
        batched=True,
        load_from_cache_file=not force_preprocess,
        keep_in_memory=keep_in_memory,
        num_proc=num_processes,
        desc="Truncating contexts",
    )
    num_truncated_samples = len(
        data.filter(
            lambda sample: sample["num_truncated_tokens"] > 0,
            desc="Filtering for truncated samples",
        )
    )
    num_truncated_tokens_average = (
        (sum(data["num_truncated_tokens"]) / num_truncated_samples)
        if num_truncated_samples > 0
        else 0
    )
    logging.info(
        f"Truncated  {num_truncated_samples} samples with an average truncation of {num_truncated_tokens_average} tokens"
    )
    logging.info(f"Removed {num_samples-len(data)} samples during truncation")
    return data


def d_chunk_context(
    samples: Dict,
    mode: str,
    context_stride: int,
    context_size: int,
    remove_instance_if_context_does_not_contain_answer: bool,
    tokenizer: PreTrainedTokenizer = None,
):
    # mode is either 'token' or 'sentence' and 'context_stride' and 'context_size' adapt to these modes
    new_samples = defaultdict(list)
    for sample_idx in range(len(samples["context"])):
        context = samples["context"][sample_idx]
        if mode == "token":
            windows = tokenizer(
                context,
                add_special_tokens=False,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                truncation=True,
                max_length=context_size,
                stride=context_stride,
            ).offset_mapping
        elif mode == "sentence":
            windows = [
                nlp(context).sentences[_idx : _idx + context_size]
                for _idx in range(0, len(nlp(context).sentences), context_stride)
            ]
        else:
            raise NotImplementedError(
                f"Invalid chunking mode '{mode}', choose either 'token' or 'sentence'"
            )

        for chunk_idx, window in enumerate(windows):
            if mode == "token":
                start_idx = window[0][0]
                end_idx = window[-1][1]
            elif mode == "sentence":
                start_idx = window[0].words[0].start_char
                end_idx = window[-1].words[-1].end_char

            if "answers" in samples:
                if remove_instance_if_context_does_not_contain_answer:
                    answers = [
                        (answer_text, answer_start)
                        for answer_text, answer_start in zip(
                            samples["answers"][sample_idx]["text"],
                            samples["answers"][sample_idx]["answer_start"],
                        )
                        if start_idx <= answer_start
                        and end_idx >= (answer_start + len(answer_text))
                    ]

                    if not answers:
                        # answer not in context
                        continue
                    answers_text, answers_start = list(zip(*answers))
                else:
                    answers_text, answers_start = (
                        samples["answers"][sample_idx]["text"],
                        samples["answers"][sample_idx]["answer_start"],
                    )

                new_samples["answers"].append(
                    {
                        "answer_start": [
                            answer_start - start_idx for answer_start in answers_start
                        ],
                        "text": answers_text,
                    }
                )

            context_window = context[start_idx:end_idx]

            # add sample with new context
            new_samples["chunk_idx"].append(chunk_idx)
            new_samples["context"].append(context_window)
            for key in samples:
                if key not in ["context", "answers"]:
                    new_samples[key].append(samples[key][sample_idx])

            # we add the full context and the answers with the correct char offsets as well
            new_samples["context_original"].append(context)
            new_samples["offset_context_original"].append(start_idx)

    return new_samples


def chunk_context(
    data: Dataset,
    mode: str,
    context_stride: int,
    context_size: int,
    remove_instance_if_context_does_not_contain_answer: bool = None,
    tokenizer: PreTrainedTokenizer = None,
    force_preprocess: bool = False,
    keep_in_memory: bool = False,
    num_processes: int = 1,
):
    global nlp
    assert context_stride is not None and context_size is not None

    logging.info(
        f"Chunking contexts in mode {mode} with stride {context_stride} and size {context_size}"
    )
    if nlp is None:
        # initialize tokenizer (for sentence segmentation)
        nlp = stanza.Pipeline(lang="en", processors="tokenize", verbose=False)

    data = data.map(
        d_chunk_context,
        fn_kwargs=dict(
            mode=mode,
            context_stride=context_stride,
            context_size=context_size,
            remove_instance_if_context_does_not_contain_answer=remove_instance_if_context_does_not_contain_answer,
            tokenizer=tokenizer,
        ),
        batched=True,
        load_from_cache_file=not force_preprocess,
        keep_in_memory=keep_in_memory,
        num_proc=1,
        desc="Chunking contexts",
    )
    logging.info(f"{len(data)} samples after chunking contexts")
    return data


def unpack_answers(samples: Dict, separate_answers: bool = False):
    if "answers" not in samples or isinstance(samples["answers"][0]["text"][0], str):
        # samples do not contain an answer or are already unpacked
        return samples

    processed_samples = {k: [] for k in samples}
    keys = samples.keys()
    for values in zip(*[samples[k] for k in keys]):
        if separate_answers:
            # split answer to create new instances
            sample = dict(zip(keys, values))
            for answer_idx, (answer_start, text) in enumerate(
                zip(sample["answers"]["answer_start"], sample["answers"]["text"])
            ):
                # most keys stame the same, only special handling for 'answers' and 'id'
                for key in keys:
                    if key not in ["answers", "id"]:
                        processed_samples[key].append(sample[key])
                processed_samples["id"].append(sample["id"] + f"_ans#{answer_idx}")
                processed_samples["answers"].append(
                    {
                        "answer_start": answer_start,
                        "text": text,
                    }
                )
        else:
            # simply unpack answer
            for key, value in zip(keys, values):
                if key != "answers":
                    # copy everything except answer
                    processed_samples[key].append(value)
                else:
                    processed_samples[key].append(
                        {
                            "answer_start": [
                                answer_start
                                for list_answer_start in value["answer_start"]
                                for answer_start in list_answer_start
                            ],
                            "text": [
                                answer_text
                                for list_answer_text in value["text"]
                                for answer_text in list_answer_text
                            ],
                        }
                    )
    return processed_samples


@dataclass
class NamedDataset:
    data: Union[Dataset, List]
    path: str
    config: str
    split: str = None

    def __post_init__(self):
        if isinstance(self.data, DatasetDict):
            self.split = None
        elif self.split is None and self.data.split is not None:
            self.split = str(self.data.split)


def load_hf_dataset(path: str, keep_in_memory: bool = False, **kwargs):
    # check whether path can be loaded from disk with datasets library
    try:
        # try to load as DatasetDict
        dataset = DatasetDict.load_from_disk(path, keep_in_memory=keep_in_memory)
        if "split" in kwargs and kwargs["split"] is not None:
            return [
                NamedDataset(
                    DatasetDict.load_from_disk(path, keep_in_memory=keep_in_memory)[
                        kwargs["split"]
                    ],
                    path,
                    kwargs["split"],
                )
            ]
        return [
            NamedDataset(
                DatasetDict.load_from_disk(path, keep_in_memory=keep_in_memory),
                path,
                None,
            )
        ]
    except FileNotFoundError:
        pass
    try:
        # try to load as Dataset
        return [
            NamedDataset(
                Dataset.load_from_disk(path, keep_in_memory=keep_in_memory), path, None
            )
        ]
    except FileNotFoundError:
        pass

    download_config = (
        DownloadConfig(cache_dir=kwargs["cache_dir"]) if "cache_dir" in kwargs else None
    )
    # check if config is chosen in case dataset has several configs
    # if this is not the case then we load all configs and return them in a dict
    available_configs = get_dataset_config_names(path)
    if len(available_configs) > 1:
        if "name" in kwargs and kwargs["name"] is not None:
            # config specified
            config_obj = kwargs["name"]
            if isinstance(config_obj, str):
                configs = config_obj.strip("[]").split(",")
            else:
                configs = config_obj

            # configs could contain regex -> expand
            available_configs = get_dataset_config_names(path)
            configs_expanded = []
            for config in configs:
                if config[:2] == "r:":
                    # regex found, match with available configs
                    for available_config in available_configs:
                        if re.search(config[2:], available_config):
                            configs_expanded.append(available_config)
                else:
                    configs_expanded.append(config)
            configs = configs_expanded

            # load specified config if only one is given
            if len(configs) == 1:
                return [
                    NamedDataset(
                        load_dataset(
                            path,
                            **dict(kwargs, name=configs[0]),
                            download_config=download_config,
                        ),
                        path,
                        configs[0],
                    )
                ]
        else:
            # use all available configs
            # NOTE maybe better throw error since loading all configs is not obvious to the user and can easily be realized via regex
            configs = get_dataset_config_names(path)
        if not isinstance(configs, (tuple, list, set)):
            configs = [configs]
        return [
            NamedDataset(
                load_dataset(
                    path, **dict(kwargs, name=config), download_config=download_config
                ),
                path,
                config,
            )
            for config in configs
        ]
    else:
        assert len(available_configs) == 1
        return [
            NamedDataset(
                load_dataset(path, **kwargs, download_config=download_config),
                path,
                available_configs[0],
            )
        ]


def get_datasets(
    paths_or_names: Union[str, List[str]],
    cache_dir: str = None,
    concatenate: bool = False,
    unpack_fn=None,
    shuffle_seed: int = 42,
    recreate_index: bool = False,
    force_preprocess: bool = False,
    keep_in_memory: bool = False,
    num_processes: int = 1,
):
    """Download data and apply some simple preprocessing"""

    def get_dataset_args(path_or_name: str):
        # extract path, split and slice from dataset string
        return re.fullmatch(
            r"^(.*?)(?:\[(.+)\])?(?::(train|validation|test))??(?::((?:\d*%?)?:?\d*%?))?$",
            path_or_name,
        ).groups()

    def parse_slice(value: str):
        """
        Parses a `slice()` from string, like `start:stop:step`.
        """
        if value:
            parts = value.split(":")
            if len(parts) == 1:
                # slice(stop)
                parts = [None, parts[0]]
            # else: slice(start, stop[, step])
        else:
            # slice()
            parts = []
        return slice(*[int(p) if p else None for p in parts])

    if paths_or_names is None:
        return None

    if isinstance(paths_or_names, str):
        paths_or_names = [paths_or_names]
        concatenate = True
    logging.info(f"Loading dataset(s): {', '.join(paths_or_names)}")

    datasets = []

    # collect datasets
    for path_or_name in paths_or_names:
        path_or_name, config, split, slice_ = get_dataset_args(path_or_name)

        # check for custom datasets first
        if path_or_name in DATASETS:
            # custom name
            # DATASETS dict may contain split and/or information on how to split data
            # we choose split first (if available in DATASETS dict (loaded from datasets.ini)) because we split only object of type Dataset further
            dataset_kwargs = DATASETS[path_or_name].copy()
            train = dataset_kwargs.pop("train", None)
            validation = dataset_kwargs.pop("validation", None)
            test = dataset_kwargs.pop("test", None)
            shuffle = dataset_kwargs.pop("shuffle", None)
            if config is not None:
                # given config overrides config from .ini file
                dataset_kwargs.update(name=config)
            datasets_loaded = load_hf_dataset(
                **dataset_kwargs, cache_dir=cache_dir, keep_in_memory=keep_in_memory
            )
            for dataset in datasets_loaded:
                # split object of type Dataset according to given information
                if train is not None or validation is not None or test is not None:
                    assert isinstance(
                        dataset.data, Dataset
                    ), "train, validation or test split information given but object is not of type Dataset."
                    # we shuffle here for the following reasons:
                    #  1. given a seed in datasets.ini we can guarantee the same splits
                    #  2. shuffle=True in method train_test_split without a seed being set results in the same seed (bug?)
                    if shuffle is not None:
                        dataset.data = dataset.data.shuffle(int(shuffle))
                    if test is not None:
                        test_split = float(test)
                        test_datasetdict = dataset.data.train_test_split(
                            test_split, shuffle=False
                        )
                        train_validation, test = (
                            test_datasetdict["train"],
                            test_datasetdict["test"],
                        )
                    else:
                        test_split = 0.0
                        train_validation = dataset.data
                    if train is not None and validation is not None:
                        train_validation_datasetdict = (
                            train_validation.train_test_split(
                                float(validation) / (1.0 - test_split),
                                float(train) / (1.0 - test_split),
                                shuffle=False,
                            )
                        )
                        train, validation = (
                            train_validation_datasetdict["train"],
                            train_validation_datasetdict["test"],
                        )
                    elif validation is not None:
                        validation_datasetdict = train_validation.train_test_split(
                            float(validation) / (1.0 - test_split), shuffle=False
                        )
                        validation = validation_datasetdict["test"]
                    elif train is not None:
                        train_datasetdict = train_validation.train_test_split(
                            train_size=float(train) / (1.0 - test_split), shuffle=False
                        )
                        train = train_datasetdict["train"]
                    dataset_dict = {}
                    if train is not None:
                        dataset_dict["train"] = train
                    if validation is not None:
                        dataset_dict["validation"] = validation
                    if test is not None:
                        dataset_dict["test"] = test
                    if len(dataset_dict) > 1:
                        # create DatasetDict object
                        dataset.data = DatasetDict(dataset_dict)
                        dataset.split = None
                    else:
                        # there is only one dataset in the dict, use it
                        dataset.split, dataset.data = list(dataset_dict.items())[0]
                # choose split
                if split is not None:
                    assert isinstance(
                        dataset.data, DatasetDict
                    ), "dataset is not a DatasetDict, cannot specify split."
                    if split not in dataset.data.keys():
                        raise ValueError(
                            f"Split '{split}' does not exist for dataset '{dataset.path}'. Available splits are '{', '.join(dataset.data.keys())}'."
                        )
                    dataset.data = dataset.data[split]
                    dataset.split = split
                if shuffle_seed is not None:
                    # most of the time shuffling doesn't hurt and is needed for low-resource experiments where we pick n samples randomly
                    # can be disabled by setting `shuffle_seed` to `None`
                    dataset.data = dataset.data.shuffle(shuffle_seed)
                if slice_ is not None:
                    dataset.data = Dataset.from_dict(dataset.data[parse_slice(slice_)])
                datasets.append(dataset)
        else:  # path_or_name in list_datasets():
            # name appears in datasets library
            datasets_loaded = load_hf_dataset(
                path_or_name,
                name=config,
                split=split,
                cache_dir=cache_dir,
                keep_in_memory=keep_in_memory,
            )
            for dataset in datasets_loaded:
                if shuffle_seed is not None:
                    # most of the time shuffling doesn't hurt and is needed for low-resource experiments where we pick n samples randomly
                    # one can disable it by setting `shuffle_seed` to `None`
                    dataset.data = dataset.data.shuffle(shuffle_seed)
                if slice_ is not None:
                    dataset.data = Dataset.from_dict(dataset.data[parse_slice(slice_)])
                datasets.append(dataset)
        # else:
        #     raise ValueError(f"{path_or_name} is not a valid dataset name.")
        # print(datasets[-1])
        # print(datasets[-1][0])
        # input()

    # recreate index
    if recreate_index:
        for dataset in datasets:
            logging.warning("Recreating index, be sure you know what you do!")
            dataset.data = process_hf_dataset(
                dataset.data,
                lambda _, idx: {"id": idx},
                force_preprocess=force_preprocess,
                keep_in_memory=keep_in_memory,
                num_processes=num_processes,
                with_indices=True,
            )

    # enforce unique ids per dataset
    for dataset in datasets:
        assert len(dataset.data.unique("id")) == len(
            dataset.data
        ), f"IDs are not unique in dataset {dataset}!"

    # unpack samples if necessary
    if any("questions" in dataset.data.column_names for dataset in datasets):
        assert (
            unpack_fn is not None
        ), "Please provide a filter mechanism since data is packed"
    for dataset in datasets:
        if "questions" in dataset.data.column_names:
            dataset.data = unpack_fn(dataset.data)

    if concatenate:
        # concatenate datasets and remove config names
        datasets = [dataset.data for dataset in datasets]
        if len(datasets) == 1:
            # we don't have to concatenate datasets in this case but can return the only one
            return datasets[0]

        # keep only columns all datasets have in common
        feature_set = set(datasets[0].column_names)
        for dataset in datasets:
            feature_set &= set(dataset.column_names)
        logging.info(
            f"Keeping only columns {', '.join(feature_set)} for concatenation of datasets."
        )

        for i in range(len(datasets)):
            datasets[i] = datasets[i].remove_columns(
                set(datasets[i].column_names) - feature_set
            )
            datasets[i].reset_format()

        # try converting data into same features by re-creating datasets from dicts (sometimes casting doesn't work)
        features = datasets[0].features
        datasets_new = [datasets[0]] if len(datasets[0]) > 0 else []
        for i in range(1, len(datasets)):
            if len(datasets[i]) > 0:
                # the from_dict mehtod will fail if the features don't match (what is ok since one shouldn't concatenate in this case)
                datasets_new.append(
                    Dataset.from_dict(
                        dicts_to_feature_dict((sample for sample in datasets[i])),
                        features=features,
                    )
                )
        # cast to same features
        feats = datasets_new[0].features
        datasets_new = [dataset.cast(feats) for dataset in datasets_new]
        return concatenate_datasets(datasets_new)
    return datasets


def clean_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str, append_str: str = None):
    text = clean_whitespace(
        re.sub(rf"(/?\w+>|/\w*|<\w+)|[{string.punctuation}]|DOC|PAR|TLE", " ", text)
    )
    if append_str is not None and text:
        # only add '?' if string is not empty otherwise it would never be discared in filtering
        text += append_str
    return text


if __name__ == "__main__":
    # print memory usage
    print_mem_usage()

    data = get_datasets("techqa-rc:train", "~/arbeitsdaten/cache")
    print("Num samples:", len(data))

    # print memory usage
    print_mem_usage()

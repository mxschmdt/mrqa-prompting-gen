import configparser
import logging
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import replace
from glob import glob
from operator import itemgetter
from typing import Dict, Iterable, Optional, Union

# some parameters might be in ini file
config = configparser.ConfigParser()
config.read("config.ini")
if config.has_option("Paths", "cache_dir"):
    CACHE_DIR = config.get("Paths", "cache_dir")
    if not CACHE_DIR:
        CACHE_DIR = None
    else:
        CACHE_DIR = os.path.abspath(os.path.expanduser(CACHE_DIR))
        os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
        os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
        # storage for dynamic modules
        os.environ["HF_MODULE_CACHE"] = CACHE_DIR
else:
    CACHE_DIR = None


# we have to import comet_ml and wandb before PyTorch if used
try:
    import comet_ml
except ModuleNotFoundError:
    pass
try:
    import wandb
except ModuleNotFoundError:
    pass

from data.utils.utils import monkeypatch, select_unique

monkeypatch()

import stanza
import torch
from datasets import Dataset

nlp = stanza.Pipeline(lang="en", processors="tokenize,ner", verbose=False)

from openprompt import PromptDataLoader
from openprompt.data_utils import InputExample
from openprompt.plms import _MODEL_CLASSES, ModelClass, get_model_class, load_plm
from openprompt.prompts import (
    MixedTemplate,
    PrefixTuningTemplate,
    PTRTemplate,
    PtuningTemplate,
)
from tqdm import tqdm
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    BartTokenizerFast,
    PreTrainedTokenizer,
    TrainerCallback,
)

from data.qg.data import HeuristicFilter
from data.qg.evaluation import QGEvaluator
from data.rc.evaluation import (
    HfRCGenEvaluator,
    compute_f1_em,
    match_prediction_with_context,
)
from data.utils.data import (
    chunk_context,
    clean_text,
    expand_answers,
    get_datasets,
    normalize_answer,
    rectify_sample,
    truncate_context,
)
from data.utils.utils import (
    HfArgumentParser,
    PyLoggerCallback,
    check_positive,
    get_best_std,
    init_random,
    setup,
)
from utils.args import PromptingDataArguments as DataArguments
from utils.args import PromptingModelArguments as ModelArguments
from utils.args import PromptingTrainingArguments as TrainingArguments
from utils.args import finish_experiment, log_to_experiment
from utils.data import POSTPROCESSOR_CLASSES
from utils.model import PromptForGeneration
from utils.tokenizer import BartTokenizerWrapper
from utils.trainer import PromptTrainer
from utils.utils import fix_openprompt_tokenizers, safe_input

fix_openprompt_tokenizers()


_MODEL_CLASS_ALIASES = {
    "gerpt2": "gpt2",
    "aragpt2": "gpt2",
}

_MODEL_CLASS_MAPPING = {
    "dbmdz/german-gpt2": "gpt2",
    "l3cube-pune/hing-gpt-devanagari": "gpt2",
}

_PROMPT_CLASS_MAPPING = {
    "soft": MixedTemplate,
    "prefix": PrefixTuningTemplate,
    "ptuning": PtuningTemplate,
    "ptr": PTRTemplate,
}

_MODEL_CLASSES["bart"] = ModelClass(
    **{
        "config": BartConfig,
        "tokenizer": BartTokenizer,
        "model": BartForConditionalGeneration,
        "wrapper": BartTokenizerWrapper,
    }
)


def infer_model_class(model):
    # some do not follow the naming conventions
    if model in _MODEL_CLASS_MAPPING:
        return _MODEL_CLASS_MAPPING[model]
    model_class = model.split("/")[-1].split("-")[0]
    # check aliases
    if model_class in _MODEL_CLASS_ALIASES:
        return _MODEL_CLASS_ALIASES[model_class]
    return model_class


def count_params(model: torch.nn.Module):
    num_params = 0
    num_params_trainable = 0
    for param in model.parameters():
        num_params += param.numel()
        if param.requires_grad:
            num_params_trainable += param.numel()
    return num_params, num_params_trainable


TRANSFORMER_MODELS = {
    "t5-v1-1-large-rss": "tau/t5-v1_1-large-rss",
    "t5-v1-1-large": "google/t5-v1_1-large",
    "t5-v1-1-xl": "google/t5-v1_1-xl",
    "t5-v1-1-xxl": "google/t5-v1_1-xxl",
    "mt5-large": "google/mt5-large",
    "bart-large": "facebook/bart-large",
}


class EarlyStoppingCallback(TrainerCallback):
    "A callback that stops training after a specified amount of training steps"

    def __init__(self, num_train_steps: int, *args, **kwargs) -> None:
        super().__init__()
        self.num_train_steps = num_train_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.num_train_steps:
            control.should_training_stop = True
            control.should_evaluate = True
            control.should_log = True
            control.should_save = True
            return control


def main(
    args,
    train_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments,
):
    # (shallow) copy arguments so that we do not modify them for future runs
    train_args = replace(train_args)
    data_args = replace(data_args)
    model_args = replace(model_args)

    # conditional requirement checks
    if train_args.do_train and args.train_datasets is None:
        raise ValueError(
            "You have to specify a training dataset using --train-dataset for performing training"
        )
    if train_args.do_predict and args.predict_dataset is None:
        raise ValueError(
            "You have to specify a prediction dataset using --predict-dataset for performing prediction"
        )
    if (
        train_args.resume_from_checkpoint is None
        and train_args.load_checkpoint is None
        and args.template_idx is None
    ):
        raise ValueError(
            "Template index cannot be None, please specify one using --template-idx"
        )
    if train_args.do_predict and not args.dry_run and args.predict_dir is None:
        raise ValueError(
            "Prediction directory cannot be None, please specify one using --predict-dir"
        )
    if args.task == "qg" and train_args.do_predict:
        if args.answer_sampler is None:
            raise ValueError(
                "You have to specify an answer sampler using --answer-sampler"
            )

    # log parameters
    logging.info(f"Task: {args.task}")
    if args.task == "qg":
        logging.info(
            f"Answer sampling strategy: {args.answer_sampler}{(' (num max answers sampled: %s)' % (args.num_answers if args.num_answers is not None else 'all')) if args.answer_sampler != 'original' else ''}"
        )
    logging.info(
        f"Running on GPU: {not train_args.no_cuda and torch.cuda.is_available()}"
    )
    logging.info(f"Training: {train_args.do_train}")
    logging.info(f"Evaluation: {train_args.do_eval}")
    logging.info(f"Prediction: {train_args.do_predict}")
    logging.info(f"Training Arguments: {train_args.to_dict()}")

    # for convenience we allow some abbreviations
    if model_args.transformer in TRANSFORMER_MODELS:
        model_args.transformer = TRANSFORMER_MODELS[model_args.transformer]

    model_class_name = infer_model_class(model_args.transformer)
    if model_class_name not in ["t5", "gpt2"]:
        logging.warning(
            f"Model {model_class_name} hasn't been tested with Prompting and results may be incorrect"
        )
    model, tokenizer, model_config, wrapper_class = load_plm(
        model_class_name, model_args.transformer
    )
    model_class = get_model_class(plm_type=model_class_name)
    # tokenizer used for OpenPrompt is a slow tokenizer, but in some places we need a fast tokenizer
    if model_class_name == "bart":
        # somehow use_fast=True does not work with the general BartTokenizer class therefore we're instantiating the correct class manually
        fast_tokenizer = BartTokenizerFast.from_pretrained(model_args.transformer)
    else:
        fast_tokenizer = model_class.tokenizer.from_pretrained(
            model_args.transformer, use_fast=True
        )

    # fix mt5 tokenizer sequence length
    if model_class_name == "mt5":
        tokenizer.model_max_length = 1024
        fast_tokenizer.model_max_length = 1024

    # set up evaluation
    logger = PyLoggerCallback()
    evaluation_fn = lambda dataset, *_args, dataset_map_fn=None: (
        HfRCGenEvaluator(
            (
                dataset
                if isinstance(dataset, Dataset)
                else (
                    list(map(dataset_map_fn, dataset))
                    if dataset_map_fn is not None
                    else Dataset.from_list([sample.meta for sample in dataset])
                )
            ),
            *_args,
            tokenizer=tokenizer,
            no_answer_option=not data_args.disable_no_answer,
            match_answers_with_context=args.match_answers,
        )
        if args.task == "rc"
        else QGEvaluator(
            [sample.meta["question"] for sample in dataset],
            *_args,
            tokenizer=tokenizer,
            metrics=["bleu", "meteor", "rouge"],
        )
    )

    # load best checkpoint
    if train_args.load_best_checkpoint:
        checkpoint = (
            train_args.resume_from_checkpoint
            if train_args.resume_from_checkpoint is not None
            else train_args.load_checkpoint
        )
        assert (
            checkpoint is not None
        ), "You have to specify a checkpoint for loading the best checkpoint, use --resume_from_checkpoint or --load_checkpoint"
        # detect best checkpoint, i.e., the one with least steps (assuming that only best and most recent model are saved)
        checkpoints = glob(os.path.join(checkpoint, "checkpoint-*"))
        best_checkpoint_step = min(
            int(re.search(r"checkpoint-(\d+)", checkpoint).group(1))
            for checkpoint in checkpoints
        )
        checkpoint = os.path.join(checkpoint, f"checkpoint-{best_checkpoint_step}")
        if train_args.resume_from_checkpoint is not None:
            train_args.resume_from_checkpoint = checkpoint
        else:
            train_args.load_checkpoint = checkpoint

    # set up template
    if (
        train_args.resume_from_checkpoint is not None
        or train_args.load_checkpoint is not None
    ):
        # in this case we will load template from checkpoint
        template = PromptTrainer._load_template_from_checkpoint(
            train_args.resume_from_checkpoint
            if train_args.resume_from_checkpoint is not None
            else train_args.load_checkpoint
        )
    else:
        if args.prompt_method not in _PROMPT_CLASS_MAPPING:
            raise ValueError(f"{args.prompt_method} is not a valid prompt method")
        prompt_class = _PROMPT_CLASS_MAPPING[args.prompt_method]
        template = prompt_class(model, tokenizer).from_file(
            f"templates_{args.task}_{args.lang}.txt", args.template_idx
        )
    logging.info(f"Template: {template.text}")

    prompt_sample_fn = lambda sample, with_label: InputExample(
        meta=rectify_sample(sample, disable=True),
        tgt_text=(
            (sample["answers"]["text"][0] if args.task == "rc" else sample["question"])
            if with_label
            else None
        ),
    )

    def preprocess_dataset(
        dataset,
        separate_answers: bool = False,
        context_stride: Optional[int] = None,
        context_size: Optional[int] = None,
        chunking_mode: Union[bool, str, None] = False,
        truncation_mode: Union[bool, str, None] = False,
        consider_question: Optional[bool] = None,
        consider_answer: Optional[bool] = None,
        remove_instance_if_context_does_not_contain_answer: Optional[bool] = None,
        additional_input_length: int = 0,
        verify_input_seq_length: bool = False,
        override_model_max_input_seq_len: bool = False,
    ):
        if dataset is None:
            return None

        dataset = expand_answers(
            dataset,
            separate_answers=separate_answers,
            keep_in_memory=data_args.keep_in_memory,
            num_processes=data_args.num_workers,
        )

        # compute template length, -2 accounts for context and question and +1 accounts for eos token
        template_len = (
            len(
                template.wrap_one_example(
                    prompt_sample_fn(
                        dict(context="", question="", answers=dict(text=[""])), False
                    )
                )[0]
            )
            - 2
            + 1
        )

        if context_size == -1:
            max_question_len = max(
                len(tokenizer.tokenize(sample["question"])) for sample in dataset
            )
            context_size = tokenizer.model_max_length - template_len - max_question_len
            # input(max_question_len)
            # input(context_size)

        # chunk contexts
        if chunking_mode is not None and (
            not isinstance(chunking_mode, bool) or chunking_mode
        ):
            dataset = chunk_context(
                dataset,
                mode=chunking_mode,
                context_stride=context_stride,
                context_size=context_size,
                remove_instance_if_context_does_not_contain_answer=remove_instance_if_context_does_not_contain_answer,
                tokenizer=fast_tokenizer,
                force_preprocess=data_args.preprocess,
                keep_in_memory=data_args.keep_in_memory,
                num_processes=data_args.num_workers,
            )

        # truncate contexts
        if truncation_mode is not None and (
            not isinstance(truncation_mode, bool) or truncation_mode
        ):
            dataset = truncate_context(
                dataset,
                tokenizer=fast_tokenizer,
                max_length=tokenizer.model_max_length,
                truncate_sentences_only=(truncation_mode == "sentences"),
                additional_length=additional_input_length + template_len,
                consider_question=consider_question,
                consider_answer=consider_answer,
                remove_instance_if_context_does_not_contain_answer=remove_instance_if_context_does_not_contain_answer,
                force_preprocess=data_args.preprocess,
                keep_in_memory=data_args.keep_in_memory,
                num_processes=data_args.num_workers,
            )

        # check max length
        if verify_input_seq_length or override_model_max_input_seq_len:
            logging.info(
                "Verifying that input lengths do not exceed model maximum input length"
            )
            dataloader = PromptDataLoader(
                dataset=[
                    prompt_sample_fn(
                        dict(context="", question="", answers=dict(text=[""])), False
                    )
                ],
                template=template,
                tokenizer=tokenizer,
                tokenizer_wrapper_class=wrapper_class,
                max_seq_length=tokenizer.model_max_length,
                decoder_max_length=tokenizer.model_max_length,
                batch_size=1,
                shuffle=False,
                teacher_forcing=False,
                predict_eos_token=True,
                truncate_method="tail",
            )
            dataloader.tokenizer_wrapper.padding = lambda *arg, **kwargs: (
                kwargs["input_dict"] if "input_dict" in kwargs else arg[0]
            )
            max_input_length = max(
                len(
                    dataloader.tokenizer_wrapper.tokenize_one_example(
                        template.wrap_one_example(prompt_sample_fn(sample, False)),
                        False,
                    )["input_ids"]
                )
                for sample in dataset
            )

            # adapt max model input sequence length as needed
            if (
                override_model_max_input_seq_len
                and max_input_length > tokenizer.model_max_length
            ):
                logging.info(
                    f"Maximum input sequence length ({max_input_length}) exceeds model maximum input sequence length ({tokenizer.model_max_length}), increasing model maxium input sequence length to {max_input_length}"
                )
                tokenizer.model_max_length = max_input_length
                fast_tokenizer.model_max_length = max_input_length

            assert (
                max_input_length <= tokenizer.model_max_length
            ), f"Maximum input sequence length {max_input_length} exceeds model maximum of {tokenizer.model_max_length}; either activate truncation or chunking"
            logging.info(f"Maximum input sequence length is {max_input_length}")

        return dataset

    def prepare_dataset_for_model(dataset: Iterable, with_label: bool):
        if dataset is None:
            return None
        return [prompt_sample_fn(sample, with_label) for sample in dataset]

    ## set up datasets

    train_dataset = get_datasets(
        args.train_datasets,
        concatenate=True,
        keep_in_memory=data_args.keep_in_memory,
        unpack_fn=None,
    )
    train_dataset = preprocess_dataset(
        train_dataset,
        chunking_mode=data_args.train_chunking_mode,
        context_stride=data_args.train_context_stride,
        context_size=data_args.train_context_size,
        truncation_mode=data_args.train_truncate_contexts,
        remove_instance_if_context_does_not_contain_answer=True,
        consider_question=(args.task == "rc"),
        consider_answer=(args.task == "qg"),
        verify_input_seq_length=True,
        override_model_max_input_seq_len=data_args.train_chunking_mode is not None
        and (
            not isinstance(data_args.train_chunking_mode, bool)
            or data_args.train_chunking_mode
        ),
    )
    train_dataset = prepare_dataset_for_model(train_dataset, with_label=True)

    eval_dataset = get_datasets(
        args.eval_dataset,
        concatenate=False,
        keep_in_memory=data_args.keep_in_memory,
        unpack_fn=None,
    )
    eval_dataset = preprocess_dataset(
        eval_dataset,
        chunking_mode=data_args.eval_chunking_mode,
        context_stride=data_args.eval_context_stride,
        context_size=data_args.eval_context_size,
        truncation_mode=data_args.eval_truncate_contexts,
        remove_instance_if_context_does_not_contain_answer=(args.task == "qg"),
        consider_question=(args.task == "rc"),
        consider_answer=(args.task == "qg"),
        verify_input_seq_length=True,
        override_model_max_input_seq_len=data_args.eval_chunking_mode is not None
        and (
            not isinstance(data_args.eval_chunking_mode, bool)
            or data_args.eval_chunking_mode
        ),
    )
    eval_dataset = prepare_dataset_for_model(eval_dataset, with_label=True)

    additional_eval_datasets = get_datasets(
        args.add_eval_datasets,
        concatenate=False,
        keep_in_memory=data_args.keep_in_memory,
        unpack_fn=None,
    )
    if additional_eval_datasets:
        for _dataset in additional_eval_datasets:
            _dataset.data = preprocess_dataset(
                _dataset.data,
                truncation_mode=True,
                remove_instance_if_context_does_not_contain_answer=(args.task == "qg"),
                consider_question=(args.task == "rc"),
                consider_answer=(args.task == "qg"),
                verify_input_seq_length=True,
                override_model_max_input_seq_len=data_args.eval_chunking_mode
                is not None
                and (
                    not isinstance(data_args.eval_chunking_mode, bool)
                    or data_args.eval_chunking_mode
                ),
            )
            _dataset.data = prepare_dataset_for_model(_dataset.data, with_label=True)

    # freezing the plm will also freeze the raw_embedding of the template
    prompt_model = PromptForGeneration(
        plm=model,
        template=template,
        freeze_plm=not args.ft_model,
        plm_eval_mode=not args.ft_model,
        tokenizer=tokenizer,
    )
    generate_postprocess_class = POSTPROCESSOR_CLASSES.get(model_class_name, None)
    callbacks = [logger]

    # early stopping & compute training steps
    if args.early_stopping is not None:
        callbacks.append(EarlyStoppingCallback(args.early_stopping))
        train_args.max_steps = args.early_stopping
    elif train_args.min_steps is not None:
        if train_args.max_steps > 0:
            # assuming that dataset is infinite
            if train_args.min_steps < train_args.max_steps:
                logging.info(
                    f"Setting training steps to {train_args.min_steps} since minimum ({train_args.min_steps}) is lower than current ({train_args.max_steps})"
                )
                train_args.max_steps = train_args.min_steps
        else:
            # NOTE we do not take distributed computation into account
            num_train_steps = (
                len(train_dataset)
                // (
                    train_args.per_device_train_batch_size
                    * train_args.gradient_accumulation_steps
                )
            ) * train_args.num_train_epochs
            if train_args.min_steps > num_train_steps:
                logging.info(
                    f"Setting training steps to {train_args.min_steps} since minimum ({train_args.min_steps}) is lower than current ({num_train_steps})"
                )
                train_args.max_steps = train_args.min_steps

    trainer = PromptTrainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if train_args.do_eval else None,
        add_eval_datasets=additional_eval_datasets if train_args.do_eval else None,
        wrapper_class=wrapper_class,
        generate_postprocess_fn=(
            generate_postprocess_class(tokenizer)
            if generate_postprocess_class is not None
            else None
        ),
        model=prompt_model,
        args=train_args,
        data_collator=None,
        tokenizer=tokenizer,
        compute_metrics=(
            evaluation_fn(eval_dataset)
            if evaluation_fn is not None and eval_dataset
            else None
        ),
        evaluator_fn=evaluation_fn,
        callbacks=callbacks,
    )

    if (
        train_args.load_checkpoint is not None
        or train_args.resume_from_checkpoint is not None
    ):
        # load checkpoint
        trainer._load_from_checkpoint(
            train_args.resume_from_checkpoint
            if train_args.resume_from_checkpoint is not None
            else train_args.load_checkpoint
        )

    # these values include the template
    num_params_total, num_params_total_trainable = count_params(prompt_model)
    # these values do not include the template but only the transformer
    # we subtract the embedding size once since the MixedTemplate adds one token embedding for padding (signaling no soft token hence this is never used at all)
    embedding_size = trainer.model.prompt_model.template.raw_embedding.embedding_dim
    num_params_total -= embedding_size
    num_params_total_trainable -= embedding_size
    # sanity check
    assert num_params_total_trainable >= 0
    logging.info(
        f"Number of parameters: {num_params_total} (of which {num_params_total_trainable} (≈{num_params_total_trainable/num_params_total*100:.2f}%) are trainable)"
    )
    if train_args.do_train and num_params_total_trainable == 0:
        train_args.do_train = False
        logging.warning("Model has no trainable parameters, disabling training.")

    # evaluate first (unless if we do not train or explicitly ask for it)
    if train_args.do_eval:  # and not train_args.do_train:
        logging.info(trainer.evaluate(metric_key_prefix="eval"))
    if train_args.do_train:
        trainer.train(train_args.resume_from_checkpoint)
        # make sure that model is saved in the end
        trainer._save_checkpoint(None, None)

    # log infos to experiment tracker
    log_to_experiment(
        {
            "Template": template.text,
            "Num parameters": num_params_total,
            "Num trainable parameters": num_params_total_trainable,
        }
    )

    if train_args.do_predict:
        predict_dataset = get_datasets(
            args.predict_dataset,
            concatenate=True,
            keep_in_memory=data_args.keep_in_memory,
            unpack_fn=None,
        )
        len_predict_dataset = len(predict_dataset)
        if args.predict_exclude_dataset is not None:
            assert (
                args.predict_exclude_dataset_columns is not None
            ), "You have to specify --predict-exclude-dataset-columns if you want to exclude data from prediction"
            # exclude samples from  prediction as specified
            predict_exclude_dataset: Dataset = get_datasets(
                args.predict_exclude_dataset,
                concatenate=True,
                keep_in_memory=data_args.keep_in_memory,
                unpack_fn=None,
            )
            for column_to_exclude in args.predict_exclude_dataset_columns:
                # filter fo specific column
                values_to_exclude_for_column = predict_exclude_dataset.unique(
                    column_to_exclude
                )
                predict_dataset = predict_dataset.filter(
                    lambda sample: sample[column_to_exclude]
                    not in values_to_exclude_for_column,
                    num_proc=data_args.num_workers,
                )
            logging.info(
                f"{len(predict_dataset)} from {len_predict_dataset} samples remain after excluding samples for prediction"
            )

        # to be on the safe side, we remove the answer or question, respectively
        if args.task == "rc":
            predict_dataset = predict_dataset.remove_columns("answers")
        elif args.task == "qg":
            predict_dataset = predict_dataset.remove_columns("question")

        if args.task == "qg":
            if args.answer_sampler == "ner":

                def get_context_and_answer(
                    samples: Dict,
                    nlp: stanza.Pipeline,
                    num_answer_candidates_max: Optional[int],
                    entity_types: Optional[list] = None,
                    max_answer_length: Optional[int] = None,
                    tokenizer: Optional[PreTrainedTokenizer] = None,
                ):
                    augmented_samples = defaultdict(list)
                    for idx in range(len(samples["id"])):
                        id_ = samples["id"][idx]
                        context = samples["context"][idx]

                        doc = nlp(context)

                        # these entities have char offsets based on the whole document
                        entities = doc.ents

                        # exclude some domain-specific entities
                        entities = [
                            ent
                            for ent in entities
                            if not any(
                                word in ent.text for word in ["DOC", "PAR", "TLE"]
                            )
                        ]

                        if entity_types is not None:
                            # restrict entity types
                            entities = [
                                ent for ent in entities if ent.type in entity_types
                            ]

                        # remove duplicates
                        entities = list({ent.text: ent for ent in entities}.values())

                        if max_answer_length is not None:
                            entities = [
                                ent
                                for ent in entities
                                if len(tokenizer.tokenize(ent.text))
                                <= max_answer_length
                            ]

                        # num_answer_candidates might be zero hence no sample will be added
                        answers = entities
                        if num_answer_candidates_max is not None:
                            # sample at most `num_answer_candidates_max` answers
                            num_answer_candidates = min(
                                len(answers), num_answer_candidates_max
                            )
                            answers = random.sample(answers, num_answer_candidates)
                        num_answers = len(answers)

                        # set context and ids
                        # we omit the question question signaling that question has yet to be generated
                        augmented_samples["context"].extend([context] * num_answers)
                        augmented_samples["id"].extend(
                            (f"{id_}_ner#{i}") for i in range(num_answers)
                        )
                        # account for context offset in answer start char
                        augmented_samples["answers"].extend(
                            {"text": [answer.text], "answer_start": [answer.start_char]}
                            for answer in answers
                        )

                        # keep original contexts if available
                        if "context_original" in samples:
                            augmented_samples["context_original"].extend(
                                [samples["context_original"][idx]] * num_answers
                            )
                            augmented_samples["offset_context_original"].extend(
                                [samples["offset_context_original"][idx]] * num_answers
                            )
                    return augmented_samples

                # select unique contexts since we only take the context into account for sampling answers and generating questions
                predict_dataset = select_unique(predict_dataset, "context")
                logging.info(
                    f"{len(predict_dataset)} unique contexts are used for answer sampling"
                )

                # unpack data
                predict_dataset = preprocess_dataset(
                    predict_dataset,
                    separate_answers=False,
                    chunking_mode=False,
                    truncation_mode=False,
                )

                # select answer candidates
                logging.info(
                    f"Sampling answers using NER on {len(predict_dataset)} contexts"
                )
                predict_dataset = predict_dataset.map(
                    get_context_and_answer,
                    fn_kwargs=dict(
                        nlp=nlp,
                        num_answer_candidates_max=args.num_answers,
                        entity_types=args.entity_types,
                        max_answer_length=10,
                        tokenizer=tokenizer,
                    ),
                    num_proc=1,
                    batched=True,
                    remove_columns=predict_dataset.column_names,
                    desc="Sampling answers",
                )

                # subsample dataset to specified amount
                if len(predict_dataset) > 1000000:
                    logging.info(
                        f"Selecting 1000000 from {len(predict_dataset)} samples"
                    )
                    predict_dataset = predict_dataset.shuffle(seed=42).select(
                        range(1000000)
                    )

                logging.info(
                    f"{len(predict_dataset)} samples with sampled answers are used for QG prediction"
                )
            elif args.answer_sampler == "original":
                # only unpack data
                predict_dataset = preprocess_dataset(
                    predict_dataset,
                    separate_answers=True,
                    chunking_mode=False,
                    truncation_mode=False,
                )
            else:
                raise ValueError(
                    f"Unknown answer sampling strategy: {args.answer_sampler}"
                )

        # shard dataset
        if args.predict_num_shards is None:
            shards = [None]  # signals no sharding
        else:
            if args.predict_shard_indices:
                shards = args.predict_shard_indices
            else:
                shards = range(args.predict_num_shards)

        for shard in shards:
            if shard is None:
                # no sharding -> shard is just a reference to full dataset
                predict_dataset_shard = predict_dataset
            else:
                # get correct shard
                predict_dataset_shard = predict_dataset.shard(
                    args.predict_num_shards, shard, contiguous=True
                )
                logging.info(
                    f"Processing shard index {shard} of {args.predict_num_shards} shards in total ({len(predict_dataset_shard)/len(predict_dataset)*100}%)"
                )

            # we manually create a id to index mapping so that we can directly access the dataset by id
            id_to_index_mapping = {
                sample_id: idx
                for idx, sample_id in enumerate(predict_dataset_shard["id"])
            }

            # preprocess data to create InputExamples for the model (features will be created on-the-fly in the dataloader)
            predict_dataset_shard_processed = preprocess_dataset(
                predict_dataset_shard,
                separate_answers=False,
                chunking_mode=data_args.predict_chunking_mode,
                context_stride=data_args.predict_context_stride,
                context_size=data_args.predict_context_size,
                truncation_mode=data_args.predict_truncate_contexts,
                remove_instance_if_context_does_not_contain_answer=(args.task == "qg"),
                consider_question=(args.task == "rc"),
                consider_answer=(args.task == "qg"),
                verify_input_seq_length=True,
                override_model_max_input_seq_len=data_args.predict_chunking_mode
                is not None
                and (
                    not isinstance(data_args.predict_chunking_mode, bool)
                    or data_args.predict_chunking_mode
                ),
            )
            predict_dataset_shard_processed = prepare_dataset_for_model(
                predict_dataset_shard_processed, with_label=False
            )

            # remove the compute_metrics fn since we do not have labels for prediction
            # this only works because we do not perform evaluation on the eval dataset afterwards
            trainer.compute_metrics = (
                evaluation_fn(predict_dataset_shard_processed)
                if "question" in predict_dataset_shard_processed[0].meta
                else None
            )

            # run prediction, labels are just dummy values
            predictions, scores, metrics = trainer.predict(
                test_dataset=predict_dataset_shard_processed,
            )

            # aggregate predictions, using samples before preprocessing as reference as this will keep original contexts and answers
            predictions_aggregated = {}
            for prediction, score, sample in zip(
                predictions, scores, predict_dataset_shard_processed
            ):
                if (
                    sample.meta["id"] not in predictions_aggregated
                    or predictions_aggregated[sample.meta["id"]][2] < score
                ):
                    predictions_aggregated[sample.meta["id"]] = [
                        predict_dataset_shard[id_to_index_mapping[sample.meta["id"]]],
                        prediction,
                        score,
                    ]

            if args.task == "rc" and args.match_answers:
                predictions_rectified = match_prediction_with_context(
                    [sample["context"] for sample, _, _ in predictions_aggregated],
                    list(map(itemgetter(1), predictions_aggregated.values())),
                )

            # print info
            print(f"Dataset: {args.predict_dataset}")
            print(f"Template: {template.text}")
            print(f"Metrics: {metrics}")
            if args.verbose:
                if not args.no_shell:
                    safe_input()
                if args.task == "rc":
                    if args.match_answers:
                        for (sample, answer, score), answer_rectified in zip(
                            predictions_aggregated.values(), predictions_rectified
                        ):
                            answers = [
                                normalize_answer(answer)
                                for answer in sample["answers"]["text"]
                            ]
                            f1, em = compute_f1_em(normalize_answer(answer), answers)
                            f1_rectified, em_rectified = compute_f1_em(
                                normalize_answer(answer_rectified), answers
                            )
                            print("-------------------------")
                            print(f"Sample: {sample}")
                            print(f"Generated answer: '{answer}' (F1: {f1} - EM: {em})")
                            print(
                                f"Generated answer (rectified): '{answer_rectified}' (F1: {f1_rectified} - EM: {em_rectified})"
                            )
                            if not args.no_shell:
                                safe_input()
                    else:
                        for sample, answer, score in predictions_aggregated.values():
                            answers = [
                                normalize_answer(answer)
                                for answer in sample["answers"]["text"]
                            ]
                            f1, em = compute_f1_em(normalize_answer(answer), answers)
                            print("-------------------------")
                            print(f"Sample: {sample}")
                            print(f"Generated answer: '{answer}' (F1: {f1} - EM: {em})")
                            if not args.no_shell:
                                safe_input()
                elif args.task == "qg":
                    for sample, question, score in predictions_aggregated.values():
                        print("-------------------------")
                        print(f"Sample: {sample}")
                        print(f"Generated question: '{question}'")
                        print(
                            f"Generated question (cleaned): '{clean_text(question) + '?' if question[-1] == '?' else ''}'"
                        )
                        print(
                            f"Question kept (in heuristic filtering): {HeuristicFilter().filter_sample(question, sample['answers']['text'])}"
                        )
                        if not args.no_shell:
                            safe_input()

            if args.task == "qg":
                if args.clean_questions:
                    for prediction in tqdm(
                        predictions_aggregated.values(), desc="Cleaning questions"
                    ):
                        prediction[1] = clean_text(prediction[1])

            if not args.dry_run:
                keys = list(next(iter(predictions_aggregated.values()))[0].keys())
                columns = zip(
                    *(
                        [sample[key] for key in keys]
                        for sample, _, _ in predictions_aggregated.values()
                    )
                )
                dataset_dict = dict(zip(keys, list(columns)))

                if args.task == "rc":
                    # update answers
                    dataset_dict.update(
                        answers=[
                            dict(text=[prediction[1]])
                            for prediction in predictions_aggregated
                        ]
                    )
                    if args.match_answers:
                        dataset_dict_rectified = dict(
                            dataset_dict,
                            answers=[
                                dict(text=[prediction])
                                for prediction in predictions_rectified
                            ],
                        )
                elif args.task == "qg":
                    # update questions
                    dataset_dict.update(
                        question=list(
                            map(itemgetter(1), predictions_aggregated.values())
                        )
                    )

                # save dataset
                path = args.predict_dir
                if shard is not None:
                    path += f"_{shard}-{args.predict_num_shards}"
                dataset_generated = Dataset.from_dict(dataset_dict)
                dataset_generated.save_to_disk(path)
                logging.info(
                    f"Saved {len(dataset_generated)} samples with generated {'answers' if args.task == 'rc' else 'questions'} to '{path}'."
                )
                if args.task == "rc" and args.match_answers:
                    dataset_generated = Dataset.from_dict(dataset_dict_rectified)
                    path_rectified = path + "_rectified"
                    dataset_generated.save_to_disk(path_rectified)
                    logging.info(
                        f"Saved {len(dataset_generated)} samples with generated answers (rectified) to '{path_rectified}'."
                    )

                # save template
                with open(
                    os.path.join(path, "template.txt"),
                    "wt",
                    encoding="utf-8",
                    errors="replace",
                ) as f:
                    f.write(str(template.text))
                logging.info(f"Saved template to '{path}'.")

    # finish any running experiment logger
    finish_experiment()

    return logger.get_logs()


def _main(args=None):
    # print command
    print("python", " ".join(args if args is not None else sys.argv), flush=True)

    # fmt: off

    ### parent parsers
    # general parser
    parser = HfArgumentParser(description="A parser including general arguments", dataclass_types=[TrainingArguments, DataArguments, ModelArguments], add_help=True)
    parser.add_argument("task", type=str, help="Specifies the task to perform", choices=["rc", "qg"],)
    parser.add_argument("-pa", "--pre-allocation", action="store_true", help="Enable pre-allocation of GPU memory (this will allocate 95%% of memory)",)
    parser.add_argument("--lang", type=str, choices=["en", "de", "es", "hi", "vi", "zh", "ar"], required=True, help="The language for choosing the prompt template",)
    parser.add_argument("--template-idx", type=int, required=False, help='The template index (zero-based) from the file which will be loaded ("templates_{task}_{language}")',)
    parser.add_argument("--no-shell", action="store_true", help="Disable shell interaction by skipping input form user",)
    parser.add_argument("--prompt-method", type=str, default='soft', choices=['soft', 'prefix', 'p', 'ptr'], help="The Prompting method to use",)
    parser.add_argument("--ft-model", action="store_true", default=False, help="Will fine-tune the LM if True",)
    parser.add_argument("--early-stopping", type=int, help="Stop training after specified amount of training steps",)
    
    # training related arguments
    parser.add_argument("--runs", type=check_positive, default=1, help="The number of runs (usually with different seeds)",)
    parser.add_argument("--train-datasets", required=False, nargs="+", metavar="dataset", help="the dataset(s) used for training the model",)
    
    # evaluation related arguments
    parser.add_argument("--match-answers", action="store_true", help="Try matching generated answers with context (RC only)",)
    parser.add_argument("--eval-dataset", metavar="dataset", help="the dataset used for evaluating the model and selecting the best model checkpoint",)
    parser.add_argument("--add-eval-datasets", nargs="+", metavar="dataset", help="the additional dataset(s) used for evaluating the model",)

    # prediction related arguments
    parser.add_argument("--predict-dataset", metavar="dataset", help="the dataset used for prediction",)
    parser.add_argument("--predict-exclude-dataset", metavar="dataset", help="the dataset which is excluded from prediction, i.e. values in columns specified by --predict-exclude-dataset-columns are excluded from prediction",)
    parser.add_argument("--predict-exclude-dataset-columns", type=str, nargs='+', help="The columns which are used to determine the values to exclude from prediction",)
    parser.add_argument("--predict-num-shards", type=int, help="enables sharding for prediction with the number of shards specified",)
    parser.add_argument("--predict-shard-indices", type=int, nargs='+', help="the shard indices used for prediction if used, will default to all shards if empty",)
    parser.add_argument("-v", "--verbose", action="store_true", required=False, help="enable printing of generated sequences",)
    parser.add_argument("-i", "--interactive", action="store_true", help="enable interactive mode")
    parser.add_argument("--dry-run", action="store_true", required=False, help="do not store predictions on disk",)
    parser.add_argument("--predict-dir", type=os.path.expanduser, required=False, help="folder for storing predictions",)
    parser.add_argument("--num-answers", type=int, required=False, help="the number of answers sampled (if strategy is not original)",)
    parser.add_argument("--answer-sampler", choices=['original', 'ner'], required=False, help="the strategy to sample answers for QG",)
    parser.add_argument("--entity-types", nargs='+', choices=nlp.processors['ner'].get_known_tags(), required=False, help="restrict the entity types for ner based answer sampling for QG",)
    parser.add_argument("--clean-questions", action='store_true', help="Whether or not to clean questions (based on handcrafted rules) generated in QG Prediction",)

    # check for args file
    if len(sys.argv) > 1 and sys.argv[1].endswith('.args'):
        argfile = sys.argv[1]
        args = sys.argv[2:]
    else:
        argfile = None
        args = sys.argv[1:]
    train_args, data_args, model_args, args = parser.parse_args_into_dataclasses(args, args_filename=argfile)
    # do setup before any logging to make sure that no default handler is created
    setup(train_args.log_level, args.pre_allocation and not train_args.no_cuda)

    # do multiple runs
    seeds_from_arg = train_args.seed
    seeds = []
    metrics_all = defaultdict(list)
    for run in range(args.runs):
        logging.info("===== Run %d/%d =====", run + 1, args.runs)
        # set seed
        seed = init_random(
            seeds_from_arg[run]
            if seeds_from_arg is not None and len(seeds_from_arg) > run
            else None
        )
        seeds.append(seed)

        # call command-specific function
        train_args.seed = seed
        metrics = main(args, train_args, data_args, model_args)

        # aggregate metrics from multiple runs
        for key, value in metrics.items():
            metrics_all[key].append(value)

    if metrics_all and len(next(iter(metrics_all.values()))) > 1:
        # there is only one evaluation per run hence there is no best model checkpoint
        logging.info(
            f"Metrics from {args.runs} runs: {get_best_std(metrics_all, None)}"
        )


if __name__ == "__main__":
    _main()

import logging
import re
from typing import Dict, Iterable, List, Optional, Union

import torch
from datasets import Dataset
from openprompt.data_utils import InputExample
from openprompt.plms import get_model_class, load_plm
from tqdm import tqdm

from data.rc.evaluation import HfRCGenEvaluator, compute_f1_em
from data.utils.data import (
    chunk_context,
    clean_text,
    clean_whitespace,
    expand_answers,
    normalize_answer,
    rectify_sample,
    truncate_context,
)
from utils.args import PromptingTrainingArguments
from utils.data import POSTPROCESSOR_CLASSES
from utils.model import PromptForGeneration
from utils.trainer import PromptDataLoader, PromptTrainer


def d_clean_question(sample: Dict):
    """A function cleaning a question which can be used with `datasets.map`"""
    return {"question": clean_text(sample["question"])}


class Filter:
    name: str = None

    def filter_sample(self, context, question: str, answer: List[str], *args, **kwargs):
        raise NotImplementedError(
            "Function `filter_sample` is not implemented for class `Filter`. Please subclass and override."
        )

    def filter_dataset(self, dataset: Dataset, num_workers: int = None):
        """Filters a `datasets.Dataset`"""
        return dataset.filter(
            lambda sample: self.filter_sample(
                sample["question"], sample["answers"]["text"]
            ),
            batched=False,
            num_proc=num_workers,
            desc=f"Filtering questions{(' using ' + self.name) if self.name is not None else ''}",
        )

    def filter_iterable(self, predictions: Iterable, references: Iterable):
        references, predictions = zip(
            *[
                (sample, prediction)
                for sample, prediction in tqdm(
                    zip(references, predictions),
                    desc=f"Filtering questions{(' using ' + self.name) if self.name is not None else ''}",
                )
                if self.filter_sample(
                    prediction, sample.meta["answers"]["text"], clean=False
                )
            ]
        )
        return references, predictions


class HeuristicFilter(Filter):
    # applies heuristics to filter generated questions
    name: str = "heuristic"

    def filter_sample(self, question: str, answers: List[str], clean: bool = True):
        def is_any_answer_in_question(question, answers):
            question = question.lower()
            return any(clean_text(answer.lower()) in question for answer in answers)

        # question may not contain any answer
        if is_any_answer_in_question(question, answers):
            return False

        if clean:
            question = clean_text(question)
            # question may not contain any answer
            if is_any_answer_in_question(question, answers):
                return False

        # additional cleaning only for filtering
        question = question.lower()
        # remove question words
        question = clean_whitespace(
            re.sub(
                r"\b(why|when|where|how|did|do|have|is|not|in|the|a|and|or|question)\b",
                " ",
                question,
            )
        )

        # question may not contain any answer
        if is_any_answer_in_question(question, answers):
            return False

        # empty question (e.g. due to cleaning)
        if question == "":
            return False
        # question may not contain bad words
        if any(
            sequence in question
            for sequence in [
                "stretch",
                "brainer",
                "good one",
                "answer",
                "good choice",
                "good match",
                "simple",
                "logical",
                "easy",
                "simple",
                "good place",
                "guess",
                "complicated",
                "trivial",
                "obvious",
                "fit",
                "confusing",
                "joke",
                "contradiction",
                "difficult",
                "interesting",
                "<P",
                "</P",
                "P>",
                "correct",
                "incorrect",
                "important",
            ]
        ):
            return False
        # question may not contain any answer
        if is_any_answer_in_question(question, answers):
            return False

        return True


class ConsistencyFilter(Filter):
    # applies an RC model (currently a Prompting model) for filtering
    name: str = "consistency"

    def __init__(
        self, checkpoint_path: str, batch_size: int, num_workers: int = None
    ) -> None:
        model_class_name = "t5"
        model_name = "tau/t5-v1_1-large-rss"

        model, tokenizer, model_config, wrapper_class = load_plm(
            model_class_name, model_name
        )

        model_class = get_model_class(plm_type=model_class_name)
        # tokenizer used for OpenPrompt is a slow tokenizer, but in some places we need a fast tokenizer
        fast_tokenizer = model_class.tokenizer.from_pretrained(
            model_name, use_fast=True
        )

        # fix mt5 tokenizer sequence length
        if model_class_name == "mt5":
            tokenizer.model_max_length = 1024
            fast_tokenizer.model_max_length = 1024

        self.evaluation_fn = lambda dataset, *_args, dataset_map_fn=None: HfRCGenEvaluator(
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
            no_answer_option=False,
            # max_answer_length=data_args.max_answer_length,
            match_answers_with_context=False,
        )

        ## dataset preprocessing

        prompt_sample_fn = lambda sample, with_label: InputExample(
            meta=rectify_sample(sample, disable=True),
            tgt_text=sample["answers"]["text"][0] if with_label else None,
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
                num_processes=num_workers,
            )

            # compute template length, -2 accounts for context and question and +1 accounts for eos token
            template_len = (
                len(
                    template.wrap_one_example(
                        prompt_sample_fn(
                            dict(context="", question="", answers=dict(text=[""])),
                            False,
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
                context_size = (
                    tokenizer.model_max_length - template_len - max_question_len
                )

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
                    num_processes=num_workers,
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
                    num_processes=num_workers,
                )

            # check max length
            if verify_input_seq_length or override_model_max_input_seq_len:
                logging.info(
                    "Verifying that input lengths do not exceed model maximum input length"
                )
                dataloader = PromptDataLoader(
                    dataset=[
                        prompt_sample_fn(
                            dict(context="", question="", answers=dict(text=[""])),
                            False,
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

        def prepare_dataset_for_model(dataset, with_label: bool):
            if dataset is None:
                return None
            return [prompt_sample_fn(sample, with_label) for sample in dataset]

        self.preprocess_fn = lambda dataset: prepare_dataset_for_model(
            preprocess_dataset(
                dataset,
                chunking_mode="token",
                context_stride=100,
                context_size=450,
                truncation_mode=False,
                remove_instance_if_context_does_not_contain_answer=True,
                consider_question=True,
                consider_answer=False,
                verify_input_seq_length=True,
                override_model_max_input_seq_len=True,
            ),
            with_label=False,
        )

        ## model setup
        template = PromptTrainer._load_template_from_checkpoint(checkpoint_path)

        prompt_model = PromptForGeneration(
            plm=model,
            template=template,
            freeze_plm=True,
            plm_eval_mode=True,
            tokenizer=tokenizer,
        )
        generate_postprocess_class = POSTPROCESSOR_CLASSES.get(model_class_name, None)
        self.trainer = PromptTrainer(
            wrapper_class=wrapper_class,
            generate_postprocess_fn=(
                generate_postprocess_class(tokenizer)
                if generate_postprocess_class is not None
                else None
            ),
            model=prompt_model,
            args=PromptingTrainingArguments(
                output_dir="./tmp", seed=42, per_device_eval_batch_size=batch_size
            ),
            data_collator=None,
            tokenizer=tokenizer,
        )

        # load checkpoint
        self.trainer._load_from_checkpoint(checkpoint_path)

    def filter_sample(self, predictions, references):
        raise NotImplementedError()

    def filter_dataset(self, dataset: Dataset):
        """Uses a `Trainer` for filtering samples."""

        processed_dataset = self.preprocess_fn(dataset)

        self.trainer.compute_metrics = self.evaluation_fn(processed_dataset)

        # run prediction, labels are just dummy values
        predictions, scores, metrics = self.trainer.predict(
            test_dataset=processed_dataset,
        )

        # aggregate predictions
        predictions_aggregated = {}
        for prediction, score, sample in zip(predictions, scores, processed_dataset):
            if (
                sample.meta["id"] not in predictions_aggregated
                or predictions_aggregated[sample.meta["id"]][2] < score
            ):
                predictions_aggregated[sample.meta["id"]] = [sample, prediction, score]

        assert len(dataset) == len(predictions_aggregated)

        scores = []
        for reference in dataset:
            prediction = predictions_aggregated[reference["id"]][1]
            scores.append(
                compute_f1_em(
                    normalize_answer(prediction),
                    [
                        normalize_answer(_reference)
                        for _reference in reference["answers"]["text"]
                    ],
                )[0]
            )
        scores = torch.tensor(scores)
        # threshold is set to F1 of 80%
        mask = scores.ge(0.8)
        filtered_predictions = dataset.select(mask.nonzero())
        return filtered_predictions

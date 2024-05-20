import heapq
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import zip_longest
from typing import Dict, List, Tuple, Union

import numpy
import torch
from evaluate import load as load_metric
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import EvalPrediction, PreTrainedTokenizer

from ..utils.data import normalize_answer, get_per_sample_indices
from ..utils.utils import select_unique


class LossMetric:
    def __init__(self) -> None:
        # loss
        self.loss = 0.0
        self.total_loss = 0

    def __getitem__(self, key):
        return self.get()[key]

    def __repr__(self):
        return self.get()

    def __str__(self):
        return str(self.get())

    def add_loss(self, loss, num_samples):
        self.total_loss += num_samples
        self.loss += loss

    def get(self):
        return {}  # do not return loss as this is handled by transformers


class RCMetric(LossMetric):
    """A class for computing and recording metrics within one epoch."""

    def __init__(self):
        super().__init__()

        # score for no-answer labels
        self.total_no_answer = 0
        self.recall_no_answer = 0

        # score for answer labels
        self.total_answer = 0
        self.em_answer = 0
        self.f1_answer = 0.0

        # scores for best answer and top 5
        self.f1_ha_1 = 0.0
        self.f1_ha_5 = 0.0
        self.em_ha_1 = 0
        self.em_ha_5 = 0

    def get(self):
        return dict(
            super().get(),
            **{
                "recall_no_answer": (
                    100 * self.recall_no_answer / self.total_no_answer
                    if self.total_no_answer > 0
                    else 0.0
                ),
                "em_answer": (
                    100 * self.em_answer / self.total_answer
                    if self.total_answer > 0
                    else 0.0
                ),
                "f1_answer": (
                    100 * self.f1_answer / self.total_answer
                    if self.total_answer > 0
                    else 0.0
                ),
                "f1": (
                    100
                    * (self.recall_no_answer + self.f1_answer)
                    / (self.total_answer + self.total_no_answer)
                    if (self.total_answer + self.total_no_answer) > 0
                    else 0.0
                ),
                "f1_ha_1": (
                    100 * self.f1_ha_1 / self.total_answer
                    if self.total_answer > 0
                    else 0.0
                ),
                "f1_ha_5": (
                    100 * self.f1_ha_5 / self.total_answer
                    if self.total_answer > 0
                    else 0.0
                ),
                "em_ha_1": (
                    100 * self.em_ha_1 / self.total_answer
                    if self.total_answer > 0
                    else 0.0
                ),
                "em_ha_5": (
                    100 * self.em_ha_5 / self.total_answer
                    if self.total_answer > 0
                    else 0.0
                ),
            },
        )

    @staticmethod
    def em(prediction, truth):
        return int(prediction == truth)

    @staticmethod
    def f1(prediction, truth):
        if prediction is None:
            # cannot compute overlap if no-answer is predicted
            if truth is None:
                return 1.0
            else:
                return 0.0

        if (
            isinstance(truth, (tuple, list))
            and len(truth) == 2
            and isinstance(truth[0], int)
        ):
            # measure character span overlap
            num_prediction_items = prediction[1] - prediction[0]
            num_truth_items = truth[1] - truth[0]
            num_overlaps = max(
                0, min(prediction[1], truth[1]) - max(prediction[0], truth[0])
            )
        else:
            # measure whitespace-split token overlap
            prediction_items = Counter(prediction.split())
            num_prediction_items = sum(prediction_items.values())
            truth_items = Counter(truth.split())
            num_truth_items = sum(truth_items.values())
            num_overlaps = sum((prediction_items & truth_items).values())

        if num_overlaps > 0:
            # f1 score is bigger than 0
            precision = num_overlaps / num_prediction_items
            recall = num_overlaps / num_truth_items
            return (2 * precision * recall) / (precision + recall)
        else:
            return 0.0

    @staticmethod
    def max_over_ground_truths(metric_fn, prediction, truths):
        return max((metric_fn(prediction, truth) for truth in truths))

    def add_prediction(
        self,
        label: List[str],
        prediction,
        predicted_answers: List[Union[Tuple[int], str]],
    ):
        if label is None:
            # unanswerable sample
            self.total_no_answer += 1
            self.recall_no_answer += int(prediction is None)
        else:
            # answerable sample
            # NOTE prediction should include document ID for char spans to make sure that answer span
            # prediction is either None or an answer while predicted_answers always contains best spans
            self.total_answer += 1
            # exact match (EM) score
            f1, em = compute_f1_em(prediction, label)
            self.em_answer += em
            # F1 score
            self.f1_answer += f1

            # calculate auxiliary scores for answerable samples
            #   1. evaluation on best predicted answer for HA-F1@1
            #   2. evaluation on best 5 predicted answers for HA-F1@5
            # exact match (EM) score
            self.em_ha_1 += self.max_over_ground_truths(
                self.em, predicted_answers[0], label
            )
            self.em_ha_5 += max(
                self.max_over_ground_truths(self.em, predicted_answer, label)
                for predicted_answer in predicted_answers
            )
            # F1 score
            self.f1_ha_1 += self.max_over_ground_truths(
                self.f1, predicted_answers[0], label
            )
            self.f1_ha_5 += max(
                self.max_over_ground_truths(self.f1, predicted_answer, label)
                for predicted_answer in predicted_answers
            )


def compute_f1_em(prediction, label):
    return RCMetric.max_over_ground_truths(
        RCMetric.f1, prediction, label
    ), RCMetric.max_over_ground_truths(RCMetric.em, prediction, label)


@dataclass()
class RCPrediction:
    num_answers: int
    sample_id: str = None
    predictions: list = field(default_factory=list)
    no_answer_score: float = None
    contexts: Dict[str, list] = field(
        default_factory=dict
    )  # stores context per instance_id (document)

    def add(
        self,
        sample_id,
        instance_id=None,
        context=None,
        scores=None,
        spans=None,
        answers=None,
        no_answer_score=None,
    ) -> None:
        if sample_id is not None and self.sample_id != sample_id:
            assert self.sample_id is None, f"Cannot change sample id of {self}"
            self.sample_id = sample_id

        assert (
            (instance_id is None) == (scores is None) == (spans is None)
        ), "Scores and answers can only be given together"
        if scores is not None:
            if context is not None:
                # set context for instance_id
                self.contexts[instance_id] = context

            # push all predictions to heap until it's full
            # consider answer if available
            for score, span, answer in zip_longest(scores, spans, answers):
                if len(self.predictions) < self.num_answers:
                    heapq.heappush(self.predictions, (score, span, answer, instance_id))
                else:
                    if score.item() > self.predictions[0][0]:
                        # replace item in heap if new prediction is better
                        heapq.heapreplace(
                            self.predictions, (score, span, answer, instance_id)
                        )
        # only add better no-answer score
        if no_answer_score is not None:
            if self.no_answer_score is None or no_answer_score > self.no_answer_score:
                self.no_answer_score = no_answer_score

    def extract_answers(self, num_answers, force_extract: bool = False):
        scores, spans, answers, instance_ids = zip(
            *heapq.nlargest(num_answers, self.predictions)
        )
        return (
            scores,
            [
                (
                    self.contexts[instance_id][span[0] : span[1]]
                    if force_extract or answer is None
                    else answer
                )
                for answer, span, instance_id in zip(answers, spans, instance_ids)
            ],
            spans,
        )


def extract_span(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    length: int = None,
    token_char_mapping=None,
    context_offset=None,
    enforce_logits_positive: bool = False,
    topk: int = 1,
    extract_answer: bool = False,
    context: str = None,
    input_ids=None,
    tokenizer: PreTrainedTokenizer = None,
    raw_span: bool = False,
    max_length=None,
):
    # inputs are batch x sequence
    if length is None:
        length = start_logits.size(0)

    # consider all possible combinations (by addition of scores) of start and end token
    # vectorize by broadcasting start/end token probabilites to matrix and adding both
    # afterwards we can take the maximum of the upper half including the diagonal (end >= start) and limit the max length
    slice_relevant_tokens = slice(context_offset, length)
    len_relevant_tokens = length - context_offset

    start_score_matrix = (
        start_logits[slice_relevant_tokens].unsqueeze(1).expand(-1, len_relevant_tokens)
    )
    end_score_matrix = end_logits[slice_relevant_tokens].expand(
        len_relevant_tokens, len_relevant_tokens
    )  # new dimension is by default added to the front
    score_matrix = (
        start_score_matrix + end_score_matrix
    ).triu()  # return upper triangular part including diagonal, rest is 0
    if not enforce_logits_positive:
        # values can be lower than 0 -> make sure to set lower triangular matrix to very low value
        lower_triangular_matrix = torch.tril(
            torch.ones_like(score_matrix, dtype=torch.long), diagonal=-1
        )
        score_matrix.masked_fill_(
            lower_triangular_matrix, float("-inf")
        )  # make sure that lower triangular matrix is set -inf to ensure end >= start
    if max_length is not None and max_length > 0:
        # mask diagonal band therefore set upper triangular matrix to low value (lower triangular matrix is already not considered)
        upper_triangular_matrix = torch.ones_like(score_matrix, dtype=torch.long).triu(
            diagonal=max_length - 1
        )
        score_matrix.masked_fill_(
            upper_triangular_matrix,
            float("-inf") if not enforce_logits_positive else 0.0,
        )

    best_scores, indices = score_matrix.view(-1).topk(
        k=min(len_relevant_tokens, topk)
    )  # find top k combinations of start + end (with a maximum of len_relevant_tokens)
    spans_start = torch.div(indices, len_relevant_tokens, rounding_mode="floor")
    spans_end = indices % len_relevant_tokens
    # NOTE spans_start and spans_end can be used to index score_matrix: score_matrix[spans_start, spans_end]
    # shift span to context
    spans_start += context_offset
    spans_end += context_offset
    assert (
        spans_start <= spans_end
    ).all(), "Spans cannot end before start, there might be an issue with the implementation!"

    if raw_span:
        best_spans = [
            (span_start.item(), span_end.item())
            for span_start, span_end in list(zip(spans_start, spans_end))
        ]
    else:
        best_spans = list(
            zip(
                list(map(lambda x: token_char_mapping[x.item()][0], spans_start)),
                list(map(lambda x: token_char_mapping[x.item()][1], spans_end)),
            )
        )
    no_answer_scores = (
        start_logits[0] + end_logits[0]
    )  # CLS token as indicator for null answer
    # usually we do not need all answers directly therefore allow to choose
    if extract_answer:
        # extract answer
        if raw_span:
            return (
                best_scores,
                best_spans,
                no_answer_scores,
                [
                    tokenizer.decode(input_ids[_span_start:_span_end])
                    for _span_start, _span_end in best_spans
                ],
            )
        else:
            return (
                best_scores,
                best_spans,
                no_answer_scores,
                [
                    context[_span_start:_span_end]
                    for _span_start, _span_end in best_spans
                ],
            )
    else:
        return best_scores, best_spans, no_answer_scores


def rc_predict(
    logits,
    sample_ids,
    instance_ids,
    contexts,
    context_offsets,
    lengths,
    offset_mappings,
    num_answers,
    max_length=None,
):
    predictions = defaultdict(partial(RCPrediction, num_answers))
    # NOTE logits are padded with -100 by transformers library by default

    if isinstance(logits, numpy.ndarray):
        logits = torch.from_numpy(logits)

    with torch.no_grad():
        # split logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(dim=-1), end_logits.squeeze(
            dim=-1
        )

        # iterate over chunks grouped by instances
        # instance_ids = [extended_features.get_chunk(chunk_idx=chunk_id, return_instance=True)['instance_idx'] for chunk_id in chunk_ids]
        (per_instance_indices,) = get_per_sample_indices(
            instance_ids, list(range(len(instance_ids)))
        )
        for _indices in per_instance_indices:
            for _idx in _indices:
                sample_id = sample_ids[_idx]
                instance_id = instance_ids[_idx]
                # get best spans
                scores, spans, no_answer_score = extract_span(
                    start_logits[_idx],
                    end_logits[_idx],
                    lengths[_idx] - 1,
                    offset_mappings[_idx],
                    context_offsets[_idx],
                    topk=num_answers,
                    enforce_logits_positive=True,
                    raw_span=False,
                    extract_answer=False,
                    max_length=max_length,
                )

                predictions[sample_id].add(
                    sample_id=sample_id,
                    instance_id=instance_id,
                    context=contexts[_idx],
                    scores=scores,
                    spans=spans,
                    answers=[],
                    no_answer_score=no_answer_score,
                )  # samples in batch have the same context since they stem from the same sample
                # answers are only available if the sample is answerable
    return predictions


def longest_common_subsequence(predicate, sequence_a, sequence_b):
    # this algorithm searches for the longest common substring in two sequences, and will find the optimal value
    # runtime is O(n * 2m+m)
    # additionally the predicate is a callable and also to skip values where it evaluates to Trues
    start_idx = -1
    max_length = 0
    for idx_a, item_a in enumerate(sequence_a):
        if predicate(item_a):
            continue
        for idx_b, item_b in enumerate(sequence_b):
            if predicate(item_b):
                continue
            common_prefix_length = len(
                os.path.commonprefix((sequence_a[idx_a:], sequence_b[idx_b:]))
            )
            if common_prefix_length > max_length:
                max_length = common_prefix_length
                start_idx = idx_a
    return start_idx, max_length


def match_prediction_with_context(contexts, predictions):
    if isinstance(contexts, str):
        contexts = [contexts]
    if isinstance(predictions, str):
        predictions = [predictions]
    predictions_rectified = []
    filter_fn = lambda x: x == ""
    for context, prediction in zip(contexts, predictions):
        context_normalized = normalize_answer(context)
        prediction_normalized = normalize_answer(prediction)
        start_idx, length = longest_common_subsequence(
            filter_fn, context_normalized, prediction_normalized
        )
        prediction_normalized_rectified = context_normalized[
            start_idx : start_idx + length
        ]
        predictions_rectified.append(prediction_normalized_rectified)
    return predictions_rectified


class RCEvaluator:
    def __init__(
        self,
        dataset: Dataset,
        max_answer_length: int,
        num_answers: int = 5,
        no_answer_option: bool = True,
        use_extracted_answer_if_possible: bool = True,
        prediction_file: str = None,
    ):
        # using the dataset we can extract the labels etc.
        self.dataset = dataset

        # number of answers to predict for each sample
        self.num_answers = num_answers
        self.no_answer_option = no_answer_option
        self.max_answer_length = max_answer_length
        self.use_extracted_answer_if_possible = use_extracted_answer_if_possible

        # get labels
        self.labels = self._get_labels()

        # prediction related
        self.prediction_file = prediction_file

    def _get_label(self, sample):
        answers = sample["answers"]
        if self.use_extracted_answer_if_possible and "extracted_answer" in sample:
            # use answer extracted from span; since we run sanity checks for the char span this should be redundant anyway
            answers += sample["extracted_answer"]
        return [normalize_answer(answer) for answer in answers]

    def _get_labels(self):
        # collect answers
        labels = {}
        for sample in self.dataset:
            if sample["answers"] is not None:
                # answerable sample
                label = self._get_label(sample)
                # label should be a sequence
                if not isinstance(label, (list, set, tuple)):
                    label = [label]
                labels[sample["id"]] = label
            else:
                # non-answerable sample
                labels[sample["id"]] = None
        return labels

    def __call__(self, predictions_and_labels: EvalPrediction):
        # numpy array are returned (for whatever reason)
        predictions = predictions_and_labels.predictions
        # logits = np.stack(logits, axis=2)
        assert len(predictions) == len(self.dataset)
        scores = predictions_and_labels.label_ids

        # collect best spans for each instance
        predictions = self._predict(predictions, scores)

        if self.prediction_file is not None:
            # write predictions to file
            self._write_predictions(
                {
                    id_: predicted_answer[0][0]
                    for id_, predicted_answer in predictions.items()
                }
            )

        # evaluate answers
        return self._evaluate(predictions)

    def _write_predictions(self, predictions: Dict):
        if self.prediction_file is not None:
            with open(self.prediction_file, "w") as f:
                json.dump(predictions, f)

    def _evaluate(self, predictions: Dict):
        if self.no_answer_option:
            return self._evaluate_thresholds(predictions)
        else:
            # don't check for thresholds since we don't have to compare answer scores with no-answer option
            return self._evaluate_predictions(predictions)

    def _predict(self, logits, scores=None):
        # here we can make use of self.dataset because data is not shuffled during prediction/evaluation
        return self._predict_answers(
            rc_predict(
                logits,
                self.dataset["id"],
                self.dataset["id"],
                self.dataset["context"],
                self.dataset["context_offset"],
                self.dataset["length"],
                self.dataset["offset_mapping"],
                self.num_answers,
                self.max_answer_length,
            )
        )

    def _predict_answers(self, predictions: Dict[str, RCPrediction]):
        """Predict answers from all predictions per sample."""

        # per sample prediction (considering all predictions)
        predicted_answers = {}
        for sample_id, prediction in predictions.items():
            no_answer_score = (
                prediction.no_answer_score
            )  # contains best no-answer-score over all chunks
            scores, answers, spans = prediction.extract_answers(self.num_answers)
            answers = [
                normalize_answer(answer) for answer in answers
            ]  # normalize answers, keeps order

            predicted_answers[sample_id] = answers, spans, scores, no_answer_score
        return predicted_answers

    def _evaluate_prediction(
        self, label, scores, answers, spans, no_answer_score, threshold: float = None
    ):
        # threshold is added to non-answer score; non-answer score is in [0,2]
        # answers contain self.num_answers predicted answers sorted by their score

        # max(scores) - no_answer_score is in range [-1,2]
        # if threshold is None then always return an answer
        if threshold is not None and no_answer_score + threshold >= max(scores):
            return None, answers
        else:
            return answers[0], answers

    def _evaluate_predictions(self, predictions: Dict, threshold: float = None):
        metrics = RCMetric()

        for sample_id, label in self.labels.items():
            answers, spans, scores, no_answer_score = predictions[sample_id]
            metrics.add_prediction(
                label,
                *self._evaluate_prediction(
                    label=label,
                    scores=scores,
                    answers=answers,
                    spans=spans,
                    no_answer_score=no_answer_score,
                    threshold=threshold,
                ),
            )
        return metrics.get()

    def _get_thresholds(self, predictions):
        # try some thresholds (grid search)

        thresholds = [float("-inf"), float("inf")] + torch.arange(-1, 2, 0.01).tolist()
        return thresholds

    def _evaluate_thresholds(self, predictions, metric_optimize_for: str = "f1"):
        thresholds = self._get_thresholds(predictions)
        logging.info(
            f"Evaluating {len(thresholds)} thresholds: {', '.join(map(str, thresholds))}"
        )

        # find best threshold
        best_metrics: dict = None
        best_threshold = None
        for threshold in thresholds:
            metrics = self._evaluate_predictions(predictions, threshold)
            if (
                best_metrics is None
                or metrics[metric_optimize_for] > best_metrics[metric_optimize_for]
            ):
                best_threshold = threshold
                best_metrics = metrics

        logging.info(
            f"Best threshold is {best_threshold} with an {metric_optimize_for} score of {best_metrics[metric_optimize_for]}"
        )
        return best_metrics


class HfRCEvaluator(RCEvaluator):
    def __init__(self, dataset: Dataset, no_answer_option: bool, **kwargs):
        super().__init__(
            dataset, num_answers=1, no_answer_option=no_answer_option, **kwargs
        )

        # load hf metric
        self.metric = load_metric("squad" if not no_answer_option else "squad_v2")

    def _get_labels(self):
        # SQuAD hf metric needs id and answers.text (answer_start is just there to match the feature format)
        return [
            {
                "id": sample["id"],
                "answers": {"text": sample["answers"], "answer_start": []},
            }
            for sample in select_unique(self.dataset, column="id")
        ]

    def _evaluate(self, predictions: Dict):
        return self.metric.compute(
            predictions=[
                {"id": sample_id, "prediction_text": prediction[0][0]}
                for sample_id, prediction in predictions.items()
            ],
            references=self.labels,
        )


class TechQAEvaluator(RCEvaluator):
    def _get_label(self, sample):
        # for techqa-style evaluation we use the character overlap using the spans
        return sample["char_spans"]

    def _evaluate_prediction(
        self, label, scores, answers, spans, no_answer_score, threshold
    ):
        # TechQA uses a different evaluation with a different threshold:
        # systems must always return 5 answers (including score) and a threshold which is valid for the entire run over all documents
        # if all answer scores are lower than the threshold then predicted answer is no-answer
        # spans contain self.num_answers predicted answer char spans sorted by their score

        # account for span end being exclusive
        spans = [(span[0], span[1] - 1) for span in spans]

        # if threshold is None then always return an answer
        if threshold is not None and threshold > max(scores):
            return None, spans
        else:
            return spans[0], spans

    def _get_thresholds(self, predictions):
        # compute possible thresholds
        thresholds = [float("inf")]
        for prediction in predictions.values():
            _, _, scores, _ = prediction
            thresholds.append(
                max(scores)
            )  # append best score so that not all scores of those predictions are lower than this threshold

        return thresholds


class RCGenEvaluator(RCEvaluator):
    def __init__(
        self,
        *args,
        tokenizer: PreTrainedTokenizer,
        match_answers_with_context: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args, **dict(kwargs, no_answer_option=False), max_answer_length=None
        )

        # using the dataset we can extract the labels etc.
        self.tokenizer = tokenizer

        self.match_answers_with_context = match_answers_with_context

    def _get_label(self, sample):
        # answers = sample['answers']
        answers = sample["answers"]["text"]
        assert len(answers) > 0 and isinstance(
            answers, list
        ), f"Expected answers of type list but got type {type(answers)}: {answers}"
        if self.use_extracted_answer_if_possible and "extracted_answer" in sample:
            # use answer extracted from span; since we run sanity checks for the char span this should be redundant anyway
            answers += sample["extracted_answer"]
        return [normalize_answer(answer) for answer in answers]

    def _predict(self, predictions, scores):
        # extract best answer per sample (aggregate chunks)
        answers = [
            self.tokenizer.decode(
                [_id for _id in ids if _id != -100], skip_special_tokens=True
            )
            for ids in tqdm(predictions, desc="Decoding generated sequences")
        ]
        predictions_per_sample = defaultdict(partial(RCPrediction, 1))
        for sample_id, answer, score in zip(self.dataset["id"], answers, scores):
            predictions_per_sample[sample_id].add(
                sample_id=sample_id,
                instance_id=sample_id,
                context=None,
                scores=[score],
                spans=[],
                answers=[answer],
                no_answer_score=None,
            )  # samples in batch have the same context since they stem from the same sample
        return self._predict_answers(predictions_per_sample)


class HfRCGenEvaluator(RCGenEvaluator):
    def __init__(self, dataset: Dataset, **kwargs):
        super().__init__(dataset, num_answers=1, **kwargs)

        # load hf metric
        self.metric = load_metric("squad")

    def _get_labels(self):
        # SQuAD hf metric needs id and answers.text (answer_start is just there to match the feature format)
        return [
            self._metric_ref_format(sample_id, answer)
            for sample_id, answer in super()._get_labels().items()
        ]

    @staticmethod
    def _metric_ref_format(sample_id, answer):
        return {"id": sample_id, "answers": {"text": answer, "answer_start": []}}

    @staticmethod
    def _metric_pred_format(sample_id, answer):
        return {"id": sample_id, "prediction_text": answer}

    def _evaluate_predictions(self, predictions: List):
        for ref in self.labels:
            answers, spans, scores, no_answer_score = predictions[ref["id"]]
            self.metric.add(
                prediction=self._metric_pred_format(
                    ref["id"],
                    self._evaluate_prediction(
                        label=None,
                        scores=scores,
                        answers=answers,
                        spans=spans,
                        no_answer_score=no_answer_score,
                        threshold=None,
                    )[0],
                ),
                reference=ref,
            )
        return self.metric.compute()


def get_evaluator_class(style: str):
    if style == "hf-squad":
        logging.info("Running with SQuAD (from hf) evaluation")
        evaluator_class = HfRCEvaluator
    elif style == "squad":
        logging.info("Running with SQuAD evaluation")
        evaluator_class = RCEvaluator
    elif style == "techqa":
        logging.info("Running with TechQA evaluation")
        evaluator_class = TechQAEvaluator
    else:
        raise ValueError(f"Unknown evaluation style '{style}'")

    return evaluator_class

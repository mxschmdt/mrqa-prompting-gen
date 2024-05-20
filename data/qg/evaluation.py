from typing import Iterable, List

import evaluate
from tqdm import tqdm
from transformers import EvalPrediction, PreTrainedTokenizer


class QGEvaluator:
    def __init__(
        self,
        reference_questions: Iterable[str],
        tokenizer: PreTrainedTokenizer,
        metrics: List[str] = None,
    ):
        # re store the references for evaluation
        self.reference_questions = reference_questions
        self.tokenizer = tokenizer

        self.metrics = [evaluate.load(metric) for metric in metrics]

    def __call__(self, predictions_and_labels: EvalPrediction):
        # numpy array are returned (for whatever reason)
        generated_token_ids = predictions_and_labels.predictions
        assert len(generated_token_ids) == len(self.reference_questions)
        # labels = predictions_and_labels.label_ids

        generated_sequences = [
            self.tokenizer.decode(
                [_id for _id in ids if _id != -100], skip_special_tokens=True
            )
            for ids in tqdm(generated_token_ids, desc="Decoding generated sequences")
        ]

        # evaluate answers
        scores = {}
        for metric in self.metrics:
            scores.update(
                metric.compute(
                    references=self.reference_questions, predictions=generated_sequences
                )
            )

        return scores

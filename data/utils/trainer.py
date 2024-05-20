from typing import Callable, List

from transformers import Seq2SeqTrainer, Trainer

from data.utils.data import NamedDataset


class MultiEvalTrainerMixIn:
    def __init__(
        self,
        add_eval_datasets: List[NamedDataset] = None,
        evaluators: List[Callable] = None,
        evaluator_fn: Callable = None,
        metric_key_prefix: str = None,
        *args,
        **kwargs,
    ):
        # make sure to call other parent classes in multiple inheritance
        super().__init__(*args, **kwargs)

        self.metric_key_prefix = metric_key_prefix

        self.add_eval_datasets = add_eval_datasets
        if self.add_eval_datasets:
            if evaluators is not None:
                self.evaluators = evaluators
            else:
                assert (
                    evaluator_fn is not None
                ), "`evaluator_fn` cannot be None if `evaluators` is None and `add_eval_datasets` is given."
                self.evaluators = [
                    evaluator_fn(dataset=dataset.data) for dataset in add_eval_datasets
                ]

    def evaluate(self, *args, **kwargs):
        if self.metric_key_prefix is not None:
            metric_key_prefix = kwargs.pop("metric_key_prefix", "")
            metric_key_prefix += self.metric_key_prefix
            kwargs.update(metric_key_prefix=metric_key_prefix)
        if not self.add_eval_datasets:
            return super().evaluate(*args, **kwargs)

        ## keep behaviour, i.e. evaluate on eval dataset
        try:
            metrics = super().evaluate(*args, **kwargs)
        except:
            # no evaluation dataset set
            metrics = {}
        ## evaluate on additional datasets

        # backup compute_metrics fn (used in super().evaluate)
        compute_metrics_backup = self.compute_metrics

        # remove metric_key_prefix from args if specified
        if len(args) >= 3:
            args = args[:2] + args[3:]
        metric_key_prefix = kwargs.pop("metric_key_prefix", None)

        # remove eval_dataset from args if specified
        if len(args) > 1:
            args = args[1:]
        else:
            args = ()
        kwargs.pop("eval_dataset", None)

        # run super().evaluate for each additional eval dataset
        for dataset, evaluator in zip(self.add_eval_datasets, self.evaluators):
            self.compute_metrics = evaluator
            metrics.update(
                super().evaluate(
                    *args,
                    **kwargs,
                    eval_dataset=dataset.data,
                    metric_key_prefix=f"{(metric_key_prefix + '/') if metric_key_prefix else ''}{dataset.path}/{dataset.config}/{dataset.split}",
                )
            )

        # reset compute_metrics fn (used in super().evaluate)
        self.compute_metrics = compute_metrics_backup
        return metrics


class MultiEvalTrainer(MultiEvalTrainerMixIn, Trainer):
    pass


class MultiEvalSeq2SeqTrainer(MultiEvalTrainerMixIn, Seq2SeqTrainer):
    pass

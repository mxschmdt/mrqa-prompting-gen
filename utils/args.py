from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from transformers import (
    IntervalStrategy,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    integrations,
)


def log_to_experiment(log: Dict, log_as_hparam: bool = False):
    """Logs keys and their values to active experiments (currently CometML and WandB).

    Args:
        log (Dict): The keys and values to log.
        log_as_hparam (bool, optional): Whether to log as hyperparameter or not (only applicaple to CometML). Defaults to False.
    """
    # log to CometML experiment
    if integrations.is_comet_available():
        experiment: integrations.comet_ml.Experiment = (
            integrations.comet_ml.config.get_global_experiment()
        )
        if experiment is not None:
            if log_as_hparam:
                log_fn = experiment.log_parameter
            else:
                log_fn = experiment.log_other

            for key, value in log.items():
                log_fn(key, value)

    # log to WandB experiment
    if integrations.is_wandb_available() and integrations.wandb.run is not None:
        integrations.wandb.run.config.update(log)


def finish_experiment():
    # finish WandB and CometML logging
    if integrations.is_wandb_available():
        integrations.wandb.finish()

    if integrations.is_comet_available():
        experiment: integrations.comet_ml.Experiment = (
            integrations.comet_ml.config.get_global_experiment()
        )
        if experiment is not None:
            experiment.flush()
            experiment.end()


@dataclass
class PromptingTrainingArguments(Seq2SeqTrainingArguments):
    """`transformers.Seq2SeqTrainingArguments` with updated default values for Prompting"""

    predict_with_generate: bool = field(
        default=True,
        metadata={
            "help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."
        },
    )

    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

    per_device_eval_batch_size: int = field(
        default=40, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    gradient_accumulation_steps: int = field(
        default=2,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )

    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )

    learning_rate: float = field(
        default=3e-5, metadata={"help": "The initial learning rate for AdamW."}
    )

    # evaluation_strategy: IntervalStrategy = field(
    #     default="steps",
    #     metadata={"help": "The evaluation strategy to use."},
    # )

    save_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )

    logging_first_step: bool = field(
        default=False, metadata={"help": "Log the first global_step"}
    )
    logging_steps: int = field(
        default=1000, metadata={"help": "Log every X updates steps."}
    )
    eval_steps: int = field(
        default=1000, metadata={"help": "Run an evaluation every X steps."}
    )
    save_steps: int = field(
        default=1000, metadata={"help": "Save checkpoint every X updates steps."}
    )

    save_total_limit: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir (but never the best model if tracked (`load_best_model_at_end=True`)). Default is 1 checkpoint"
            )
        },
    )

    seed: Optional[int] = field(
        default=None,
        metadata={
            "help": "Random seed that will be set at the beginning of training.",
            "nargs": "+",
        },
    )

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not to load the best model found during training at the end of training."
        },
    )

    metric_for_best_model: Optional[str] = field(
        default=None,
        metadata={"help": "The metric to use to compare two different models."},
    )

    use_legacy_prediction_loop: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use the legacy prediction_loop in the Trainer."
        },
    )

    tags: List[str] = field(
        default_factory=list,
        metadata={"help": "Tags to be added to the comet.ml experiment."},
    )

    load_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to a folder with a valid checkpoint for your model. Does not continue training from this checkpoint, i.e. optimizer and scheduler are not loaded."
        },
    )

    load_best_checkpoint: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to load best checkpoints from given path (assuming only best and most recent model are stored)"
        },
    )

    min_steps: Optional[int] = field(
        default=None,
        metadata={"help": "The minimum number of training steps."},
    )


@dataclass
class PromptingModelArguments:
    """model arguments (passed to `init`)"""

    transformer: str = field(
        metadata={"help": "The transformer used in the model."},
    )

    pretrained: Optional[str] = field(
        default=None, metadata={"help": "Start with the given model checkpoint."}
    )

    freeze_encoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Freeze the encoder, i.e. do not train its weights."},
    )

    freeze_encoder_embedding: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Freeze the encoder's embedding weights, i.e. do not train these weights."
        },
    )

    lambda_pretrained_embedding_weight_decay: float = field(
        default=0.0, metadata={"help": "Pre-trained embedding weight decay lambda."}
    )

    lambda_pretrained_output_weight_decay: float = field(
        default=0.0, metadata={"help": "Pre-trained output layer weight decay lambda."}
    )

    train_layers: int = field(
        default=-1,
        metadata={
            "help": "Train only top layers down to index `train_layers`, where 0 is the output layer. Default is -1 (train all layers)."
        },
    )

    reinit_layers: int = field(
        default=-1,
        metadata={
            "help": "Re-initialize top layers down to index `train_layers`, where 0 is the output layer. Default is -1 (train all layers)."
        },
    )


@dataclass
class PromptingDataArguments:
    """Arguments concerning data processing"""

    preprocess: bool = field(
        metadata={"help": "Will force to preprocess any data."},
    )

    rectify_answers: bool = field(
        default=False,
        metadata={"help": "Rectify answers."},
    )

    rectify_questions: bool = field(
        default=True,
        metadata={"help": "Don't rectify questions."},
    )

    eval_style: str = field(
        default="squad",
        metadata={"help": "Set the evaluation style."},
    )

    disable_no_answer: bool = field(
        default=False,
        metadata={"help": "Non-answers disabled if set to True."},
    )

    max_input_length: int = field(
        default=-1,
        metadata={
            "help": "The maximum input length for the model, overflowing tokens will be sliced into chunks."
        },
    )

    skip_question_length: int = field(
        default=-1,
        metadata={
            "help": "'The maximum question length in tokens (questions with more tokens will be skipped)."
        },
    )

    truncate_question_length: int = field(
        default=-1,
        metadata={
            "help": "The maximum question length in tokens (questions with more tokens will be truncated)."
        },
    )

    max_answer_length: int = field(
        default=-1,
        metadata={
            "help": "The maximum length of the answer to predict in case of the rc model."
        },
    )

    stride: int = field(
        default=128,
        metadata={
            "help": "The maximum input length for the model, overflowing tokens will be sliced into chunks."
        },
    )

    num_workers: int = field(
        default=1,
        metadata={"help": "The number of worker used for preprocessing data."},
    )

    separate_answers: bool = field(
        default=False,
        metadata={
            "help": "Whether answers are unpacked by creating new instances or not."
        },
    )

    keep_in_memory: bool = field(
        default=False,
        metadata={"help": "Will keep preprocessed data in memory if True."},
    )

    unique: str = field(
        default=None,
        metadata={
            "metavar": "column",
            "help": "The column to be unique in the dataset. If None then no filtering is applied.",
        },
    )

    train_chunking_mode: str = field(
        default=None,
        metadata={
            "help": "The mode for chunking contexts of training data, either 'token' or 'sentence'",
            "choices": ["token", "sentence"],
        },
    )

    train_context_stride: int = field(
        default=None,
        metadata={"help": "The stride for chunking contexts of training data"},
    )

    train_context_size: int = field(
        default=None,
        metadata={"help": "The size for chunking contexts of training data"},
    )

    train_truncate_contexts: bool = field(
        default=None,
        metadata={"help": "Whether to truncate contexts of training data or not"},
    )

    eval_chunking_mode: str = field(
        default=None,
        metadata={
            "help": "The mode for chunking contexts of evaluation data, either 'token' or 'sentence'",
            "choices": ["token", "sentence"],
        },
    )

    eval_context_stride: int = field(
        default=None,
        metadata={"help": "The stride for chunking contexts of evaluation data"},
    )

    eval_context_size: int = field(
        default=None,
        metadata={"help": "The size for chunking contexts of evaluation data"},
    )

    eval_truncate_contexts: bool = field(
        default=None,
        metadata={"help": "Whether to truncate contexts of evaluation data or not"},
    )

    predict_chunking_mode: str = field(
        default=None,
        metadata={
            "help": "The mode for chunking contexts of prediction data, either 'token' or 'sentence'",
            "choices": ["token", "sentence"],
        },
    )

    predict_context_stride: int = field(
        default=None,
        metadata={"help": "The stride for chunking contexts of prediction data"},
    )

    predict_context_size: int = field(
        default=None,
        metadata={"help": "The size for chunking contexts of prediction data"},
    )

    predict_truncate_contexts: bool = field(
        default=None,
        metadata={"help": "Whether to truncate contexts of prediction data or not"},
    )


@dataclass
class RCTrainingArguments(TrainingArguments):
    """`transformers.TrainingArguments` with updated default values."""

    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )

    per_device_eval_batch_size: int = field(
        default=40, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )

    gradient_accumulation_steps: int = field(
        default=2,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )

    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )

    learning_rate: float = field(
        default=3e-5, metadata={"help": "The initial learning rate for AdamW."}
    )

    logging_first_step: bool = field(
        default=False, metadata={"help": "Log the first global_step"}
    )
    logging_steps: int = field(
        default=1000, metadata={"help": "Log every X updates steps."}
    )
    eval_steps: int = field(
        default=1000, metadata={"help": "Run an evaluation every X steps."}
    )
    save_steps: int = field(
        default=1000, metadata={"help": "Save checkpoint every X updates steps."}
    )

    save_total_limit: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir (but never the best model if tracked (`load_best_model_at_end=True`)). Default is 1 checkpoint"
            )
        },
    )

    seed: Union[int, None] = field(
        default=None,
        metadata={
            "help": "Random seed that will be set at the beginning of training.",
            "nargs": "+",
        },
    )

    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether or not to load the best model found during training at the end of training."
        },
    )

    metric_for_best_model: Optional[str] = field(
        default=None,
        metadata={"help": "The metric to use to compare two different models."},
    )

    use_legacy_prediction_loop: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use the legacy prediction_loop in the Trainer."
        },
    )

    tags: List[str] = field(
        default_factory=list,
        metadata={"help": "Tags to be added to the comet.ml experiment."},
    )

    load_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to a folder with a valid checkpoint for your model. Does not continue training from this checkpoint, i.e. optimizer and scheduler are not loaded."
        },
    )

    load_best_checkpoint: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to load best checkpoints from given path (assuming only best and most recent model are stored)"
        },
    )


@dataclass
class RCModelArguments:
    """model arguments (passed to `init`)"""

    transformer: str = field(
        metadata={"help": "The transformer used in the model."},
    )

    pretrained: Optional[str] = field(
        default=None, metadata={"help": "Start with the given model checkpoint."}
    )

    freeze_encoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Freeze the encoder, i.e. do not train its weights."},
    )

    freeze_encoder_embedding: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Freeze the encoder's embedding weights, i.e. do not train these weights."
        },
    )

    lambda_pretrained_embedding_weight_decay: float = field(
        default=0.0, metadata={"help": "Pre-trained embedding weight decay lambda."}
    )

    lambda_pretrained_output_weight_decay: float = field(
        default=0.0, metadata={"help": "Pre-trained output layer weight decay lambda."}
    )

    train_layers: int = field(
        default=-1,
        metadata={
            "help": "Train only top layers down to index `train_layers`, where 0 is the output layer. Default is -1 (train all layers)."
        },
    )

    reinit_layers: int = field(
        default=-1,
        metadata={
            "help": "Re-initialize top layers down to index `train_layers`, where 0 is the output layer. Default is -1 (train all layers)."
        },
    )

    output_layers: List[int] = field(
        default=None,
        metadata={
            "help": "The layers for the span extraction head (specified as output dimensions), a layer mapping to 2 (num classes) is always added."
        },
    )

    adapter_mode: Optional[str] = field(
        default=None, metadata={"help": "The adapter mode for the model."}
    )

    # TODO add cl arg


@dataclass
class RCDataArguments:
    """Arguments concerning data processing"""

    preprocess: bool = field(
        metadata={"help": "Will force to preprocess any data."},
    )

    rectify_answers: bool = field(
        default=False,
        metadata={"help": "Rectify answers."},
    )

    rectify_questions: bool = field(
        default=True,
        metadata={"help": "Don't rectify questions."},
    )

    eval_style: str = field(
        default="squad",
        metadata={"help": "Set the evaluation style."},
    )

    disable_no_answer: bool = field(
        default=False,
        metadata={"help": "Non-answers disabled if set to True."},
    )

    max_input_length: int = field(
        default=-1,
        metadata={
            "help": "The maximum input length for the model, overflowing tokens will be sliced into chunks."
        },
    )

    skip_question_length: int = field(
        default=-1,
        metadata={
            "help": "'The maximum question length in tokens (questions with more tokens will be skipped)."
        },
    )

    truncate_question_length: int = field(
        default=-1,
        metadata={
            "help": "The maximum question length in tokens (questions with more tokens will be truncated)."
        },
    )

    max_answer_length: int = field(
        default=-1,
        metadata={
            "help": "The maximum length of the answer to predict in case of the rc model."
        },
    )

    stride: int = field(
        default=128,
        metadata={
            "help": "The maximum input length for the model, overflowing tokens will be sliced into chunks."
        },
    )

    num_workers: int = field(
        default=1,
        metadata={"help": "The number of worker used for preprocessing data."},
    )

    separate_answers: bool = field(
        default=False,
        metadata={
            "help": "Whether answers are unpacked by creating new instances or not."
        },
    )

    keep_in_memory: bool = field(
        default=False,
        metadata={"help": "Will keep preprocessed data in memory if True."},
    )

    unique: str = field(
        default=None,
        metadata={
            "metavar": "column",
            "help": "The column to be unique in the dataset. If None then no filtering is applied.",
        },
    )

    recreate_train_index: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This will create new indices for the training samples, only use if really needed and if you know what you're doing!"
        },
    )

import copy
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import dill
import openprompt
import torch
from datasets.fingerprint import Hasher
from openprompt import Template, Verbalizer
from openprompt.data_utils import InputFeatures
from openprompt.plms.utils import TokenizerWrapper
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from transformers import GenerationConfig, PreTrainedTokenizer, is_datasets_available
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import seed_worker

from data.utils.trainer import MultiEvalSeq2SeqTrainer

logger = logging.getLogger(__name__)

TEMPLATE_NAME = "template.bin"


class PromptDataLoader(openprompt.PromptDataLoader):
    """Using this class, transformers' Trainer will correctly print the number of available samples."""

    def __init__(
        self,
        dataset: Union[Dataset, List],
        template: Template,
        tokenizer_wrapper: Optional[TokenizerWrapper] = None,
        tokenizer: PreTrainedTokenizer = None,
        tokenizer_wrapper_class=None,
        verbalizer: Optional[Verbalizer] = None,
        max_seq_length: Optional[str] = 512,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = False,
        teacher_forcing: Optional[bool] = False,
        decoder_max_length: Optional[int] = -1,
        predict_eos_token: Optional[bool] = False,
        truncate_method: Optional[str] = "tail",
        drop_last: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            dataset,
            template,
            tokenizer_wrapper,
            tokenizer,
            tokenizer_wrapper_class,
            verbalizer,
            max_seq_length,
            batch_size,
            shuffle,
            teacher_forcing,
            decoder_max_length,
            predict_eos_token,
            truncate_method,
            drop_last,
            **kwargs,
        )

        if self.shuffle:
            sampler = RandomSampler(self.tensor_dataset)
        else:
            sampler = None

        self.dataloader = DataLoader(
            self.tensor_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=default_collate,
            drop_last=drop_last,
        )

        logger.info(f"Truncation rate: {self.tokenizer_wrapper.truncate_rate}")

    @property
    def dataset(self):
        return self.dataloader.dataset

    def tokenize(self) -> None:
        r"""Pass the wraped text into a prompt-specialized tokenizer,
        the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
        """
        for idx, wrapped_example in tqdm(
            enumerate(self.wrapped_dataset), desc="tokenizing"
        ):
            # we deepcopy the wrapped_examples since some tokenizer_wrapper update in-place!
            # for idx, wrapped_example in enumerate(self.wrapped_dataset):
            # get rid of InputFeatures
            inputfeatures = dict(
                **InputFeatures(
                    **self.tokenizer_wrapper.tokenize_one_example(
                        copy.deepcopy(wrapped_example), self.teacher_forcing
                    ),
                    **wrapped_example[1],
                ).to_tensor()
            )
            if self.teacher_forcing:
                # create inputs for prediction since e.g. decoder_input_ids contain the label
                tokenized_example_for_prediction = (
                    self.tokenizer_wrapper.tokenize_one_example(
                        copy.deepcopy(wrapped_example), False
                    )
                )

                for k, v in tokenized_example_for_prediction.items():
                    inputfeatures["predict_" + k] = torch.tensor(v)

            self.tensor_dataset.append(inputfeatures)


gen_kwargs = {
    "max_new_tokens": 50,
    "min_length": 1,
    "do_sample": False,
    "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
}
gen_config = GenerationConfig.from_dict(gen_kwargs)


class PromptTrainer(MultiEvalSeq2SeqTrainer):
    def __init__(self, wrapper_class, *args, generate_postprocess_fn=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.generate_postprocess_fn = generate_postprocess_fn
        self.wrapper_class = wrapper_class
        self._cached_eval_dataloader_mapping = {}

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # load template from checkpoint
        self.model.prompt_model.template = self._load_template_from_checkpoint(
            resume_from_checkpoint
        )

        # load remaining
        super()._load_from_checkpoint(resume_from_checkpoint, model)

        # we don't save the tokenizer of the template since it causes trouble during deserialization
        if (
            hasattr(self.model.prompt_model.template, "tokenizer")
            and self.model.prompt_model.template.tokenizer is None
        ):
            self.model.prompt_model.template.tokenizer = self.tokenizer

        # we don't save the raw_embedding of the template to save space hence we set it here
        if (
            hasattr(self.model.prompt_model.template, "raw_embedding")
            and self.model.prompt_model.template.raw_embedding is None
        ):
            self.model.prompt_model.template.raw_embedding = (
                self.model.prompt_model.plm.get_input_embeddings()
            )

    @staticmethod
    def _load_template_from_checkpoint(checkpoint):
        # load template from checkpoint

        if not os.path.isfile(os.path.join(checkpoint, TEMPLATE_NAME)):
            raise ValueError(f"Can't find a valid template at {checkpoint}.")

        template = torch.load(os.path.join(checkpoint, TEMPLATE_NAME))
        logging.info(f"Loaded template from checkpoint ({checkpoint}): {template.text}")
        return template

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        # after saving the model also save the template so that one can load it with the model weights
        # note that any weights attached to the (soft) template are added to the model anyway so we only need the template to process inputs similar to training
        super().save_model(output_dir, _internal_call)

        if output_dir is None:
            output_dir = self.args.output_dir

        # save template (this also saves the weights additionally)
        # for some reason copy.copy() is not sufficient
        template = copy.deepcopy(self.model.prompt_model.template)
        if hasattr(self.model.prompt_model.template, "raw_embedding"):
            # omit the raw embedding which is a reference to the model input embedding and only used on template creation or if it is changed
            template.raw_embedding = None

        # remove tokenizer form template since it causes trouble during deserialization
        template.tokenizer = None

        torch.save(
            template, os.path.join(output_dir, TEMPLATE_NAME), pickle_module=dill
        )
        logging.info(f"Template saved in {os.path.join(output_dir, TEMPLATE_NAME)}")

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        # data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )

        if self.model.config.is_encoder_decoder:
            max_seq_length = self.tokenizer.model_max_length
        else:
            max_seq_length = self.tokenizer.model_max_length - 25

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                raise NotImplementedError(
                    "PromptDataloader has not been implemented for multiprocessing."
                )

            return PromptDataLoader(
                dataset=train_dataset,
                template=self.model.prompt_model.template,
                tokenizer=self.tokenizer,
                tokenizer_wrapper_class=self.wrapper_class,
                max_seq_length=max_seq_length,
                decoder_max_length=self.tokenizer.model_max_length,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                teacher_forcing=True,
                predict_eos_token=True,
                truncate_method="head",
            )

        train_sampler = self._get_train_sampler()

        return PromptDataLoader(
            dataset=train_dataset,
            template=self.model.prompt_model.template,
            tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.wrapper_class,
            max_seq_length=max_seq_length,
            decoder_max_length=self.tokenizer.model_max_length,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
            teacher_forcing=True,
            predict_eos_token=True,
            truncate_method="head",
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
                the `model.forward()` method are automatically removed. It must implement `__len__`.
        """

        def generate_fingerprint(obj) -> str:
            hasher = Hasher()
            if isinstance(obj, list):
                hasher.update(obj)
            else:
                state = obj.__dict__
                for key in sorted(state):
                    hasher.update(key)
                    hasher.update(state[key])
            return hasher.hexdigest()

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        # check if we already cached the eval dataloader
        fingerprint = generate_fingerprint(eval_dataset)
        if fingerprint in self._cached_eval_dataloader_mapping:
            logging.info("Using cached dataloader for evaluation")
            return self._cached_eval_dataloader_mapping[fingerprint]

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )

        if self.model.config.is_encoder_decoder:
            max_seq_length = self.tokenizer.model_max_length
        else:
            # TODO make dependend on max generated tokens
            max_seq_length = self.tokenizer.model_max_length - 25

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                raise NotImplementedError(
                    "PromptDataloader has not been implemented for multiprocessing yet."
                )
            dataloader = PromptDataLoader(
                dataset=eval_dataset,
                template=self.model.prompt_model.template,
                tokenizer=self.tokenizer,
                tokenizer_wrapper_class=self.wrapper_class,
                max_seq_length=max_seq_length,
                decoder_max_length=self.tokenizer.model_max_length,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                teacher_forcing=True,
                predict_eos_token=False,
                truncate_method="head",
            )
            # cache dataloader
            self._cached_eval_dataloader_mapping[fingerprint] = dataloader
            return dataloader

        dataloader = PromptDataLoader(
            dataset=eval_dataset,
            template=self.model.prompt_model.template,
            tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.wrapper_class,
            max_seq_length=max_seq_length,
            decoder_max_length=self.tokenizer.model_max_length,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            teacher_forcing=True,
            predict_eos_token=False,
            truncate_method="head",
        )
        # cache dataloader
        self._cached_eval_dataloader_mapping[fingerprint] = dataloader
        return dataloader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is an `datasets.Dataset`, columns not accepted by the `model.forward()`
                method are automatically removed. It must implement `__len__`.
        """

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="test")

        if self.model.config.is_encoder_decoder:
            max_seq_length = self.tokenizer.model_max_length
        else:
            # TODO make dependend on max generated tokens
            max_seq_length = self.tokenizer.model_max_length - 25

        if isinstance(test_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                raise NotImplementedError(
                    "PromptDataloader has not been implemented for multiprocessing yet."
                )
            return PromptDataLoader(
                dataset=test_dataset,
                template=self.model.prompt_model.template,
                tokenizer=self.tokenizer,
                tokenizer_wrapper_class=self.wrapper_class,
                max_seq_length=max_seq_length,
                decoder_max_length=self.tokenizer.model_max_length,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                teacher_forcing=False,
                predict_eos_token=False,
                truncate_method="head",
            )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return PromptDataLoader(
            dataset=test_dataset,
            template=self.model.prompt_model.template,
            tokenizer=self.tokenizer,
            tokenizer_wrapper_class=self.wrapper_class,
            max_seq_length=max_seq_length,
            decoder_max_length=self.tokenizer.model_max_length,
            batch_size=self.args.eval_batch_size,
            sampler=test_sampler,
            shuffle=False,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=False,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="head",
        )

    def _prepare_inputs(
        self, inputs: InputFeatures
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, moving them to the correct model.
        Transformers' Trainer doesn't work with `InputFeatures` but we can assume that they already contain tensors and just need to be moved to the correct device.
        """
        # get rid of InputFeatures class
        return super()._prepare_inputs(
            dict(**inputs) if isinstance(inputs, InputFeatures) else inputs
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        loss = model(inputs)
        # NOTE currently we always return None as outputs because we assume that this function is called only when training or computing the prediction loss
        return (loss, (loss, None)) if return_outputs else loss

    def predict(self, *args, **kwargs):
        predictions, scores, metrics = super().predict(*args, **kwargs)

        # convert token ids to words
        predictions = [
            self.tokenizer.decode(
                [_id for _id in ids if _id != -100], skip_special_tokens=True
            )
            for ids in tqdm(
                predictions, desc="Converting generated token ids to sequences"
            )
        ]

        return predictions, scores, metrics

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        if not self.args.predict_with_generate:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )
        batch_size = inputs["input_ids"].size(0)
        inputs = dict(**inputs)
        # if predict_* keys exist then we use them for prediction, otherwise we assume that no labels are given and we use the available keys for prediction
        # note that input_ids exists for encoder-decoder as well as decoder-only models
        has_labels = "predict_input_ids" in inputs
        if has_labels:
            prediction_inputs = {}
            for k in list(inputs.keys()):
                if k.startswith("predict_"):
                    prediction_inputs[k[8:]] = inputs.pop(k)
            loss_inputs = inputs
        else:
            prediction_inputs = inputs
        # delete inputs so that they are not used accidentally
        del inputs

        with torch.no_grad():
            # the prompt model will create the labels from decoder_input_ids hence it can always be computed
            if has_labels:
                with self.autocast_smart_context_manager():
                    loss_inputs = self._prepare_inputs(loss_inputs)
                    loss = self.compute_loss(model, loss_inputs)
                    # clear GPU memory (if used, otherwise doesn't hurt)
                    del loss_inputs
            else:
                loss = None

        if prediction_loss_only or self.args.prediction_loss_only:
            return (loss, None, None)

        # attention mask has to be part of inputs to generate() since PromptForGeneration model takes care of feeding it to the underlying generate method
        prediction_inputs = self._prepare_inputs(prediction_inputs)

        (
            batch_generated_token_ids_raw,
            batch_generated_sequences,
            batch_generated_sequences_scores,
        ) = self.model.generate(
            prediction_inputs,
            generation_config=gen_config,
        )

        # remove pad and bos token (they have been given as decoder input)
        preds_ids = torch.tensor(
            batch_generated_token_ids_raw, device=self.args.device
        )[
            :, 2:
        ]  # TODO omit first 1 or 2 tokens depending on whether eos is added or not
        # perplexity is exp(mean loss over sequence), and lower is better; therefore we * -1 since we want to have the higher the better, and omit exp() since it's a monotonic function and hence does not change the order
        pred_probs = -1.0 * torch.nn.CrossEntropyLoss(reduction="none", ignore_index=0)(
            input=torch.stack(batch_generated_sequences_scores, dim=2), target=preds_ids
        ).nan_to_num().mean(dim=1)

        if self.generate_postprocess_fn is not None:
            # use custom postprocess function
            # input lengths can be used in postprocess function to remove model input tokens
            if self.model.config.is_encoder_decoder:
                input_length = prediction_inputs["decoder_input_ids"].size(1)
            else:
                if "input_ids_len" in prediction_inputs:
                    input_length = prediction_inputs["input_ids_len"]
                else:
                    input_length = torch.sum(
                        (
                            prediction_inputs["input_ids"]
                            != self.tokenizer.pad_token_id
                        ).to(torch.int),
                        dim=-1,
                    )

            batch_generated_sequences = self.generate_postprocess_fn(
                batch_generated_token_ids_raw, input_length
            )
        else:
            # only one sequence is returned
            batch_generated_sequences = [[seq] for seq in batch_generated_sequences]
        # we cannot directly return the generated tokens since the transformers evaluation needs tensors (and pads them)
        batch_generated_token_ids = [
            [
                self.tokenizer.encode(sequence, add_special_tokens=False)
                for sequence in sequences
            ]
            for sequences in batch_generated_sequences
        ]

        # pad generated sequences so that we can return them as tensor
        max_length_tokens = max(
            len(token_ids)
            for seq_token_ids in batch_generated_token_ids
            for token_ids in seq_token_ids
        )
        batch_generated_token_ids = torch.tensor(
            [
                [
                    token_ids
                    + [self.tokenizer.pad_token_id]
                    * (max_length_tokens - len(token_ids))
                    for token_ids in sequences
                ]
                for sequences in batch_generated_token_ids
            ]
        )

        # also transformers' Trainer cannot handle additional dimensions
        batch_generated_token_ids = batch_generated_token_ids.squeeze(1)

        # we return a dummy label so that the transformers' Trainer actually runs evaluation # TODO maybe use empty tensor for less overhead?
        return (
            loss,
            batch_generated_token_ids,
            pred_probs,
        )

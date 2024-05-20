import logging
import os
import re
from glob import glob

from datasets import Dataset

from data.qg.data import ConsistencyFilter, HeuristicFilter, d_clean_question
from data.utils.data import get_datasets
from data.utils.utils import expand_path, setup
from utils.utils import fix_openprompt_tokenizers

fix_openprompt_tokenizers()


def main(args):
    # get dataset
    dataset: Dataset = get_datasets(args.dataset, args.cache_dir, concatenate=True)

    # fix dataset if there are still old keys, should not occur anymore
    ## use original context if available
    if "context_original" in dataset.column_names:
        # we also have to map the answers' spans correctly
        assert "offset_context_original" in dataset.column_names
        dataset = dataset.map(
            lambda sample: {
                "answers": {
                    "answer_start": [
                        char_start + sample["offset_context_original"]
                        for char_start in sample["answers"]["answer_start"]
                    ],
                    "text": sample["answers"]["text"],
                }
            },
            batched=False,
            num_proc=args.num_workers,
            desc="Mapping answers to original context",
        )

        dataset = dataset.remove_columns("context")
        dataset = dataset.rename_column("context_original", "context")

    ## remove unnecessary columns
    if "chunk_idx" in dataset.column_names:
        dataset = dataset.remove_columns("chunk_idx")
    if "offset_context_original" in dataset.column_names:
        dataset = dataset.remove_columns("offset_context_original")

    if args.clean_questions:
        logging.info("Cleaning questions")
        dataset = dataset.map(
            d_clean_question,
            batched=False,
            num_proc=args.num_workers,
            desc="Cleaning questions",
        )

    if args.filter_samples not in ["heuristic", "consistency", "heuristic+consistency"]:
        raise NotImplementedError()

    logging.info("Filtering questions: %s", args.filter_samples)
    num_samples = len(dataset)

    # set up filter
    if (
        args.filter_samples == "heuristic"
        or args.filter_samples == "heuristic+consistency"
    ):
        dataset = HeuristicFilter().filter_dataset(dataset)
    if (
        args.filter_samples == "consistency"
        or args.filter_samples == "heuristic+consistency"
    ):
        ## get checkpoint
        # detect if there are several checkpoingts in dir and select correct one
        filter_model_checkpoint = args.filter_model
        if filter_model_checkpoint is not None:
            checkpoints = glob(os.path.join(filter_model_checkpoint, "checkpoint-*"))
            if checkpoints:
                # detect best checkpoint, i.e. the one with least steps (assuming that only best and most recent model are saved)
                best_checkpoint_step = min(
                    int(re.search(r"checkpoint-(\d+)", checkpoint).group(1))
                    for checkpoint in checkpoints
                )
                filter_model_checkpoint = os.path.join(
                    filter_model_checkpoint, f"checkpoint-{best_checkpoint_step}"
                )

        dataset = ConsistencyFilter(
            checkpoint_path=filter_model_checkpoint,
            num_workers=args.num_workers,
            batch_size=args.filter_batch_size,
            template_str=("templates_rc_en.txt", 13),
        ).filter_dataset(dataset)

    logging.info(
        f"{len(dataset)} samples remain after filtering ({num_samples-len(dataset)} discarded)"
    )

    # save filtered dataset
    logging.info(f"Saving dataset to {args.output_dir}")
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":

    def _main():
        import configparser
        from argparse import ArgumentParser

        # some parameters might be in ini file
        config = configparser.ConfigParser()
        config.read("config.ini")

        ### parent parsers

        # general parser
        parser = ArgumentParser(
            description="A parser including general arguments", add_help=True
        )
        parser.add_argument(
            "--cache_dir",
            type=expand_path,
            default=config.get("Paths", "cache_dir", fallback="~/.cache"),
            help="the cache directory",
        )
        parser.add_argument(
            "-pa",
            "--pre_allocation",
            action="store_true",
            help="Enable pre-allocation of GPU memory (this will allocate 95%% of memory)",
        )
        parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
        parser.add_argument(
            "--log_level", default="passive", type=str, help="Set the logging level"
        )

        # data related arguments
        parser.add_argument(
            "--dataset",
            required=True,
            type=str,
            nargs="+",
            help="The dataset which will be filtered",
        )
        parser.add_argument(
            "--output_dir",
            required=True,
            type=str,
            help="The directory where the filtered dataset will be stored",
        )
        parser.add_argument(
            "--num_workers", type=int, help="The number of concurrent processes"
        )
        parser.add_argument(
            "--clean_questions",
            action="store_true",
            help="Clean questions (using handcrafted rules) for training data",
        )
        parser.add_argument(
            "--filter_samples",
            type=str,
            choices=["heuristic", "consistency", "heuristic+consistency"],
            help="Filter training data",
        )
        parser.add_argument(
            "--filter_model",
            type=str,
            help="The path for the model used in filtering (if applicable)",
        )
        parser.add_argument(
            "--filter_batch_size",
            type=int,
            help="The batch size in case of consistency filtering",
        )

        args = parser.parse_args()

        # do setup before any logging to make sure that no default handler is created
        setup(args.log_level, args.pre_allocation and not args.no_cuda)

        main(args)

    _main()

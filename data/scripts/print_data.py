import argparse
import bisect
import configparser
import logging
from operator import itemgetter

from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer

from data.utils.data import expand_answers, get_datasets, unpack_samples
from data.utils.utils import check_positive, expand_path, select_unique

logger = logging.getLogger(__name__)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


if __name__ == "__main__":
    # some parameters might be in ini file
    config = configparser.ConfigParser()
    config.read("config.ini")

    parser = argparse.ArgumentParser()
    action_dataset = parser.add_argument("dataset", nargs="+")
    parser.add_argument(
        "--exclude_dataset",
        nargs="+",
        required=False,
        metavar="dataset",
        help="the dataset(s) for which the contexts are excluded",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        help="the tokenizer used to discard samples with less than 100 tokens",
    )
    parser.add_argument(
        "--skip_context_length_below",
        default=0,
        type=int,
        help="The minimum context length in tokens (contexts with less tokens will be discarded); default is 0 (disabled)",
    )
    parser.add_argument(
        "--unique",
        metavar="column",
        type=str,
        help="The minimum context length in tokens (contexts with less tokens will be discarded); default is 0 (disabled)",
    )
    parser.add_argument("--num_workers", type=int, default=1, required=False)
    parser.add_argument(
        "--cache_dir",
        type=expand_path,
        default=config.get("Paths", "cache_dir", fallback="~/.cache"),
        help="the cache directory",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("-n", "--num_samples", type=check_positive)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    dataset = get_datasets(
        args.dataset,
        args.cache_dir,
        concatenate=True,
        unpack_fn=unpack_samples,
        shuffle_seed=args.seed,
    )
    if isinstance(dataset, DatasetDict):
        raise argparse.ArgumentError(
            action_dataset,
            f"You have to specify a split using `:split` notation. Available splits are: \n{dataset}",
        )
    # expand answers if needed
    dataset = expand_answers(dataset, False, num_processes=args.num_workers)

    if args.unique is not None:
        logger.info(f"Selecting samples with unique values of column '{args.unique}")
        dataset = select_unique(dataset, args.unique)

    if args.exclude_dataset:
        # exclude contexts for generation
        logger.info(f"Excluding contexts from specified data")
        exclude_dataset = get_datasets(
            args.exclude_dataset, args.cache_dir, concatenate=True
        )
        exclude_contexts = exclude_dataset.flatten_indices().unique("context")
        dataset = dataset.filter(
            lambda x: x["context"] not in exclude_contexts,
            num_proc=args.num_workers,
            load_from_cache_file=False,
        )

    if args.skip_context_length_below:
        assert args.tokenizer is not None
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, cache_dir=args.cache_dir
        )
        logger.info(
            f"Discarding documents with less than {args.skip_context_length_below} tokens"
        )
        dataset = dataset.filter(
            lambda x: args.skip_context_length_below
            <= len(tokenizer.tokenize(x["context"], add_special_tokens=False)),
            num_proc=args.num_workers,
            load_from_cache_file=False,
        )

    print(dataset)
    if args.interactive:
        input()
    gpt2_tokenizer = AutoTokenizer.from_pretrained(
        "gpt2-medium", cache_dir=args.cache_dir
    )

    if "questions" in dataset.column_names:
        dataset = unpack_samples(dataset)

    if args.no_color:
        mark_answer_start = "<a>"
        mark_answer_end = "</a>"
    else:
        mark_answer_start = bcolors.FAIL
        mark_answer_end = bcolors.ENDC

    for idx, sample in enumerate(dataset):
        if args.num_samples is not None and idx >= args.num_samples:
            break
        context = sample["context"]
        print(
            "==================================================================================================================================="
        )
        print(f"Context for sample with id {sample['id']}:")
        if "answers" in sample:
            answers_text_list = (
                [sample["answers"]["text"]]
                if isinstance(sample["answers"]["text"][0], str)
                else sample["answers"]["text"]
            )
            if "answer_start" in sample["answers"]:
                ## highlight answer in span
                # sort answers according to their span start so that we can construct the context with the answers highlighted
                # NOTE this is not perfect since highlighting might end too early if one answer is a subspan of another answer such that they do not have the last character in common
                answers_start_list = (
                    [sample["answers"]["answer_start"]]
                    if isinstance(sample["answers"]["answer_start"][0], int)
                    else sample["answers"]["answer_start"]
                )
                answers_start = [
                    answer_start
                    for _answer_start in answers_start_list
                    for answer_start in _answer_start
                ]
                answers_text = [
                    answer_text
                    for _answer_text in answers_text_list
                    for answer_text in _answer_text
                ]
                answers = sorted(
                    set(zip(answers_start, answers_text)), key=itemgetter(0)
                )
                offset = [0]
                insert_positions = []
                for char_start, text in answers:
                    char_start_offset = offset[
                        bisect.bisect(insert_positions, char_start)
                    ]
                    char_end = char_start + len(text)
                    char_end_offset = offset[
                        bisect.bisect(insert_positions, char_end + char_start_offset)
                    ]
                    context = (
                        context[: char_start + char_start_offset]
                        + mark_answer_start
                        + context[
                            char_start + char_start_offset : char_end + char_end_offset
                        ]
                        + mark_answer_end
                        + context[char_end + char_end_offset :]
                    )
                    # extend insert positions and offset
                    insert_positions.append(char_start)
                    insert_positions.append(char_end + len(mark_answer_start))
                    offset.append(offset[-1] + len(mark_answer_start))
                    offset.append(offset[-1] + len(mark_answer_end))
            print(context)
            print(
                "---------------------------------------------------------------------"
            )
            print(f"Question: '{sample['question']}'")
            print(
                f"Answers (occurences): {[f'{list_answer_text[0]} ({len(list_answer_text)})' for list_answer_text in answers_text_list]}"
            )
        else:
            print(context)
        if args.interactive:
            input()

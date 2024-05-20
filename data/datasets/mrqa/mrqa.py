"""Modified MRQA 2019 Shared task dataset."""

from dataclasses import replace

import datasets
from datasets import DownloadMode
from datasets.load import dataset_module_factory, import_main_class


class MrqaConfig(datasets.BuilderConfig):
    """BuilderConfig for Mrqa"""

    def __init__(self, subset: str = None, **kwargs):
        """

        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(MrqaConfig, self).__init__(**kwargs)
        self.subset = subset


download_mode = DownloadMode.REUSE_DATASET_IF_EXISTS
dataset_module = dataset_module_factory(
    "mrqa",
    download_mode=download_mode,
)
# Get dataset builder class from the processing script
builder_cls = import_main_class(dataset_module.module_path)


class Mrqa(builder_cls):
    """Modified MRQA 2019 Shared task dataset."""

    VERSION = datasets.Version("1.0.0")
    # BUILDER_CONFIG_CLASS = MrqaConfig

    BUILDER_CONFIGS = [
        MrqaConfig(name="squad", subset="SQuAD"),
        MrqaConfig(name="news", subset="NewsQA"),
        MrqaConfig(name="trivia", subset="TriviaQA"),
        MrqaConfig(name="search", subset="SearchQA"),
        MrqaConfig(name="hotpot", subset="HotpotQA"),
        MrqaConfig(name="nq", subset="NaturalQuestions"),
        MrqaConfig(name="bioasq", subset="BioASQ"),
        MrqaConfig(name="drop", subset="DROP"),
        MrqaConfig(name="duorc", subset="DuoRC"),
        MrqaConfig(name="race", subset="RACE"),
        MrqaConfig(name="re", subset="RelationExtraction"),
        MrqaConfig(name="textbook", subset="TextbookQA"),
    ]

    def _info(self):
        features = {
            "context": datasets.Value("string"),
            "id": datasets.Value("string"),
            "question": datasets.Value("string"),
            "answers": datasets.Sequence(
                {
                    "text": datasets.Value("string"),
                    "answer_start": datasets.Value("int32"),
                }
            ),
        }
        if self.config.subset is None:
            features["subset"] = datasets.Value("string")

        return replace(
            super()._info(),
            # format is adapted to SQuAD format
            features=datasets.Features(features),
        )

    def _generate_examples(self, filepaths_dict, split):
        """Yields examples."""
        for _id, sample in super()._generate_examples(filepaths_dict, split):
            if self.config.subset is not None:
                if sample["subset"] != self.config.subset:
                    continue
                del sample["subset"]
            sample["id"] = sample.pop("qid")
            del sample["context_tokens"]
            del sample["question_tokens"]
            answers = sample.pop("detected_answers")
            answers_text, answers_start_char = list(
                zip(
                    *[
                        (answer["text"], char_span["start"])
                        for answer_idx, answer in enumerate(answers)
                        for char_span in answers[answer_idx]["char_spans"]
                    ]
                )
            )
            sample["answers"] = {
                "text": list(answers_text),
                "answer_start": list(answers_start_char),
            }

            yield sample["id"], sample

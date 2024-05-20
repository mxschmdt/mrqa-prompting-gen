# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MRQA 2019 Shared task dataset."""

import json
from typing import List

import datasets

_CITATION = """\
@inproceedings{fisch2019mrqa,
    title={{MRQA} 2019 Shared Task: Evaluating Generalization in Reading Comprehension},
    author={Adam Fisch and Alon Talmor and Robin Jia and Minjoon Seo and Eunsol Choi and Danqi Chen},
    booktitle={Proceedings of 2nd Machine Reading for Reading Comprehension (MRQA) Workshop at EMNLP},
    year={2019},
}
"""

_DESCRIPTION = """\
The MRQA 2019 Shared Task focuses on generalization in question answering.
An effective question answering system should do more than merely
interpolate from the training set to answer test examples drawn
from the same distribution: it should also be able to extrapolate
to out-of-distribution examples — a significantly harder challenge.

The dataset is a collection of 18 existing QA dataset (carefully selected
subset of them) and converted to the same format (SQuAD format). Among
these 18 datasets, six datasets were made available for training,
six datasets were made available for development, and the final six
for testing. The dataset is released as part of the MRQA 2019 Shared Task.
"""

_HOMEPAGE = "https://mrqa.github.io/2019/shared.html"

_LICENSE = "Unknwon"

_URLs = {
    # Train+Dev in-domain sub-datasets
    "squad": {
        "train+SQuAD": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz",
        "validation+SQuAD": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz",
    },
    "newsqa": {
        "train+NewsQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz",
        "validation+NewsQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz",
    },
    "triviaqa": {
        "train+TriviaQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz",
        "validation+TriviaQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz",
    },
    "searchqa": {
        "train+SearchQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SearchQA.jsonl.gz",
        "validation+SearchQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz",
    },
    "hotpotqa": {
        "train+HotpotQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz",
        "validation+HotpotQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz",
    },
    "naturalquestions": {
        "train+NaturalQuestions": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz",
        "validation+NaturalQuestions": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz",
    },
    # Dev out-of-domain sub-datasets
    "bioasq": {
        "validation+BioASQ": "http://participants-area.bioasq.org/MRQA2019/",  # BioASQ.jsonl.gz
    },
    "drop": {
        "validation+DROP": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DROP.jsonl.gz",
    },
    "duorc": {
        "validation+DuoRC": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/DuoRC.ParaphraseRC.jsonl.gz",
    },
    "race": {
        "validation+RACE": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RACE.jsonl.gz",
    },
    "relationextraction": {
        "validation+RelationExtraction": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/RelationExtraction.jsonl.gz",
    },
    "textbookqa": {
        "validation+TextbookQA": "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TextbookQA.jsonl.gz",
    },
}


class Mrqa(datasets.GeneratorBasedBuilder):
    """MRQA 2019 Shared task dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            # SQuAD format
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.Sequence(
                        {
                            "text": datasets.Sequence(datasets.Value("string")),
                            "answer_start": datasets.Sequence(datasets.Value("int32")),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    @staticmethod
    def get_split_and_subset_from_header(filepaths: List[str]):
        split, subset = None, None
        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                header = json.loads(next(f))
            while isinstance(header, list):
                # sometimes header is a list, sometimes even nested, sometimes not at all
                header = header[0]
            split_from_header = (
                header["header"]["mrqa_split"]
                if "mrqa_split" in header["header"]
                else header["header"]["split"]
            )
            subset_from_header = header["header"]["dataset"]
            if split is None:
                split = split_from_header
            elif split != split_from_header:
                raise Exception("Provided files do not match same split")
            if subset is None:
                subset = subset_from_header
            elif subset != subset_from_header:
                raise Exception("Provided files do not match same subset")
        if split == "dev":
            # convert to datasets convention
            split = "validation"
        return split, subset

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_files is not None:
            files = dl_manager.download_and_extract(self.config.data_files)
            split_generators = []
            for _, filepaths in files.items():
                split, subset = self.get_split_and_subset_from_header(filepaths)
                split_generators.append(
                    datasets.SplitGenerator(
                        name=datasets.NamedSplit(split),
                        gen_kwargs={
                            "filepaths": filepaths,
                            "subset": subset,
                        },
                    )
                )
            return split_generators
        else:
            files = dl_manager.download_and_extract(_URLs[self.config.name])
            split_generators = []
            for split_and_subset, filepath in files.items():
                split, subset = split_and_subset.split("+")
                split_from_header, subset_from_header = (
                    self.get_split_and_subset_from_header([filepath])
                )
                assert (
                    split == split_from_header and subset in subset_from_header
                ), f"Consistendy check for split (given: {split} - header: {split_from_header}) or subset (given: {subset} - header: {subset_from_header}) failed"
                split_generators.append(
                    datasets.SplitGenerator(
                        name=datasets.NamedSplit(split),
                        gen_kwargs={
                            "filepaths": [filepath],
                            "subset": subset,
                        },
                    )
                )
            return split_generators

    def _generate_examples(self, filepaths, subset):
        """Yields examples."""
        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                # first row is header -> skip
                next(f)

                for row in f:
                    paragraph = json.loads(row)
                    context = paragraph["context"]
                    for qa in paragraph["qas"]:
                        qid = qa["qid"]
                        question = qa["question"].strip()
                        answers_start = []
                        answers_text = []
                        # instead of mixing answers and their occurences keep them separate, i.e. a list of list of spans (one list of spans per answer)
                        for detect_ans in qa["detected_answers"]:
                            cur_answers_start = []
                            cur_answers_text = []
                            for char_span in detect_ans["char_spans"]:
                                # answers_text.append(detect_ans["text"].strip())
                                # detected answers text are wrong sometimes, rely on char span instead
                                cur_answers_text.append(
                                    context[char_span[0] : char_span[1] + 1].strip()
                                )
                                cur_answers_start.append(char_span[0])
                            answers_start.append(cur_answers_start)
                            answers_text.append(cur_answers_text)
                        yield f"{subset}'_'{qid}", {
                            "id": qid,
                            "context": context,
                            "question": question,
                            "answers": {
                                "answer_start": answers_start,
                                "text": answers_text,
                            },
                        }

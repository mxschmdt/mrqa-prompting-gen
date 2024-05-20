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
"""ck-12 lessons dataset."""

import json
import logging
from collections import defaultdict
from functools import partial
from typing import Dict

import datasets
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from requests.models import HTTPError

_DESCRIPTION = """\
Lessons from ck12.org.
"""

_HOMEPAGE = "https://www.ck12.org/"

_SUBJECTS = {
    "physical_science": "ck-12-middle-school-physical-science-flexbook-2.0",
    "earth_science": "ck-12-middle-school-earth-science-flexbook-2.0",
    "life_science": "ck-12-middle-school-life-science-2.0",
    # 'biology': 'ck-12-biology-flexbook-2.0',
    # 'chemistry': 'ck-12-chemistry-flexbook-2.0',
    # 'physics': 'ck-12-physics-flexbook-2.0',
}

_URL = "https://flexbooks.ck12.org/flx/artifact/artifactType=cbook&artifactHandle={subject}&artifactCreator=ck12editor/descendant/{lesson}.{section}?includeTagTerms=False&includeBrowseTerms=False&includeSearchTerms=False&includeAuthors=False&includeResources=False&includeContent=True&includeProcessedContent=True&includeDescendantRelatedArtifacts=False&includeDomainCollectionContexts=True&includeRevisionStandards=True&includeRevisionStandardGrades=False&includeHandles=False&includeVocabularyInterlinks=False&excludeInterlinksForInteractiveContent=True&includeSummaryDetails=False&includeIntentsInfo=False&artifactTypesExcludedForTraversal=&considerSummativeTestsForTraversal=False"

_IGNORE_SECTIONS = ["summary", "review", "resources", "explore more"]


logger = logging.getLogger(__name__)


def download_lesson(subject, download_fn):
    logger.info("Downloading lesson %s", subject)
    # we start with section 1.1
    lesson, lesson_section = 1, 1
    while True:
        try:
            yield lesson, lesson_section, download_fn(
                _URL.format(subject=subject, lesson=lesson, section=lesson_section)
            )
        except HTTPError as e:
            # status 400 is returned if requested section does not exist
            if e.response.status_code == 400:
                # section does not exist
                if lesson_section == 1:
                    # even the first section does not exist -> return
                    return
                # otherwise move to next lesson
                lesson += 1
                lesson_section = 1
                continue
            else:
                raise

        # increment section
        lesson_section += 1


def scrape_lessons(files: Dict, individual_content: bool = False):
    # generator yielding contents from request responses
    for subject, subject_files in files.items():
        for lesson, section, filename in subject_files:
            with open(filename, "r") as f:
                response = json.load(f)
            revisions = response["response"]["artifact"]["descendantArtifact"][
                "revisions"
            ]
            _lesson, _section = map(
                int,
                response["response"]["artifact"]["descendantArtifact"][
                    "identifier"
                ].split("."),
            )

            # don't know yet whether there might be multiple revisions
            assert len(revisions) == 1
            content = revisions[0]["contentRevision"]["rawContents"]
            soup = BeautifulSoup(content, "html.parser")
            concepts = soup.find_all(class_="x-ck12-data-concept")

            # don't know yet whether there are multiple concepts per lesson
            if not concepts:
                # some content cannot be extracted by class as it is formatted differently
                # -> skip in the meanwhile
                logger.warning(
                    "Skipping section %s.%s from %s since data could not be extracted.",
                    lesson,
                    section,
                    subject,
                )
                continue
            assert len(concepts) == 1, (_lesson, _section, concepts)

            texts = []
            num_contents = 0
            for tag in concepts[0].find_all(["p", "ul", "ol"]):
                tag: Tag
                text = tag.text.replace("\xa0", " ").strip()
                paragraph_title = tag.find_previous("h3")
                # skip if there is no text, if text is Q&A (starts with 'Q:' or 'A:'), if we are in one of the sections which should be ignored or if tag is containd within image postcard (probably a figure subtitle)
                if (
                    text
                    and text[:2] not in ["Q:", "A:"]
                    and (
                        paragraph_title is None
                        or paragraph_title.text.lower() not in _IGNORE_SECTIONS
                    )
                    and (
                        tag.parent.get(key="class") is None
                        or "x-ck12-img-postcard" not in tag.parent.get(key="class")
                    )
                ):
                    if individual_content:
                        id_ = f"{subject}-{lesson}.{section}#{num_contents}"
                        num_contents += 1
                        yield id_, text
                    else:
                        texts.append(text)

            if not individual_content:
                id_ = f"{subject}-{lesson}.{section}"
                yield id_, " ".join(texts)


class Ck12Config(datasets.BuilderConfig):
    """BuilderConfig for Ck12."""

    def __init__(self, subjects, individual_content=False, **kwargs):
        """BuilderConfig for Ck12.

        Args:
        subjects: List[], the subjects which will be downloaded and processed,
        **kwargs: keyword arguments forwarded to super.
        """
        super(Ck12Config, self).__init__(**kwargs)
        self.subjects = subjects
        self.individual_content = individual_content == "True"


class Ck12(datasets.GeneratorBasedBuilder):
    """ck12 abstracts dataset."""

    BUILDER_CONFIG_CLASS = Ck12Config
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            # SQuAD format
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""

        def download_json(url, path):
            response = requests.get(url)
            response.raise_for_status()
            with open(path, "w") as f:
                json.dump(response.json(), f)

        cached_urls = defaultdict(list)

        # depending von the config download the files
        if self.config.subjects == "all" or "all" in self.config.subjects:
            subjects = _SUBJECTS.values()
        else:
            subjects = [_SUBJECTS[subject] for subject in self.config.subjects]
        for subject in subjects:
            cached_urls[subject].extend(
                download_lesson(
                    subject,
                    partial(dl_manager.download_custom, custom_download=download_json),
                )
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split("lessons"),
                gen_kwargs={
                    "files": cached_urls,
                },
            ),
        ]

    def _generate_examples(self, files: Dict):
        """Yields examples."""

        for id_, content in scrape_lessons(files, self.config.individual_content):
            yield id_, {
                "id": id_,
                "context": content,
            }

"""Utilities for running Massive Multitask Language Understanding (MMLU) benchmark."""

from collections.abc import Sequence, Set
import csv
import logging
from pathlib import Path
from random import sample
import shutil
import tarfile
import tempfile
from typing import NamedTuple

from IPython.display import display
from pandas import DataFrame
import requests
from tqdm.auto import tqdm

import llama_jax as ll
from llama_jax.chat import Thread
from llama_jax.tools import default_arg

__all__ = [
    "OPTIONS",
    "Answer",
    "Answers",
    "Categories",
    "Question",
    "Questions",
    "display_questions",
    "download_dataset",
    "generate_prompt",
    "load_dataset",
    "select_question",
]

logger = logging.getLogger(__name__)


class Question(NamedTuple):
    """An MMLU question."""

    qid: int

    category: str

    question: str

    A: str

    B: str

    C: str

    D: str

    answer: str


Questions = Sequence[Question]

Categories = Set[str]

OPTIONS = ("A", "B", "C", "D")


class Answer(NamedTuple):
    """An answer to MMLU question."""

    qid: int

    expected: str

    actual: str

    scores: dict[str, float]

    correct: bool


Answers = Sequence[Answer]


class Dataset(NamedTuple):
    """MMLU dataset."""

    questions: Questions

    examples: Questions

    categories: Categories


_mmlu_dataset_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"


def download_dataset(output_path: Path):
    """Download MMLU dataset to output_path."""
    # Check if it exists already
    if output_path.exists():
        logger.info(f"Dataset {output_path.name} exists. Skipping download.")
        return

    work_dir = tempfile.TemporaryDirectory()
    work_path = Path(work_dir.name)

    # Download tarball
    response = requests.get(_mmlu_dataset_url, stream=True)
    total = int(response.headers["Content-Length"])

    with tqdm(total=total) as progress, tempfile.NamedTemporaryFile() as tarball:
        for data in response.iter_content(chunk_size=5 * 1024 * 1024):
            tarball.write(data)
            progress.update(n=len(data))

        with tarfile.open(tarball.name) as tf:
            tf.extractall(work_path, filter="data")

        shutil.move(work_path / "data", output_path)


def load_dataset(dataset_path: Path) -> Dataset:
    """Load MMLU examples and questions."""
    executor = ll.tools.executor()

    def load_data_file(path: Path) -> Questions:
        # Infer category from file name: x_y_z_test.csv -> x y z
        category = " ".join(path.stem.split("_")[0:-1])

        with open(path, mode="r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            questions = tuple(Question(i, category, *row) for i, row in enumerate(reader))

        return questions

    def load_segment(segment: str) -> Questions:
        # Sort paths to ensure consistent order
        paths = sorted(path for path in dataset_path.glob(f"{segment}/*.csv"))

        # Load data files in parallel
        futures = [executor.submit(load_data_file, path) for path in paths]

        # Collect results
        collected = ()
        for future in futures:
            collected += future.result()

        # Reassign ids
        questions = ()
        for i, question in enumerate(collected):
            questions += (Question(i, *question[1:]),)

        return questions

    questions = load_segment("test")
    examples = load_segment("dev")
    categories = {q.category for q in questions}

    return Dataset(questions=questions, examples=examples, categories=categories)


def select_question(questions: Questions, *, qid: int | None = None, question: str | None = None) -> Question | None:
    """Select question by qid or question text."""
    if qid is not None:
        return next((q for q in questions if q.qid == qid), None)

    if question is None:
        raise ValueError("Must specify either qid or question")

    return next((q for q in questions if q.question == question), None)


def display_questions(questions: Questions, n: int | None = None):
    """Render random sample of questions as a table."""
    # Defaults
    n = default_arg(n, 5)

    # Randomly select sample
    selected = questions if len(questions) <= n else sample(questions, n)

    display(DataFrame(selected))


def generate_prompt(
    question: Question,
    *,
    n_shots: int,
    examples: Questions | None = None,
) -> Thread:
    """Generate prompt for specified question."""
    # Validate
    if n_shots < 0 or n_shots > 5:  # noqa: PLR2004
        raise ValueError("n_shots must be between 0 and 5")

    if n_shots > 0 and examples is None:
        raise ValueError("n_shots specified without examples")

    selected_examples = []
    if n_shots > 0:
        # Select examples for category
        selected_examples = [e for e in examples if e.category == question.category]

        # Deterministically select n_shots if specified
        selected_examples = selected_examples[:n_shots]

    messages = []

    # System message
    messages.append({
        "role": "system",
        "content": (
            f"You are an expert answering multiple choice questions about {question.category}. Each question "
            f"has 4 options: A, B, C, D. The prompt will contain {n_shots} example questions with correct answers "
            f"followed by a final question. Your job is to answer the final question. Your answer MUST be one "
            f"of: A, B, C, D."
        ),
    })

    # Examples
    for row in selected_examples:
        # Question
        messages.append(  # noqa: FURB113
            {
                "role": "user",
                "content": f"{row.question}\n\nA) {row.A}\nB) {row.B}\nC) {row.C}\nD) {row.D}\n",
            },
        )

        # Answer
        messages.append(
            {
                "role": "assistant",
                "content": f"{row.answer}",
            },
        )

    # Question
    messages.append(
        {
            "role": "user",
            "content": f"{question.question}\n\nA) {question.A}\nB) {question.B}\nC) {question.C}\nD) {question.D}\n",
        },
    )

    thread = {"messages": messages}

    return ll.chat.load_threads(thread)[0]

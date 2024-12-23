from pathlib import Path
import pytest

import llama_jax as ll
from llama_jax.benchmarks.mmlu import (
    OPTIONS,
    evaluate_generator,
    generate_prompt,
    load_dataset,
    select_question,
)


def test_load_dataset(mmlu_dataset_path: Path):
    #
    # Whens
    #

    # I load dataset
    dataset = load_dataset(mmlu_dataset_path)

    #
    # Thens
    #

    # There should 14,042 questions
    assert len(dataset.questions) == 14042

    # There should 57 categories
    assert len(dataset.categories) == 57

    # There should 5 examples per category
    for category in dataset.categories:
        assert len([e for e in dataset.examples if e.category == category]) == 5

    #
    # Whens
    #

    # I query question by qid
    question0 = select_question(dataset.questions, qid=120)

    #
    # Thens
    #

    # question should be
    assert question0.question == "Where is the sinoatrial node located?"
    #
    # Whens
    #

    # I query question by question
    question1 = select_question(dataset.questions, question=question0.question)

    #
    # Thens
    #

    # question1 should match question0
    assert question1 == question0


def test_prompt_zero_shot(mmlu_dataset_path: Path):
    #
    # Givens
    #

    # I loaded dataset
    dataset = load_dataset(mmlu_dataset_path)

    # I looked up question 7779
    question = select_question(dataset.questions, qid=7779)

    #
    # Whens
    #

    # I generate zero-shot prompt
    messages = generate_prompt(question, n_shots=0)

    #
    # Thens
    #

    # messages includes system message
    assert messages[0].role == "system"

    # messages includes user message
    assert messages[-1].role == "user"


@pytest.mark.wip
def test_generate_answers(mmlu_dataset_path: Path):
    #
    # Givens
    #

    # I loaded dataset and looked up question 7779
    dataset = load_dataset(mmlu_dataset_path)
    question = select_question(dataset.questions, qid=7779)

    # I loaded config for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")

    # I initialized a mmlu generator
    generator = ll.benchmarks.mmlu.generator(config)

    #
    # Whens
    #

    # I generate answer to question
    answer = next(generator([question]))

    #
    # Thens
    #

    # answer should be populated
    assert answer.qid == question.qid
    assert answer.expected == "B"
    assert isinstance(answer.actual, str)
    assert all(option in answer.scores for option in OPTIONS)
    assert isinstance(answer.correct, bool)


@pytest.mark.wip
def test_evaluate(mmlu_dataset_path: Path):
    #
    # Givens
    #

    # I loaded dataset and looked up question 7779
    dataset = load_dataset(mmlu_dataset_path)
    question = select_question(dataset.questions, qid=7779)

    # I loaded config for 3.2 3B checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B")

    # I initialized a mmlu generator
    generator = ll.benchmarks.mmlu.generator(config)

    #
    # Whens
    #

    # I evaluate generator
    score = evaluate_generator(generator, questions=[question])

    #
    # Thens
    #

    # score should be populated
    assert 0.0 <= score <= 100.0

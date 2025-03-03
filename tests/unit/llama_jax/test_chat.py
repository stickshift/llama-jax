import pytest
from tqdm.auto import tqdm

import llama_jax as ll


@pytest.mark.wip
def test_single_thread():
    #
    # Givens
    #

    # Boston prompt
    thread = {
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of Massachusetts? Answer in one word.",
            }
        ],
    }

    # I loaded config for 3.2 3B Instruct checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B-Instruct")

    # Chat generator w/
    #   * token sampling disabled
    generator = ll.chat.generator(config, temperature=0)

    #
    # Whens
    #

    # I generate single response
    event = next(generator(thread))

    #
    # Thens
    #

    # event should have 2 messages and no delta
    assert len(event.thread.messages) == 2
    assert event.delta is None

    # last message should be from assistant
    message = event.thread.messages[-1]
    assert message.role == "assistant"

    # Answer should be Boston
    assert message.content.strip() == "Boston"


@pytest.mark.wip
def test_multi_thread():
    #
    # Givens
    #

    # Capital prompts
    threads = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in one word.",
                }
            ],
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of Massachusetts? Answer in one word.",
                }
            ]
        },
    ]

    # I loaded config for 3.2 3B Instruct checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B-Instruct")

    # Chat generator w/
    #   * token sampling disabled
    generator = ll.chat.generator(config, temperature=0)

    #
    # Whens
    #

    # I generate single response
    events = next(generator(threads))

    #
    # Thens
    #

    # events should have 2 events
    assert len(events) == 2

    # each event should have ...
    for event in events:
        # 2 messages and no delta
        assert len(event.thread.messages) == 2
        assert event.delta is None

        # last message should be from assistant
        message = event.thread.messages[-1]
        assert message.role == "assistant"

    # First answer should be Paris
    assert events[0].thread.messages[-1].content.strip() == "Paris"

    # Second answer should be Boston
    assert events[1].thread.messages[-1].content.strip() == "Boston"


@pytest.mark.wip
def test_single_thread_streaming():
    #
    # Givens
    #

    # Boston prompt
    thread = {
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of Massachusetts? Answer in one word.",
            }
        ],
    }

    # I loaded config for 3.2 3B Instruct checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B-Instruct")

    # Chat generator w/
    #   * token sampling disabled
    #   * streaming
    generator = ll.chat.generator(config, temperature=0, stream=True)

    #
    # Whens
    #

    # I start stream
    it = generator(thread)

    # I wait for next event
    event = next(it)

    #
    # Thens
    #

    # event.delta should be "Boston"
    assert event.delta.content.strip() == "Boston"

    #
    # Whens
    #

    # I wait for next event
    event = next(it)

    #
    # Thens
    #

    # event.delta should be None
    assert event.delta is None


@pytest.mark.wip
def test_multi_thread_streaming():
    #
    # Givens
    #

    # Multiple threads that prompt varying length answers
    threads = [
        {
            "id": "thread0",
            "messages": [
                {
                    "role": "user",
                    "content": "List 3 fruits.",
                }
            ],
        },
        {
            "id": "thread1",
            "messages": [
                {
                    "role": "user",
                    "content": "List 5 music artists.",
                }
            ],
        },
    ]

    # I loaded config for 3.2 3B Instruct checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B-Instruct")

    # Chat generator w/
    #   * token sampling disabled
    #   * streaming
    #   * higher token limit than necessary
    generator = ll.chat.generator(config, temperature=0, stream=True, max_tokens=200)

    # Empty repository to store events
    repository = {
        "thread0": [],
        "thread1": [],
    }

    #
    # Whens
    #

    # I collect all events
    for events in tqdm(generator(threads), desc="Events"):
        for event in events:
            repository[event.thread.id].append(event)

    #
    # Thens
    #

    # repository should have multiple events for both threads
    assert len(repository["thread0"]) > 1
    assert len(repository["thread1"]) > 1

    # thread1 should have more events than thread0
    assert len(repository["thread1"]) > len(repository["thread0"])

from jax import Array

import llama_jax as ll


def test_323b():
    #
    # Givens
    #

    # Boston prompt
    messages = [
        {
            "role": "user",
            "content": "What is the capital of Massachusetts? Answer in one word.",
        }
    ]

    # I loaded config for 3.2 3B Instruct checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B-Instruct")

    # I initialized a chat generator w/ token sampling disabled
    generator = ll.chat.generator(config, temperature=0)

    #
    # Whens
    #

    # I generate single response
    response = next(generator(messages))

    #
    # Thens
    #

    # response should have 2 messages and no delta
    assert len(response.messages) == 2
    assert response.delta is None

    # last message should be from assistant
    message = response.messages[-1]
    assert message.role == "assistant"

    # response should be "Boston"
    assert message.content.strip() == "Boston"


def test_323b_streaming():
    #
    # Givens
    #

    # Boston prompt
    messages = [
        {
            "role": "user",
            "content": "What is the capital of Massachusetts? Answer in one word.",
        }
    ]

    # I loaded config for 3.2 3B Instruct checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B-Instruct")

    # I initialized a streaming chat generator w/ token sampling disabled
    generator = ll.chat.generator(config, temperature=0, stream=True)

    #
    # Whens
    #

    # I start stream
    it = generator(messages)

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

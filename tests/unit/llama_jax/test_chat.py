from jax import random

import llama_jax as ll


def test_323b():
    #
    # Givens
    #

    # rng
    key = random.key(42)

    # I loaded config and parameters for 3.2 3B Instruct checkpoint
    config = ll.checkpoint.load_config("Llama3.2-3B-Instruct")
    params = ll.checkpoint.load_parameters(config)

    # I created a Model
    model = ll.model.create(config, params)

    # Boston prompt
    messages = [
        {
            "role": "user",
            "content": "What is the capital of Massachusetts? Answer in one word.",
        }
    ]

    # I initialized a chat generator w/ token sampling disabled
    key, subkey = random.split(key)
    generator = ll.chat.generator(model, key=subkey, temperature=0)

    #
    # Whens
    #

    # I pose question
    response = generator(messages)

    #
    # Thens
    #

    #
    # Thens
    #

    # last message should be from assistant
    message = response.messages[-1]
    assert message.role == "assistant"

    # response should be "Boston"
    assert message.content.strip() == "Boston"

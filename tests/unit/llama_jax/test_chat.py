import pytest


@pytest.mark.wip
def test_323b_boston():
    #
    # Givens
    #

    # I loaded config for 3.2 3B Instruct checkpoint
    config = load_config("Llama3.2-3B-Instruct")

    # I created a model
    model = create_model(config)

    # Boston prompt
    messages = [{"role": "user", "content": "What is the capital of Massachusetts? Answer in one word."}]

    #
    # Whens
    #

    # I pose question
    response = chat_completion(model, messages=messages)

    #
    # Thens
    #

    # last message should be from assistant
    message = response.messages[-1]
    assert message.role == "assistant"

    # response should be "Boston"
    assert message.content.strip() == "Boston"
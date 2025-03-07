import llama_jax as ll


def test_complete():
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

    # I created chat session
    session = ll.chat.session()

    #
    # Whens
    #

    # I generate response
    response = ll.chat.complete(session, messages=messages)

    #
    # Thens
    #

    # response should have 2 messages
    assert len(response.messages) == 2

    # last message should be from assistant
    assert response.messages[-1].role == "assistant"

    # Answer should be Boston
    assert response.messages[-1].content.strip() == "Boston"


def test_complete_stream():
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

    # I created chat session
    session = ll.chat.session()

    #
    # Whens
    #

    # I collect event stream
    events = list(ll.chat.complete(session, messages=messages, stream=True))

    #
    # Thens
    #

    # there should be 2 events
    assert len(events) == 2

    # first event delta should be "Boston"
    assert events[0].delta.strip() == "Boston"

    # second event delta should be None
    assert events[1].delta is None

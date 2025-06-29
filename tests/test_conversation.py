import pytest

from pai.models import Conversation, Turn


@pytest.fixture
def conversation() -> Conversation:
    """Provides a fresh Conversation instance for each test."""
    return Conversation()


def test_add_and_pop_system_prompt(conversation: Conversation):
    """Test that system prompts can be added and removed from the stack."""
    assert conversation.get_system_prompts() == []
    assert conversation._get_combined_system_prompt() is None

    conversation.add_system_prompt("You are a helpful assistant.")
    assert conversation.get_system_prompts() == ["You are a helpful assistant."]
    assert conversation._get_combined_system_prompt() == "You are a helpful assistant."
    assert conversation.session_token_count > 0

    conversation.add_system_prompt("Speak like a pirate.")
    assert conversation.get_system_prompts() == [
        "You are a helpful assistant.",
        "Speak like a pirate.",
    ]
    assert "---" in conversation._get_combined_system_prompt()
    prev_token_count = conversation.session_token_count

    popped = conversation.pop_system_prompt()
    assert popped == "Speak like a pirate."
    assert conversation.get_system_prompts() == ["You are a helpful assistant."]
    assert conversation.session_token_count < prev_token_count

    conversation.clear_system_prompts()
    assert conversation.get_system_prompts() == []
    assert conversation.session_token_count == 0


def test_add_simple_turn(conversation: Conversation):
    """Test adding a single, non-agentic turn."""
    turn = Turn(
        request_data={
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        },
        response_data={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hi there!",
                    }
                }
            ]
        },
        assistant_message="Hi there!",
    )
    conversation.add_turn(turn)

    assert len(conversation.turns) == 1
    messages = conversation.get_history()
    assert len(messages) == 2
    assert messages[0] == {"role": "user", "content": "Hello"}
    assert messages[1] == {"role": "assistant", "content": "Hi there!"}
    assert conversation.session_token_count > 0


def test_add_agentic_turn_with_tool_use(conversation: Conversation):
    """Test that an agentic turn with tool use is correctly logged."""
    # This simulates the full message history that a protocol adapter would
    # place in the request_data for logging purposes.
    agentic_request_data = {
        "messages": [
            {"role": "user", "content": "What is the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "Tokyo"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "get_current_weather",
                "content": '{"location": "Tokyo", "temperature": "15", "condition": "Cloudy"}',
            },
        ]
    }
    # The final response from the model after using the tool.
    final_response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The weather in Tokyo is 15°C and Cloudy.",
                }
            }
        ]
    }
    turn = Turn(
        request_data=agentic_request_data,
        response_data=final_response_data,
        assistant_message="The weather in Tokyo is 15°C and Cloudy.",
    )
    conversation.add_turn(turn)

    assert len(conversation.turns) == 1
    # Check that the internal message history for the *next* turn is correct.
    messages = conversation.get_history()
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert "tool_calls" in messages[1]
    assert messages[2]["role"] == "tool"
    assert messages[3]["role"] == "assistant"
    assert messages[3]["content"] == "The weather in Tokyo is 15°C and Cloudy."


def test_get_messages_for_next_turn(conversation: Conversation):
    """Test that system prompts and history are correctly combined."""
    conversation.add_system_prompt("Be concise.")
    conversation._messages = [{"role": "user", "content": "Hello"}]

    next_messages = conversation.get_messages_for_next_turn("How are you?")
    assert len(next_messages) == 3
    assert next_messages[0] == {"role": "system", "content": "Be concise."}
    assert next_messages[1] == {"role": "user", "content": "Hello"}
    assert next_messages[2] == {"role": "user", "content": "How are you?"}


def test_clear_preserves_system_prompt(conversation: Conversation):
    """Test that /clear clears turns/messages but not system prompts."""
    conversation.add_system_prompt("You are a helpful assistant.")
    turn = Turn(
        request_data={"messages": [{"role": "user", "content": "Hello"}]},
        response_data={
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}}]
        },
    )
    conversation.add_turn(turn)

    token_count_before = conversation.session_token_count
    assert len(conversation.turns) == 1
    assert len(conversation.get_history()) == 2

    conversation.clear()

    assert len(conversation.turns) == 0
    assert len(conversation.get_history()) == 0
    assert conversation.get_system_prompts() == ["You are a helpful assistant."]
    # The token count should be only that of the system prompt.
    assert conversation.session_token_count < token_count_before
    assert conversation.session_token_count > 0


def test_serialization_deserialization(conversation: Conversation):
    """Test that a conversation can be serialized to JSON and back."""
    conversation.add_system_prompt("System prompt")
    turn = Turn(
        request_data={"messages": [{"role": "user", "content": "Test"}]},
        response_data={
            "choices": [{"message": {"role": "assistant", "content": "OK"}}]
        },
        assistant_message="OK",
        participant_name="test_participant",
        model_name="test_model",
    )
    conversation.add_turn(turn)

    json_data = conversation.to_json()
    new_conversation = Conversation.from_json(json_data)

    assert new_conversation.conversation_id == conversation.conversation_id
    assert new_conversation.get_system_prompts() == conversation.get_system_prompts()
    assert len(new_conversation.turns) == len(conversation.turns)
    assert new_conversation.turns[0].turn_id == conversation.turns[0].turn_id
    assert (
        new_conversation.turns[0].participant_name
        == conversation.turns[0].participant_name
    )
    assert new_conversation.session_token_count == conversation.session_token_count

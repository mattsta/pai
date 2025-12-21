"""Tests for orchestrator components."""

from unittest.mock import MagicMock, patch

import pytest

from pai.models import (
    ChatRequest,
    CompletionRequest,
    Conversation,
    RequestStats,
    RuntimeConfig,
    Turn,
    UIMode,
    UIState,
)
from pai.orchestration.base import BaseOrchestrator


class ConcreteOrchestrator(BaseOrchestrator):
    """Concrete implementation for testing the base class."""

    async def run(self, user_input: str | None = None):
        return user_input


@pytest.fixture
def mock_ui():
    """Create a mock InteractiveUI for testing."""
    ui = MagicMock()
    ui.client = MagicMock()
    ui.client.stats = MagicMock()
    ui.client.stats.last_request_stats = RequestStats(tokens_sent=100)
    ui.client.config = MagicMock()
    ui.client.config.name = "test_endpoint"
    ui.client.config.model_name = "test-model"
    ui.state = UIState(mode=UIMode.CHAT)
    ui.pt_printer = MagicMock()
    ui.runtime_config = RuntimeConfig(chat=True)
    ui.conversation = Conversation()
    ui.log_dir = MagicMock()
    ui.log_dir.__truediv__ = MagicMock(return_value=MagicMock())
    return ui


def test_base_orchestrator_init(mock_ui):
    """Test that BaseOrchestrator correctly initializes references."""
    orchestrator = ConcreteOrchestrator(mock_ui)

    assert orchestrator.ui is mock_ui
    assert orchestrator.client is mock_ui.client
    assert orchestrator.state is mock_ui.state
    assert orchestrator.pt_printer is mock_ui.pt_printer
    assert orchestrator.runtime_config is mock_ui.runtime_config
    assert orchestrator.conversation is mock_ui.conversation
    assert orchestrator.log_dir is mock_ui.log_dir


async def test_base_orchestrator_run(mock_ui):
    """Test that run method can be implemented."""
    orchestrator = ConcreteOrchestrator(mock_ui)
    result = await orchestrator.run("test input")
    assert result == "test input"


def test_log_cancelled_turn(mock_ui):
    """Test that cancelled turns are logged correctly."""
    orchestrator = ConcreteOrchestrator(mock_ui)

    # Create a mock request
    request = ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="test-model",
        max_tokens=100,
    )

    partial_text = "This is a partial res"

    # Mock the file writing
    mock_file = MagicMock()
    mock_ui.log_dir.__truediv__.return_value = mock_file

    with patch("pai.orchestration.base.save_conversation_formats"):
        orchestrator._log_cancelled_turn(request, partial_text)

    # Verify a turn was added to the conversation
    assert len(mock_ui.conversation.turns) == 1
    turn = mock_ui.conversation.turns[0]
    assert turn.assistant_message == partial_text
    assert turn.response_data["pai_note"] == "This response was cancelled by the user."


def test_chat_request_to_dict():
    """Test ChatRequest serialization."""
    request = ChatRequest(
        messages=[
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ],
        model="gpt-4",
        max_tokens=1000,
        temperature=0.7,
        stream=True,
    )

    result = request.to_dict("gpt-4")

    assert result["model"] == "gpt-4"
    assert result["max_tokens"] == 1000
    assert result["temperature"] == 0.7
    assert result["stream"] is True
    assert len(result["messages"]) == 2


def test_completion_request_to_dict():
    """Test CompletionRequest serialization."""
    request = CompletionRequest(
        prompt="Complete this sentence:",
        model="gpt-3.5-turbo-instruct",
        max_tokens=100,
        temperature=0.5,
        stream=False,
    )

    result = request.to_dict("gpt-3.5-turbo-instruct")

    assert result["model"] == "gpt-3.5-turbo-instruct"
    assert result["prompt"] == "Complete this sentence:"
    assert result["max_tokens"] == 100
    assert result["temperature"] == 0.5
    assert result["stream"] is False


def test_ui_state_transitions():
    """Test UIState mode transitions."""
    state = UIState(mode=UIMode.CHAT)

    assert state.mode == UIMode.CHAT
    assert state.multiline_input is False

    # Transition to agent mode
    state.mode = UIMode.NATIVE_AGENT
    assert state.mode == UIMode.NATIVE_AGENT

    # Transition to arena mode
    state.mode = UIMode.ARENA
    assert state.mode == UIMode.ARENA


def test_request_stats_tracking():
    """Test RequestStats token and timing tracking."""
    stats = RequestStats(tokens_sent=500)

    assert stats.tokens_sent == 500
    assert stats.tokens_received == 0
    assert stats.response_time is None  # None until finished

    # Simulate receiving tokens
    stats.tokens_received = 200
    assert stats.tokens_received == 200

    # Test TTFT recording
    stats.record_first_token()
    assert stats.ttft is not None
    assert stats.ttft >= 0

    # Test finishing
    stats.finish(success=True)
    assert stats.response_time is not None
    assert stats.response_time >= 0


def test_turn_creation():
    """Test Turn object creation and serialization."""
    request_data = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "test-model",
    }
    response_data = {
        "choices": [{"message": {"role": "assistant", "content": "Hi there!"}}]
    }
    stats = RequestStats(tokens_sent=10)
    stats.tokens_received = 5

    turn = Turn(
        request_data=request_data,
        response_data=response_data,
        assistant_message="Hi there!",
        mode=UIMode.CHAT,
        stats=stats,
        endpoint_name="openai",
    )

    assert turn.assistant_message == "Hi there!"
    assert turn.mode == UIMode.CHAT
    assert turn.endpoint_name == "openai"
    assert turn.stats.tokens_received == 5

    # Test serialization
    turn_dict = turn.to_dict()
    assert "turn_id" in turn_dict
    assert turn_dict["assistant_message"] == "Hi there!"


def test_runtime_config_budget():
    """Test RuntimeConfig budget field."""
    config = RuntimeConfig(chat=True)
    assert config.session_budget is None

    config.session_budget = 1.50
    assert config.session_budget == 1.50

    config.session_budget = None
    assert config.session_budget is None

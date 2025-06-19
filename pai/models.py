"""
Data models for Polyglot AI state, history, and requests.
"""

import json
import time
import ulid
import pathlib
from typing import Optional, Dict, Any, Union, List, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime

from .utils import estimate_tokens

if TYPE_CHECKING:
    from .protocols.base_adapter import BaseProtocolAdapter


@dataclass
class Turn:
    """Represents a single request-response cycle in a conversation."""

    turn_id: ulid.ULID = field(default_factory=ulid.new)
    timestamp: datetime = field(default_factory=datetime.now)
    request_data: Dict[str, Any] = field(default_factory=dict)
    response_data: Dict[str, Any] = field(default_factory=dict)
    assistant_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the turn to a dictionary, handling non-native JSON types."""
        return {
            "turn_id": str(self.turn_id),
            "timestamp": self.timestamp.isoformat(),
            "request_data": self.request_data,
            "response_data": self.response_data,
            "assistant_message": self.assistant_message,
        }


@dataclass
class Conversation:
    """Manages the full conversation history, composed of multiple turns."""

    conversation_id: ulid.ULID = field(default_factory=ulid.new)
    turns: List[Turn] = field(default_factory=list)
    # The messages list represents the state of the conversation *before* the next user input
    _messages: List[Dict[str, str]] = field(default_factory=list)

    def add_turn(self, turn: Turn):
        """Adds a completed turn and updates the message history."""
        self.turns.append(turn)
        # The new message history is the request's messages + the assistant's reply
        self._messages = turn.request_data.get("messages", [])
        if turn.assistant_message:
            self._messages.append(
                {"role": "assistant", "content": turn.assistant_message}
            )

    def get_messages_for_next_turn(self, user_input: str) -> List[Dict[str, str]]:
        """Returns the list of messages for the next API call, including the new user input."""
        next_messages = list(self._messages)
        next_messages.append({"role": "user", "content": user_input})
        return next_messages

    def get_history(self) -> List[Dict[str, str]]:
        """Returns the current message history."""
        return self._messages

    def clear(self):
        """Clears the conversation, keeping any system prompt."""
        self.turns = []
        self._messages = [m for m in self._messages if m["role"] == "system"]

    def set_system_prompt(self, system_prompt: str):
        """Sets a new system prompt, clearing all subsequent history."""
        self.turns = []
        self._messages = [{"role": "system", "content": system_prompt}]


@dataclass
class RequestStats:
    """Encapsulates all statistics for a single request-response cycle."""

    start_time: float = field(default_factory=time.time)
    ttft: Optional[float] = None
    response_time: Optional[float] = None
    tokens_sent: int = 0
    tokens_received: int = 0
    success: bool = True
    finish_reason: Optional[str] = None

    # Internal state for live calculations
    _first_token_time: Optional[float] = None

    def record_first_token(self):
        """Call this when the first token is received to capture TTFT."""
        if self._first_token_time is None:
            self._first_token_time = time.time()
            self.ttft = self._first_token_time - self.start_time

    def add_received_tokens(self, count: int):
        """Add to the count of received tokens for this request."""
        self.tokens_received += count

    def finish(self, success: bool):
        """Call this when the response is complete to finalize stats."""
        self.response_time = time.time() - self.start_time
        self.success = success

    @property
    def current_duration(self) -> float:
        """Returns the total duration from the start of the request until now."""
        return time.time() - self.start_time

    @property
    def live_stream_duration(self) -> float:
        """Returns the duration from the first token until now."""
        if self._first_token_time:
            return time.time() - self._first_token_time
        return 0.0

    @property
    def live_tok_per_sec(self) -> float:
        """Returns live tokens per second, calculated from the first token."""
        duration = self.live_stream_duration
        if duration > 0.01:  # Avoid division by zero and noisy initial values
            return self.tokens_received / duration
        return 0.0

    @property
    def final_tok_per_sec(self) -> float:
        """Returns the final tokens per second for the entire completed response."""
        if self.response_time and self.response_time > 0:
            return self.tokens_received / self.response_time
        return 0.0


@dataclass
class TestSession:
    start_time: datetime = field(default_factory=datetime.now)
    requests_sent: int = 0
    total_tokens_sent: int = 0
    total_tokens_received: int = 0
    total_response_time: float = 0.0
    errors: int = 0
    # Holds the stats for the most recently *completed* successful request.
    last_request_stats: Optional[RequestStats] = None

    def add_completed_request(self, stats: RequestStats):
        """
        Adds statistics from a completed request to the session totals.
        Token and time stats are always accumulated, even for failed/cancelled requests.
        """
        self.requests_sent += 1
        self.total_tokens_sent += stats.tokens_sent
        self.total_tokens_received += stats.tokens_received
        if stats.response_time:
            self.total_response_time += stats.response_time

        self.last_request_stats = stats

        if not stats.success:
            self.errors += 1

    def get_stats(self) -> Dict[str, Any]:
        successful_requests = self.requests_sent - self.errors
        success_rate = (successful_requests / max(self.requests_sent, 1)) * 100
        avg_response_time = self.total_response_time / max(successful_requests, 1)
        return {
            "session_duration": str(datetime.now() - self.start_time).split(".")[0],
            "requests_sent": self.requests_sent,
            "successful_requests": successful_requests,
            "errors": self.errors,
            "success_rate": f"{success_rate:.1f}%",
            "total_tokens": self.total_tokens_sent + self.total_tokens_received,
            "avg_response_time": f"{avg_response_time:.2f}s",
            "tokens_per_second": f"{self.total_tokens_received / max(self.total_response_time, 1):.1f}",
        }


@dataclass
class EndpointConfig:
    name: str = "default"
    api_key: Optional[str] = None
    base_url: str = ""
    model_name: str = "default/model"
    # The timeout for individual requests, in seconds.
    timeout: int = 180
    max_retries: int = 3
    backoff_factor: float = 0.3
    chat_adapter: Optional["BaseProtocolAdapter"] = None
    completion_adapter: Optional["BaseProtocolAdapter"] = None


@dataclass
class CompletionRequest:
    prompt: str
    model: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    stream: bool = False

    def to_dict(self, default_model: str) -> Dict[str, Any]:
        return {
            "model": self.model or default_model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream,
        }


@dataclass
class ChatRequest:
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    stream: bool = False
    tools: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self, default_model: str) -> Dict[str, Any]:
        """Serializes the request to a dictionary for the API call."""
        payload = {
            "model": self.model or default_model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream,
        }
        if self.tools:
            payload["tools"] = self.tools
            payload["tool_choice"] = "auto"
        return payload

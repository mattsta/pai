"""
Data models for Polyglot AI state, history, and requests.
"""

import enum
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import ulid
from pydantic import BaseModel, Field

from .utils import estimate_tokens

if TYPE_CHECKING:
    from .models import RequestStats
    from .protocols.base_adapter import BaseProtocolAdapter


class UIMode(enum.Enum):
    """Enumerates the possible interaction modes for the UI."""

    COMPLETION = "completion"
    CHAT = "chat"
    NATIVE_AGENT = "native_agent"
    LEGACY_AGENT = "legacy_agent"
    ARENA = "arena"


@dataclass
class UIState:
    """Holds the dynamic state of the user interface."""

    mode: UIMode = UIMode.CHAT
    # Holds the state for the *active* arena session, if any.
    arena: Optional["ArenaState"] = None
    multiline_input: bool = False
    # Agent-specific stats
    tools_used: int = 0
    agent_loops: int = 0


@dataclass
class Turn:
    """Represents a single request-response cycle in a conversation."""

    turn_id: ulid.ULID = field(default_factory=ulid.new)
    timestamp: datetime = field(default_factory=datetime.now)
    request_data: dict[str, Any] = field(default_factory=dict)
    response_data: dict[str, Any] = field(default_factory=dict)
    assistant_message: str = ""
    # Arena mode fields
    participant_name: str | None = None
    model_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes the turn to a dictionary, handling non-native JSON types."""
        return {
            "turn_id": str(self.turn_id),
            "timestamp": self.timestamp.isoformat(),
            "request_data": self.request_data,
            "response_data": self.response_data,
            "assistant_message": self.assistant_message,
            "participant_name": self.participant_name,
            "model_name": self.model_name,
        }


@dataclass
class Conversation:
    """Manages the full conversation history, composed of multiple turns."""

    conversation_id: ulid.ULID = field(default_factory=ulid.new)
    turns: list[Turn] = field(default_factory=list)
    # NEW: System prompts are managed in a dedicated stack.
    _system_prompts: list[str] = field(default_factory=list)
    # The _messages list now ONLY represents the user/assistant/tool turn history.
    _messages: list[dict[str, Any]] = field(default_factory=list)
    session_token_count: int = 0

    def _recalculate_token_count(self):
        """Recalculates the session token count from the current state."""
        token_count = 0
        if system_content := self._get_combined_system_prompt():
            token_count += estimate_tokens(system_content)
        for msg in self._messages:
            # Handle different message structures (text content, tool calls)
            if isinstance(content := msg.get("content"), str):
                token_count += estimate_tokens(content)
        self.session_token_count = token_count

    def _get_combined_system_prompt(self) -> str | None:
        """Joins all system prompts into a single string for the API."""
        if not self._system_prompts:
            return None
        # Join with a clear separator for the model.
        return "\n\n---\n\n".join(self._system_prompts)

    def get_system_prompts(self) -> list[str]:
        """Returns a copy of the current system prompt stack."""
        return list(self._system_prompts)

    def add_system_prompt(self, system_prompt: str):
        """Adds a new system prompt to the stack."""
        self._system_prompts.append(system_prompt)
        self._recalculate_token_count()

    def pop_system_prompt(self) -> str | None:
        """Removes the most recently added system prompt from the stack."""
        if self._system_prompts:
            popped = self._system_prompts.pop()
            self._recalculate_token_count()
            return popped
        return None

    def clear_system_prompts(self):
        """Removes all system prompts."""
        self._system_prompts.clear()
        self._recalculate_token_count()

    def add_turn(self, turn: Turn, request_stats: Optional["RequestStats"] = None):
        """Adds a completed turn and updates the message history."""
        self.turns.append(turn)
        # Set the message history from the request, excluding the system prompt.
        self._messages = [
            m for m in turn.request_data.get("messages", []) if m["role"] != "system"
        ]
        # Append the final assistant message object from the response data. This
        # is more robust as it includes tool calls, not just text content.
        if choices := turn.response_data.get("choices"):
            if message := choices[0].get("message"):
                # Ensure we don't add an empty assistant message if one already exists
                if message.get("content") or message.get("tool_calls"):
                    self._messages.append(message)

        # The old token counting is replaced by a full recalculation.
        self._recalculate_token_count()

    def get_messages_for_next_turn(self, user_input: str) -> list[dict[str, Any]]:
        """
        Constructs the list of messages for the next API call, combining the
        system prompt stack and the turn history.
        """
        messages = []
        if system_prompt := self._get_combined_system_prompt():
            messages.append({"role": "system", "content": system_prompt})

        # The turn history is now clean of system prompts.
        messages.extend(self._messages)
        messages.append({"role": "user", "content": user_input})
        return messages

    def get_history(self) -> list[dict[str, str]]:
        """Returns the current message history."""
        return self._messages

    def clear(self):
        """Clears the turn history, but preserves the system prompt stack."""
        self.turns = []
        self._messages = []
        # System prompts are preserved, so we just recalculate token count from them.
        self._recalculate_token_count()

    def get_rich_history_for_template(self) -> list[dict[str, Any]]:
        """
        Generates a enriched history list suitable for detailed HTML logging.
        It reconstructs the message flow from turns, adding participant info.
        """
        # Start with all system prompts from the stack.
        history = [
            {"role": "system", "content": prompt} for prompt in self._system_prompts
        ]

        # Create a stable mapping of participant names to a unique index for color-coding.
        participant_names = sorted(
            list(set(t.participant_name for t in self.turns if t.participant_name))
        )
        participant_index_map = {name: i for i, name in enumerate(participant_names)}

        for turn in self.turns:
            # Add the user message that initiated this turn
            for msg in turn.request_data.get("messages", []):
                # Avoid duplicating the system prompt or prior turns' messages.
                if msg["role"] == "user":
                    history.append(msg)
                    break  # Assume one user message per turn start

            # Add the assistant's response for this turn, with participant info
            p_name = turn.participant_name
            assistant_msg = {
                "role": "assistant",
                "content": turn.assistant_message,
                "participant_name": p_name,
                "model_name": turn.model_name,
                "participant_index": participant_index_map.get(p_name),
            }
            # Add tool calls from the response if they exist
            if turn.response_data.get("choices"):
                response_message = turn.response_data["choices"][0].get("message", {})
                if "tool_calls" in response_message:
                    assistant_msg["tool_calls"] = response_message["tool_calls"]
            history.append(assistant_msg)

        return history

    def set_system_prompt(self, system_prompt: str):
        """Replaces the entire system prompt stack with a single new prompt."""
        self.turns = []
        self._messages = []
        self._system_prompts = [system_prompt]
        self._recalculate_token_count()


@dataclass
class SmoothingStats:
    """Encapsulates statistics for the stream smoother."""

    queue_size: int = 0
    stream_finished: bool = False
    smoothing_aborted: bool = False
    buffer_drain_time_s: float = 0.0
    arrivals: int = 0
    min_delta: str = "N/A"
    mean_delta: str = "N/A"
    median_delta: str = "N/A"
    max_delta: str = "N/A"
    gaps: int = 0
    bursts: int = 0


@dataclass
class RequestStats:
    """Encapsulates all statistics for a single request-response cycle."""

    start_time: float = field(default_factory=time.time)
    ttft: float | None = None
    response_time: float | None = None
    tokens_sent: int = 0
    tokens_received: int = 0
    success: bool = True
    finish_reason: str | None = None
    # Cost tracking
    input_cost: float = 0.0
    output_cost: float = 0.0
    # Statistics from the stream smoother, if active
    smoothing_stats: Optional["SmoothingStats"] = None

    # Internal state for live calculations
    _first_token_time: float | None = None

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
class SessionStats:
    start_time: datetime = field(default_factory=datetime.now)
    requests_sent: int = 0
    total_tokens_sent: int = 0
    total_tokens_received: int = 0
    total_response_time: float = 0.0
    total_cost: float = 0.0
    errors: int = 0
    # Holds the stats for the most recently *completed* successful request.
    last_request_stats: RequestStats | None = None

    def add_completed_request(self, stats: RequestStats):
        """
        Adds statistics from a completed request to the session totals.
        Token and time stats are always accumulated, even for failed/cancelled requests.
        """
        self.requests_sent += 1
        self.total_tokens_sent += stats.tokens_sent
        self.total_tokens_received += stats.tokens_received
        self.total_cost += stats.input_cost + stats.output_cost
        if stats.response_time:
            self.total_response_time += stats.response_time

        self.last_request_stats = stats

        if not stats.success:
            self.errors += 1

    def get_stats(self) -> dict[str, Any]:
        successful_requests = self.requests_sent - self.errors
        success_rate = (successful_requests / max(self.requests_sent, 1)) * 100
        avg_response_time = self.total_response_time / max(self.requests_sent, 1)

        stats = {
            "session_duration": str(datetime.now() - self.start_time).split(".")[0],
            "total_cost": f"${self.total_cost:.5f}",
            "requests_sent": self.requests_sent,
            "successful_requests": successful_requests,
            "errors": self.errors,
            "success_rate": f"{success_rate:.1f}%",
            "total_tokens": self.total_tokens_sent + self.total_tokens_received,
            "avg_response_time": f"{avg_response_time:.2f}s",
            "tokens_per_second": f"{self.total_tokens_received / max(self.total_response_time, 1):.1f}",
        }

        if self.last_request_stats:
            last = self.last_request_stats
            last = self.last_request_stats
            last_req_cost = last.input_cost + last.output_cost
            stats["last_request"] = {
                "cost": f"${last_req_cost:.5f}",
                "ttft": f"{last.ttft:.2f}s" if last.ttft else "N/A",
                "response_time": f"{last.response_time:.2f}s"
                if last.response_time
                else "N/A",
                "tokens_received": last.tokens_received,
                "tokens_per_second": f"{last.final_tok_per_sec:.1f}",
                "finish_reason": last.finish_reason or "N/A",
            }
            # Pass through smoothing stats if they exist
            if last.smoothing_stats:
                # Manually create a dict for printing to match what log_utils expects
                stats["last_request"]["smoothing_stats"] = {
                    "arrivals": last.smoothing_stats.arrivals,
                    "gaps": last.smoothing_stats.gaps,
                    "bursts": last.smoothing_stats.bursts,
                    "min_delta": last.smoothing_stats.min_delta,
                    "mean_delta": last.smoothing_stats.mean_delta,
                    "median_delta": last.smoothing_stats.median_delta,
                    "max_delta": last.smoothing_stats.max_delta,
                    "aborted": last.smoothing_stats.smoothing_aborted,
                }
        return stats


# --- Arena Data Models ---


@dataclass
class ArenaParticipant:
    """Configuration for a single participant in a multi-model arena."""

    id: str  # The key from the config, e.g., "proposer"
    name: str  # Display name, e.g., "Proposer"
    endpoint: str
    model: str
    system_prompt: str
    # Each participant gets their own conversation history.
    conversation: "Conversation" = field(default_factory=Conversation)


@dataclass
class Arena:
    """Configuration for a two-participant arena session."""

    name: str  # e.g., "debate"
    participants: dict[str, ArenaParticipant]
    initiator_id: str
    judge: ArenaParticipant | None = None

    def get_participant(self, participant_id: str) -> ArenaParticipant | None:
        return self.participants.get(participant_id)

    def get_initiator(self) -> ArenaParticipant:
        """Returns the participant designated to start the conversation."""
        # This assumes the initiator_id from the config is always valid.
        return self.participants[self.initiator_id]


@dataclass
class ArenaState:
    """Holds the dynamic state of an ongoing arena session."""

    arena_config: Arena
    turn_order_ids: list[str]
    max_turns: int
    last_message: str = ""
    # Multiply by number of participants since one "turn" involves everyone speaking once.
    current_speech: int = 0

    @property
    def max_speeches(self) -> int:
        """Total number of individual model responses in the arena session."""
        return self.max_turns * len(self.arena_config.participants)


@dataclass
class EndpointConfig:
    name: str = "default"
    api_key: str | None = None
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
    model: str | None = None
    max_tokens: int = 1000
    temperature: float = 0.7
    stream: bool = False

    def to_dict(self, default_model: str) -> dict[str, Any]:
        return {
            "model": self.model or default_model,
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream,
        }


@dataclass
class ChatRequest:
    messages: list[dict[str, str]]
    model: str | None = None
    max_tokens: int = 1000
    temperature: float = 0.7
    stream: bool = False
    tools: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, default_model: str) -> dict[str, Any]:
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


# --- Configuration Models ---


class TomlParticipant(BaseModel):
    name: str
    endpoint: str
    model: str
    system_prompt_key: str


class TomlJudge(BaseModel):
    name: str
    endpoint: str
    model: str
    system_prompt_key: str


class TomlArena(BaseModel):
    initiator: str
    participants: dict[str, TomlParticipant]
    judge: TomlJudge | None = None


class TomlProfile(BaseModel):
    """Configuration for a reusable profile."""

    endpoint: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: int | None = None
    chat: bool | None = None
    system: str | None = None
    rich_text: bool | None = None
    confirm_tool_use: bool | None = None


class TomlToolConfig(BaseModel):
    directories: list[str] = []


class TomlEndpoint(BaseModel):
    name: str
    base_url: str
    api_key_env: str
    chat_adapter: str | None = None
    completion_adapter: str | None = None
    timeout: int = 180


class PolyglotConfig(BaseModel):
    """Represents the structure of the polyglot.toml file."""

    endpoints: list[TomlEndpoint] = Field(default_factory=list)
    tool_config: TomlToolConfig | None = None
    arenas: dict[str, TomlArena] = Field(default_factory=dict)
    profiles: dict[str, TomlProfile] = Field(default_factory=dict)


class RuntimeConfig(BaseModel):
    """Holds runtime configuration state, replacing argparse.Namespace."""

    config: str = "polyglot.toml"
    profile: str | None = None
    endpoint: str = "openai"
    model: str | None = None
    chat: bool = False
    prompt: str | None = None
    system: str | None = None
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout: int | None = None
    stream: bool = True
    verbose: bool = False
    debug: bool = False
    tools: bool = False
    rich_text: bool = True
    log_file: str | None = None
    confirm_tool_use: bool = False
    smooth_stream: bool = False

    class Config:
        frozen = False  # Allows mutation by commands

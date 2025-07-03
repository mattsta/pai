"""
Data models for Polyglot AI state, history, and requests.
"""

import dataclasses
import enum
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import ulid
from pydantic import BaseModel, ConfigDict, Field

from .utils import estimate_tokens

if TYPE_CHECKING:
    from .models import RequestCost, RequestStats
    from .pricing import PricingService
    from .protocols.base_adapter import BaseProtocolAdapter


class ArenaTurnOrder(str, enum.Enum):
    """Enumerates the possible turn order strategies for an arena."""

    SEQUENTIAL = "sequential"
    RANDOM = "random"


class ArenaConversationStyle(str, enum.Enum):
    """Enumerates the possible conversation styles for an arena."""

    PAIRWISE = "pairwise"  # Each model sees a 1-on-1 conversation
    CHATROOM = "chatroom"  # All models see a shared chat log


class UIMode(enum.Enum):
    """Enumerates the possible interaction modes for the UI."""

    COMPLETION = "completion"
    CHAT = "chat"
    NATIVE_AGENT = "native_agent"
    LEGACY_AGENT = "legacy_agent"
    ARENA = "arena"
    ARENA_SETUP = "arena_setup"
    TEMPLATE_COMPLETION = "template_completion"


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
    # For template completion mode
    chat_template: str | None = None
    chat_template_obj: Any | None = None


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
    # Context fields for logging
    mode: "UIMode | None" = None
    stats: "RequestStats | None" = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes the turn to a dictionary, handling non-native JSON types."""
        return {
            "turn_id": str(self.turn_id),
            "timestamp": self.timestamp.isoformat(),
            "mode": self.mode.value if self.mode else None,
            "stats": self.stats.to_dict() if self.stats else None,
            "request_data": self.request_data,
            "response_data": self.response_data,
            "assistant_message": self.assistant_message,
            "participant_name": self.participant_name,
            "model_name": self.model_name,
        }


@dataclass
class RequestCost:
    """Tracks and calculates the cost of a single API request."""

    # Dependencies for calculation
    _pricing_service: "PricingService"
    _model_pricing: "ModelPricing"

    # State
    input_tokens: int = 0
    output_tokens: int = 0

    # Calculated properties
    input_cost: float = 0.0
    output_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serializes the cost breakdown into a dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
        }

    def update(self, input_tokens: int | None = None, output_tokens: int | None = None):
        """
        Updates token counts and recalculates costs. Only updates counts that are provided.
        """
        if self._pricing_service is None or self._model_pricing is None:
            return  # Cannot calculate cost

        if input_tokens is not None:
            self.input_tokens = input_tokens
        if output_tokens is not None:
            self.output_tokens = output_tokens

        (
            self.input_cost,
            self.output_cost,
        ) = self._pricing_service.calculate_cost(
            self._model_pricing, self.input_tokens, self.output_tokens
        )

    @property
    def total_cost(self) -> float:
        """The total cost of the request."""
        return self.input_cost + self.output_cost


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

        # For CHAT mode, we rebuild the message history from the turn data
        # to ensure it's always in sync with what the model saw.
        if "messages" in turn.request_data:
            # Set the message history from the request, excluding the system prompt.
            self._messages = [
                m
                for m in turn.request_data.get("messages", [])
                if m["role"] != "system"
            ]
            # Append the final assistant message object from the response data.
            if choices := turn.response_data.get("choices"):
                if message := choices[0].get("message"):
                    if message.get("content") or message.get("tool_calls"):
                        self._messages.append(message)
        # For COMPLETION mode, the history is not cumulative. We replace the
        # message log with just the last prompt/response pair.
        elif "prompt" in turn.request_data:
            self._messages = [
                {"role": "user", "content": turn.request_data["prompt"]},
                {"role": "assistant", "content": turn.assistant_message},
            ]

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
        Generates an enriched history list suitable for detailed HTML logging.
        It correctly renders multi-step agentic turns and arena conversations.
        """
        history = [
            {"role": "system", "content": prompt} for prompt in self._system_prompts
        ]

        participant_names = sorted(
            list(set(t.participant_name for t in self.turns if t.participant_name))
        )
        participant_index_map = {name: i for i, name in enumerate(participant_names)}

        for turn in self.turns:
            # For agentic turns, the full message history is in request_data.
            # For simple turns, it's just the user prompt.
            turn_messages = [
                m
                for m in turn.request_data.get("messages", [])
                if m["role"] != "system"
            ]

            # The final response from the model is in response_data.
            # We need to append it to the history unless it's already there
            # (which can happen in agentic loops).
            if choices := turn.response_data.get("choices"):
                if message := choices[0].get("message"):
                    # For Anthropic, content is a list, not a string. Normalize it.
                    if isinstance(message.get("content"), list):
                        # The final text content is already in turn.assistant_message
                        message["content"] = turn.assistant_message

                    if not turn_messages or turn_messages[-1] != message:
                        turn_messages.append(message)

            # Process all messages for this turn, adding arena metadata where needed.
            for msg in turn_messages:
                msg_for_template = msg.copy()
                # The template needs to handle various message structures.
                # We normalize the 'tool' role to have a 'name' if possible.
                if (
                    msg_for_template["role"] == "tool"
                    and "name" not in msg_for_template
                ):
                    # Find the corresponding tool_call to get the name.
                    tool_call_id = msg_for_template.get("tool_call_id")
                    if tool_call_id:
                        for prev_msg in reversed(turn_messages):
                            if prev_msg["role"] == "assistant":
                                for tc in prev_msg.get("tool_calls", []):
                                    if tc["id"] == tool_call_id:
                                        msg_for_template["name"] = tc["function"][
                                            "name"
                                        ]
                                        break
                                if msg_for_template.get("name"):
                                    break

                # Attach turn-level context to assistant messages for display.
                if msg_for_template["role"] == "assistant":
                    if p_name := turn.participant_name:
                        msg_for_template["participant_name"] = p_name
                        msg_for_template["model_name"] = turn.model_name
                        msg_for_template["participant_index"] = (
                            participant_index_map.get(p_name)
                        )
                    # Add mode and stats for logging and display
                    msg_for_template["mode"] = turn.mode
                    msg_for_template["stats"] = (
                        turn.stats.to_dict() if turn.stats else None
                    )

                history.append(msg_for_template)

        return history

    def set_system_prompt(self, system_prompt: str):
        """Replaces the entire system prompt stack with a single new prompt."""
        self.turns = []
        self._messages = []
        self._system_prompts = [system_prompt]
        self._recalculate_token_count()

    def to_json(self) -> dict[str, Any]:
        """Serializes the conversation to a JSON-compatible dictionary."""
        return {
            "conversation_id": str(self.conversation_id),
            "turns": [turn.to_dict() for turn in self.turns],
            "_system_prompts": self._system_prompts,
            "_messages": self._messages,
            "session_token_count": self.session_token_count,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Conversation":
        """Deserializes a conversation from a dictionary."""
        # Reconstruct Turn objects from their dictionary representation
        turns = [
            Turn(
                turn_id=ulid.parse(t["turn_id"]),
                timestamp=datetime.fromisoformat(t["timestamp"]),
                request_data=t["request_data"],
                response_data=t["response_data"],
                assistant_message=t["assistant_message"],
                participant_name=t.get("participant_name"),
                model_name=t.get("model_name"),
            )
            for t in data.get("turns", [])
        ]

        convo = cls(
            conversation_id=ulid.parse(data["conversation_id"]),
            turns=turns,
            _system_prompts=data.get("_system_prompts", []),
            _messages=data.get("_messages", []),
            session_token_count=data.get("session_token_count", 0),
        )
        return convo


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
    stdev_delta: str = "N/A"
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
    cost: "RequestCost | None" = None
    # Statistics on stream jitter, if applicable
    jitter_stats: Optional["SmoothingStats"] = None

    # Internal state for live calculations
    _first_token_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes request stats into a dictionary."""
        return {
            "start_time": self.start_time,
            "ttft": self.ttft,
            "response_time": self.response_time,
            "tokens_sent": self.tokens_sent,
            "tokens_received": self.tokens_received,
            "success": self.success,
            "finish_reason": self.finish_reason,
            "cost": self.cost.to_dict() if self.cost else None,
            "jitter_stats": dataclasses.asdict(self.jitter_stats)
            if self.jitter_stats
            else None,
        }

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
class TieredCost:
    """Represents a single tier in a tiered pricing model."""

    up_to: int  # The upper bound of the token count for this tier (exclusive). -1 for infinity.
    cost: float  # The cost per million tokens for this tier.


@dataclass
class TimeWindowPricing:
    """Encapsulates all pricing information for a specific time window."""

    start_hour: int  # The start hour of the window (0-23).
    end_hour: int  # The end hour of the window (0-23).
    input_cost: float = 0.0  # Flat input cost per million tokens.
    output_cost: float = 0.0  # Flat output cost per million tokens.
    input_tiers: list[TieredCost] = field(default_factory=list)
    output_tiers: list[TieredCost] = field(default_factory=list)


@dataclass
class ModelPricing:
    """Unified model for pricing, supporting flat, tiered, and time-based rates."""

    # "Anytime" flat rates (used if no time windows match or are defined)
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    # "Anytime" tiered rates
    tiered_input_costs: list[TieredCost] = field(default_factory=list)
    tiered_output_costs: list[TieredCost] = field(default_factory=list)
    # Time-window specific rates
    time_windows: list[TimeWindowPricing] = field(default_factory=list)
    # LiteLLM specific batch pricing (for compatibility)
    input_cost_per_token_batches: float = 0.0
    output_cost_per_token_batches: float = 0.0


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
        if stats.cost:
            self.total_cost += stats.cost.total_cost
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
            last_req_cost = last.cost.total_cost if last.cost else 0.0
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
            # Pass through jitter stats if they exist
            if last.jitter_stats:
                # Manually create a dict for printing to match what log_utils expects
                stats["last_request"]["jitter_stats"] = {
                    "arrivals": last.jitter_stats.arrivals,
                    "gaps": last.jitter_stats.gaps,
                    "bursts": last.jitter_stats.bursts,
                    "min_delta": last.jitter_stats.min_delta,
                    "mean_delta": last.jitter_stats.mean_delta,
                    "stdev_delta": last.jitter_stats.stdev_delta,
                    "max_delta": last.jitter_stats.max_delta,
                    "aborted": last.jitter_stats.smoothing_aborted,
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
    turn_order: ArenaTurnOrder = ArenaTurnOrder.SEQUENTIAL
    wildcards_enabled: bool = False
    conversation_style: ArenaConversationStyle = ArenaConversationStyle.PAIRWISE

    def get_participant(self, participant_id: str) -> ArenaParticipant | None:
        return self.participants.get(participant_id)

    def to_log_dict(self) -> dict[str, Any]:
        """Creates a serializable dictionary for logging, omitting circular references."""
        participants_log = {}
        for p_id, p in self.participants.items():
            if p_id == "judge":
                continue
            participants_log[p_id] = {
                "name": p.name,
                "endpoint": p.endpoint,
                "model": p.model,
                "system_prompt": p.system_prompt,
            }
        judge_log = None
        if self.judge:
            judge_log = {
                "name": self.judge.name,
                "endpoint": self.judge.endpoint,
                "model": self.judge.model,
                "system_prompt": self.judge.system_prompt,
            }
        return {
            "name": self.name,
            "initiator_id": self.initiator_id,
            "turn_order": self.turn_order.value,
            "wildcards_enabled": self.wildcards_enabled,
            "conversation_style": self.conversation_style.value,
            "participants": participants_log,
            "judge": judge_log,
        }

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
    initial_prompt: str = ""
    last_message: str = ""
    # Multiply by number of participants since one "turn" involves everyone speaking once.
    current_speech: int = 0

    @property
    def max_speeches(self) -> int:
        """Total number of individual model responses in the arena session."""
        # Use turn_order_ids as it correctly excludes the judge.
        return self.max_turns * len(self.turn_order_ids) if self.turn_order_ids else 0


# --- Arena Configuration File Models ---
class ArenaConfigParticipant(BaseModel):
    """A participant model for standalone arena config files."""

    name: str
    endpoint: str
    model: str
    system_prompt: str | None = None
    system_prompt_key: str | None = None


class ArenaConfigJudge(ArenaConfigParticipant):
    """A judge model for standalone arena config files."""

    pass


class ArenaConfigFile(BaseModel):
    """Represents the structure of a standalone arena YAML/TOML file."""

    name: str
    initiator: str
    max_turns: int = 10
    participants: dict[str, ArenaConfigParticipant]
    judge: ArenaConfigJudge | None = None
    turn_order: ArenaTurnOrder = ArenaTurnOrder.SEQUENTIAL
    wildcards_enabled: bool = False
    conversation_style: ArenaConversationStyle = ArenaConversationStyle.PAIRWISE


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
    system_prompt: str | None = None
    system_prompt_key: str | None = None


class TomlJudge(BaseModel):
    name: str
    endpoint: str
    model: str
    system_prompt: str | None = None
    system_prompt_key: str | None = None


class TomlArena(BaseModel):
    initiator: str
    participants: dict[str, TomlParticipant]
    judge: TomlJudge | None = None
    turn_order: str | None = None
    wildcards_enabled: bool | None = None
    conversation_style: str | None = None


# --- Custom Pricing Configuration Models ---
class TomlTieredCost(BaseModel):
    """A single tier in a custom pricing config."""

    up_to: int
    cost: float


class TomlTimeWindowCost(BaseModel):
    """A time-based pricing window in a custom pricing config."""

    start_hour: int
    end_hour: int
    input_cost: float | None = None
    output_cost: float | None = None
    input_tiers: list[TomlTieredCost] = []
    output_tiers: list[TomlTieredCost] = []


class TomlCustomModelPricing(BaseModel):
    """Pricing definition for a single model in a custom config."""

    input_cost: float | None = None
    output_cost: float | None = None
    input_tiers: list[TomlTieredCost] = []
    output_tiers: list[TomlTieredCost] = []
    time_windows: list[TomlTimeWindowCost] = []


class TomlCustomPricing(BaseModel):
    """The root model for a custom pricing TOML file."""

    pricing: dict[str, dict[str, TomlCustomModelPricing]] = Field(default_factory=dict)


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
    api_key_env: str | None = None
    user_agent: str | None = None
    chat_adapter: str | None = None
    completion_adapter: str | None = None
    timeout: int = 180


class PolyglotConfig(BaseModel):
    """Represents the structure of the pai.toml file."""

    custom_pricing_file: str | None = Field(None, alias="custom-pricing-file")
    endpoints: list[TomlEndpoint] = Field(default_factory=list)
    tool_config: TomlToolConfig | None = None
    arenas: dict[str, TomlArena] = Field(default_factory=dict)
    profiles: dict[str, TomlProfile] = Field(default_factory=dict)


class RuntimeConfig(BaseModel):
    """Holds runtime configuration state, replacing argparse.Namespace."""

    model_config = ConfigDict(frozen=False)  # Allows mutation by commands

    config: str = "pai.toml"
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
    smooth_stream: bool = True
    custom_pricing_file: str | None = Field(None, alias="custom-pricing-file")

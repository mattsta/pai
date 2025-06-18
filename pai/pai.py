"""
Polyglot AI: An Interactive, Multi-Provider CLI for the OpenAI API Format.
This version is a direct refactoring of the original code, preserving all features
and fixing the circular import error.
"""

import os
import sys
import json
import time
import argparse
import threading
import ulid
import pathlib
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass, field
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from jinja2 import Environment, FileSystemLoader, select_autoescape

import toml

# --- Protocol Adapter Imports ---
from .protocols.base_adapter import BaseProtocolAdapter, ProtocolContext
from .protocols.openai_chat_adapter import OpenAIChatAdapter
from .protocols.legacy_completion_adapter import LegacyCompletionAdapter

# --- Global Definitions ---
session = PromptSession()
ADAPTER_MAP = {
    "openai_chat": OpenAIChatAdapter(),
    "legacy_completion": LegacyCompletionAdapter(),
}


# --- [NEW] Spinner Class for Loading Indication ---
class Spinner:
    """A simple threaded spinner to indicate background activity."""

    def __init__(self, message: str = "Working..."):
        self.message = message
        self.running = False
        self.thread = None
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # Braille spinner

    def _spin(self):
        while self.running:
            for char in self.spinner_chars:
                if not self.running:
                    break
                sys.stdout.write(f"\r{char} {self.message}")
                sys.stdout.flush()
                time.sleep(0.1)

    def start(self):
        """Starts the spinner in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        """Stops the spinner and clears the line."""
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)
        # Clear the line
        sys.stdout.write("\r" + " " * (len(self.message) + 5) + "\r")
        sys.stdout.flush()


# --- Data Classes for State Management ---
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


# --- Data Classes (Identical to original) ---
@dataclass
class EndpointConfig:
    name: str = "default"
    api_key: Optional[str] = None
    base_url: str = ""
    model_name: str = "default/model"
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 0.3
    chat_adapter: Optional[BaseProtocolAdapter] = None
    completion_adapter: Optional[BaseProtocolAdapter] = None


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

    def to_dict(self, default_model: str) -> Dict[str, Any]:
        return {
            "model": self.model or default_model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": self.stream,
        }


@dataclass
class TestSession:
    start_time: datetime = field(default_factory=datetime.now)
    requests_sent: int = 0
    total_tokens_sent: int = 0
    total_tokens_received: int = 0
    total_response_time: float = 0.0
    errors: int = 0
    # Stats for the most recent request
    last_ttft: float = 0.0
    last_response_time: float = 0.0
    last_tokens_received: int = 0

    def add_request(
        self,
        tokens_sent: int,
        tokens_received: int,
        response_time: float,
        success: bool = True,
        ttft: Optional[float] = None,
    ):
        self.requests_sent += 1
        self.total_tokens_sent += tokens_sent
        self.total_tokens_received += tokens_received
        self.total_response_time += response_time
        if not success:
            self.errors += 1
        else:
            # Only update last request stats on success
            if ttft is not None:
                self.last_ttft = ttft
            self.last_response_time = response_time
            self.last_tokens_received = tokens_received

    def get_stats(self) -> Dict[str, Any]:
        successful_requests = self.requests_sent - self.errors
        success_rate = (successful_requests / max(self.requests_sent, 1)) * 100
        return {
            "session_duration": str(datetime.now() - self.start_time).split(".")[0],
            "requests_sent": self.requests_sent,
            "successful_requests": successful_requests,
            "errors": self.errors,
            "success_rate": f"{success_rate:.1f}%",
            "total_tokens": self.total_tokens_sent + self.total_tokens_received,
            "avg_response_time": f"{self.total_response_time / max(self.requests_sent, 1):.2f}s",
            "tokens_per_second": f"{self.total_tokens_received / max(self.total_response_time, 1):.1f}",
        }


# --- [MODIFIED] StreamingDisplay now manages the Spinner ---
class StreamingDisplay:
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.start_time: Optional[float] = None
        self.ttft: Optional[float] = None
        self.line_count: int = 0
        self.chunk_count: int = 0
        self.current_response: str = ""
        self.spinner = Spinner("Waiting for first token...")
        self.first_token_received = False

    def start_response(self):
        """Starts the spinner and prepares for a new response."""
        self.current_response = ""
        self.ttft = None
        self.start_time = time.time()
        self.line_count = 0
        self.chunk_count = 0
        self.first_token_received = False
        if not self.debug_mode:
            self.spinner.start()

    def show_raw_line(self, line: str):
        if self.debug_mode:
            # In debug mode, we stop the spinner immediately and show the debug header
            if not self.first_token_received:
                self.spinner.stop()  # Stop spinner if running
                print("🔍 DEBUG MODE: Showing raw protocol traffic\n" + "=" * 60)
                self.first_token_received = (
                    True  # Prevent header from printing multiple times
                )
            self.line_count += 1
            timestamp = time.time() - (self.start_time or 0)
            prefix = f"⚪ [{timestamp:6.2f}s] L{self.line_count:03d}: "
            if line.startswith("data: "):
                prefix = f"🔵 [{timestamp:6.2f}s] L{self.line_count:03d}: "
            print(f"{prefix}{repr(line)}")

    def show_parsed_chunk(self, chunk_data: Dict, chunk_text: str):
        """Stops the spinner on the first chunk, then prints content."""
        if not self.first_token_received and not self.debug_mode:
            self.spinner.stop()
            if self.start_time:
                self.ttft = time.time() - self.start_time
            print("\n🤖 Assistant: ", end="", flush=True)
            self.first_token_received = True

        self.chunk_count += 1
        self.current_response += chunk_text
        if self.debug_mode:
            timestamp = time.time() - (self.start_time or 0)
            print(
                f"🟢 [{timestamp:6.2f}s] C{self.chunk_count:03d} TEXT: {repr(chunk_text)}"
            )
        else:
            print(chunk_text, end="", flush=True)

    def finish_response(self) -> tuple[float, int, Optional[float]]:
        """Ensures spinner is stopped and calculates final stats for the turn."""
        self.spinner.stop()  # Ensure spinner is always stopped at the end
        elapsed = time.time() - (self.start_time or 0)
        # TODO: This is a poor way to count tokens. Replace with tiktoken or similar.
        tokens_received = len(self.current_response.split())
        if self.debug_mode:
            print(
                "=" * 60
                + f"\n🔍 DEBUG SUMMARY: {self.line_count} lines, {self.chunk_count} chunks, {elapsed:.2f}s\n"
                + "=" * 60
            )
        elif (
            self.first_token_received
        ):  # Only print stats if we actually received something
            tok_per_sec = tokens_received / max(elapsed, 0.1)
            ttft_str = f" | TTFT: {self.ttft:.2f}s" if self.ttft else ""
            print(
                f"\n\n📊 Response in {elapsed:.2f}s ({tokens_received} tokens, {tok_per_sec:.1f} tok/s{ttft_str})"
            )
        return elapsed, tokens_received, self.ttft


class APIError(Exception):
    pass


class PolyglotClient:
    def __init__(self, args: argparse.Namespace, loaded_config: Dict):
        self.all_endpoints = loaded_config.get("endpoints", [])
        self.config = EndpointConfig()
        self.stats = TestSession()
        self.display = StreamingDisplay(args.debug)
        self.http_session = requests.Session()
        self.tools_enabled = args.tools
        self.switch_endpoint(args.endpoint)
        if self.config and args.model:
            self.config.model_name = args.model

    def switch_endpoint(self, name: str):
        endpoint_data = next(
            (e for e in self.all_endpoints if e["name"].lower() == name.lower()), None
        )
        if not endpoint_data:
            print(f"❌ Error: Endpoint '{name}' not found in configuration file.")
            return
        api_key = os.getenv(endpoint_data["api_key_env"])
        if not api_key:
            print(
                f"❌ Error: API key for '{name}' not found. Set {endpoint_data['api_key_env']}."
            )
            return
        self.config.name = endpoint_data["name"]
        self.config.base_url = endpoint_data["base_url"]
        self.config.api_key = api_key
        self.config.chat_adapter = ADAPTER_MAP.get(endpoint_data.get("chat_adapter"))
        self.config.completion_adapter = ADAPTER_MAP.get(
            endpoint_data.get("completion_adapter")
        )
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.http_session.mount("https://", adapter)
        self.http_session.mount("http://", adapter)
        self.http_session.headers.update(
            {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
        )
        print(f"✅ Switched to endpoint: {self.config.name}")

    def generate(
        self, request: Union[CompletionRequest, ChatRequest], verbose: bool
    ) -> Dict[str, Any]:
        is_chat = isinstance(request, ChatRequest)
        adapter = (
            self.config.chat_adapter if is_chat else self.config.completion_adapter
        )
        if not adapter:
            raise APIError(
                f"Endpoint '{self.config.name}' does not support {'chat' if is_chat else 'completion'} mode."
            )
        if verbose:
            print(
                f"\n🚀 Sending request via endpoint '{self.config.name}' using adapter '{type(adapter).__name__}'"
            )
            print(f"📝 Model: {self.config.model_name}")
            print(
                f"🎛️ Parameters: temp={request.temperature}, max_tokens={request.max_tokens}, stream={request.stream}"
            )

        context = ProtocolContext(
            http_session=self.http_session,
            display=self.display,
            stats=self.stats,
            config=self.config,
            tools_enabled=self.tools_enabled,
        )
        return adapter.generate(context, request, verbose)


def print_banner():
    print("🪶 Polyglot AI: A Universal CLI for the OpenAI API Format 🪶")


def print_stats(stats: TestSession):
    stat_dict = stats.get_stats()
    print("\n📊 SESSION STATISTICS\n" + "=" * 50)
    for key, value in stat_dict.items():
        print(f"{key.replace('_', ' ').title():<22}{value}")
    print("=" * 50)


def closing(stats: TestSession):
    print("\n\n📊 Final Statistics:")
    print_stats(stats)
    print("\n👋 Session ended.")
    sys.exit(0)


def save_conversation_formats(conversation: "Conversation", session_dir: pathlib.Path):
    """Serializes a conversation to multiple HTML formats using Jinja2 templates."""
    try:
        script_dir = pathlib.Path(__file__).parent
        template_dir = script_dir / "templates"

        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )
        env.filters["prettyjson"] = lambda v: json.dumps(v, indent=2)
        history = conversation.get_history()

        # Defines the templates to render and their output filenames.
        formats = {
            "conversation.html": "conversation.html",
            "gptwink_format.html": "gptwink_conversation.html",
        }

        for template_name, output_filename in formats.items():
            try:
                template = env.get_template(template_name)
                final_html = template.render(
                    conversation_id=conversation.conversation_id, history=history
                )
                output_path = session_dir / output_filename
                output_path.write_text(final_html, encoding="utf-8")
            except Exception as e:
                print(f"\n⚠️ Warning: Could not render template '{template_name}': {e}")
                # Don't fallback here, just try the next template

    except Exception as e:
        print(f"\n⚠️ Warning: Could not initialize template environment: {e}")
        # As a global fallback, just write the raw turn data as JSON.
        try:
            all_turns = [turn.to_dict() for turn in conversation.turns]
            fallback_path = session_dir / "conversation_fallback.json"
            fallback_path.write_text(json.dumps(all_turns, indent=2), encoding="utf-8")
            print(f"  -> Fallback data saved to {fallback_path}")
        except Exception as fallback_e:
            print(f"  -> Could not even save fallback JSON: {fallback_e}")


def get_toolbar_text(
    client: "PolyglotClient", args: argparse.Namespace, session_dir: pathlib.Path
) -> HTML:
    """Generates the HTML for the bottom toolbar in interactive mode."""
    endpoint = client.config.name
    model = client.config.model_name
    total_tokens = client.stats.total_tokens_sent + client.stats.total_tokens_received

    # Session tokens/sec
    session_tok_per_sec = client.stats.total_tokens_received / max(
        client.stats.total_response_time, 1
    )

    # Last request stats
    last_ttft = client.stats.last_ttft
    last_tok_per_sec = (
        client.stats.last_tokens_received / max(client.stats.last_response_time, 1)
        if client.stats.last_response_time > 0
        else 0
    )

    tools_status = "ON" if client.tools_enabled else "OFF"
    debug_status = "ON" if client.display.debug_mode else "OFF"
    stream_status = "ON" if args.stream else "OFF"
    interface_mode = "Chat" if args.chat else "Completion"

    # Using prompt_toolkit's style attributes for colors
    # ref: https://python-prompt-toolkit.readthedocs.io/en/master/pages/printing_text.html#style-attributes
    parts = [
        f"<b><style bg='ansiblack' fg='white'> {endpoint}:{model} </style></b>",
        f"<b>Total Tokens:</b> {total_tokens}",
        f"<b>Session Tok/s:</b> {session_tok_per_sec:.1f}",
        f"<b>Last TTFT:</b> {last_ttft:.2f}s",
        f"<b>Last Tok/s:</b> {last_tok_per_sec:.1f}",
        f"<b>Tools:</b> {tools_status}",
        f"<b>Debug:</b> {debug_status}",
        f"<b>Stream:</b> {stream_status}",
        f"<b>Mode:</b> {interface_mode}",
        f"<style fg='grey'>{session_dir}</style>",
    ]

    return HTML(" ∙ ".join(parts))


def print_help():
    print("""
🔧 AVAILABLE COMMANDS (ALL ORIGINAL FEATURES ARE PRESENT):
  /help                  - Show this help message
  /stats                 - Show session statistics
  /quit, /exit, /q       - Exit the program
  /endpoints             - List available endpoints from config file
  /switch <name>         - Switch to a different endpoint
  /model <name>          - Change the default model for the session
  /temp <value>          - Change temperature for subsequent requests (e.g., /temp 0.9)
  /tokens <num>          - Change max_tokens for subsequent requests (e.g., /tokens 100)
  /stream                - Toggle streaming mode on/off
  /verbose               - Toggle verbose request logging on/off
  /debug                 - Toggle raw protocol debug mode on/off
  /tools                 - Toggle tool calling on/off (requires chat mode)
  
  --- Chat Mode Only ---
  /system <text>         - Set a new system prompt (clears history)
  /history               - Show conversation history
  /clear                 - Clear conversation history
    """)


def interactive_mode(client: PolyglotClient, args: argparse.Namespace):
    is_chat_mode = args.chat
    conversation = Conversation()
    if is_chat_mode and args.system:
        conversation.set_system_prompt(args.system)

    # --- Session Persistence ---
    session_dir = pathlib.Path("sessions") / datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S-interactive"
    )
    session_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"🎯 {'Chat' if is_chat_mode else 'Completion'} Mode | Endpoint: {client.config.name} | Model: {client.config.model_name}"
    )
    print(f"💾 Session logs will be saved to: {session_dir}")
    print("Type '/help' for commands, '/quit' to exit.")
    print("-" * 60)

    while True:
        try:
            user_input = session.prompt(
                f"\n👤 ({client.config.name}) User: ",
                bottom_toolbar=lambda: get_toolbar_text(client, args, session_dir),
                refresh_interval=0.5,
            ).strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                parts = user_input[1:].lower().split(" ", 1)
                cmd, params = parts[0], parts[1] if len(parts) > 1 else ""
                if cmd in ["quit", "exit", "q"]:
                    break
                elif cmd == "stats":
                    print_stats(client.stats)
                elif cmd == "help":
                    print_help()
                elif cmd == "endpoints":
                    print("Available Endpoints:")
                    [print(f" - {ep['name']}") for ep in client.all_endpoints]
                elif cmd == "switch" and params:
                    client.switch_endpoint(params)
                elif cmd == "model" and params:
                    client.config.model_name = params
                    print(f"✅ Model set to: {params}")
                elif cmd == "temp" and params:
                    try:
                        args.temperature = float(params)
                        print(
                            f"✅ Temperature for next request set to: {args.temperature}"
                        )
                    except ValueError:
                        print("❌ Invalid value.")
                elif cmd == "tokens" and params:
                    try:
                        args.max_tokens = int(params)
                        print(
                            f"✅ Max tokens for next request set to: {args.max_tokens}"
                        )
                    except ValueError:
                        print("❌ Invalid value.")
                elif cmd == "stream":
                    args.stream = not args.stream
                    print(f"✅ Streaming {'enabled' if args.stream else 'disabled'}.")
                elif cmd == "verbose":
                    args.verbose = not args.verbose
                    print(
                        f"✅ Verbose mode {'enabled' if args.verbose else 'disabled'}."
                    )
                elif cmd == "debug":
                    client.display.debug_mode = not client.display.debug_mode
                    print(
                        f"✅ Debug mode {'enabled' if client.display.debug_mode else 'disabled'}."
                    )
                elif cmd == "tools":
                    client.tools_enabled = not client.tools_enabled
                    print(
                        f"✅ Tool calling {'enabled' if client.tools_enabled else 'disabled'}."
                    )
                elif is_chat_mode and cmd == "history":
                    print(json.dumps(conversation.get_history(), indent=2))
                elif is_chat_mode and cmd == "clear":
                    conversation.clear()
                    print("🧹 History cleared.")
                elif is_chat_mode and cmd == "system" and params:
                    conversation.set_system_prompt(params)
                    print(f"🤖 System prompt set.")
                else:
                    print("❌ Unknown command.")
                continue

            request: Union[ChatRequest, CompletionRequest]
            if is_chat_mode:
                next_messages = conversation.get_messages_for_next_turn(user_input)
                request = ChatRequest(
                    messages=next_messages,
                    model=client.config.model_name,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stream=args.stream,
                )
            else:
                # Completion mode does not have conversation history
                request = CompletionRequest(
                    prompt=user_input,
                    model=client.config.model_name,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stream=args.stream,
                )

            result = client.generate(request, args.verbose)

            # For chat mode, create a Turn and add it to the conversation
            if is_chat_mode and result:
                turn = Turn(
                    request_data=result.get("request", {}),
                    response_data=result.get("response", {}),
                    assistant_message=result.get("text", ""),
                )
                conversation.add_turn(turn)

                # --- Save Turn & Conversation ---
                try:
                    # Save the raw Turn object to its own file
                    turn_file = session_dir / f"{turn.turn_id}-turn.json"
                    with open(turn_file, "w", encoding="utf-8") as f:
                        json.dump(turn.to_dict(), f, indent=2)

                    # Save the full conversation to all configured HTML formats
                    save_conversation_formats(conversation, session_dir)
                except Exception as e:
                    print(f"\n⚠️  Warning: Could not save session turn: {e}")

        except KeyboardInterrupt:
            client.display.spinner.stop()
            print()
            continue
        except EOFError:
            break
        except Exception as e:
            client.display.spinner.stop()
            print(f"\n❌ ERROR: {e}")
    closing(client.stats)


def main():
    parser = argparse.ArgumentParser(
        description="Polyglot AI: A Universal CLI for the OpenAI API Format",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # The rest of the main function is identical to the last correct version.
    parser.add_argument(
        "--config", default="polyglot.toml", help="Path to the TOML configuration file."
    )
    parser.add_argument(
        "--endpoint",
        default="openai",
        help="The name of the endpoint from the config file to use.",
    )
    parser.add_argument("--model", help="Override the default model for the session.")
    parser.add_argument(
        "--chat", action="store_true", help="Enable chat mode. Required for tool use."
    )
    parser.add_argument("-p", "--prompt", help="Send a single prompt and exit.")
    parser.add_argument("--system", help="Set a system prompt for chat mode.")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--no-stream", action="store_false", dest="stream")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging of request parameters.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable raw protocol debug mode for streaming.",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Enable tool calling feature (requires chat mode).",
    )
    args = parser.parse_args()

    print_banner()
    try:
        with open(args.config, "r") as f:
            loaded_config = toml.load(f)
    except FileNotFoundError:
        sys.exit(f"❌ FATAL: Config file not found at '{args.config}'")
    except Exception as e:
        sys.exit(f"❌ FATAL: Could not parse '{args.config}': {e}")

    try:
        client = PolyglotClient(args, loaded_config)
        if args.prompt:
            if args.chat:
                messages = (
                    [{"role": "system", "content": args.system}] if args.system else []
                )
                messages.append({"role": "user", "content": args.prompt})
                request = ChatRequest(
                    messages=messages,
                    model=client.config.model_name,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stream=args.stream,
                )
            else:
                request = CompletionRequest(
                    prompt=args.prompt,
                    model=client.config.model_name,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stream=args.stream,
                )
            client.generate(request, args.verbose)
            print_stats(client.stats)
        else:
            interactive_mode(client, args)
    except (APIError, ValueError, ConnectionError) as e:
        sys.exit(f"❌ Error: {e}")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

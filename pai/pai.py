"""
Polyglot AI: An Interactive, Multi-Provider CLI for the OpenAI API Format.
This version is a direct refactoring of the original code, preserving all features
and fixing the circular import error.
"""

import asyncio
import json
import os
import pathlib
import re
import sys
import time
from datetime import datetime
from html import escape
from typing import Any, Optional

import httpx
import toml
import typer
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.defaults import load_key_bindings
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.widgets import SearchToolbar

from .commands import CommandHandler
from .log_utils import closing, print_stats, save_conversation_formats
from .models import (
    ArenaState,
    ChatRequest,
    CompletionRequest,
    Conversation,
    EndpointConfig,
    PolyglotConfig,
    RequestStats,
    RuntimeConfig,
    TestSession,
    Turn,
)

# --- Protocol Adapter Imports ---
from .protocols.base_adapter import ProtocolContext
from .protocols.legacy_completion_adapter import LegacyCompletionAdapter
from .protocols.openai_chat_adapter import OpenAIChatAdapter
from .tools import get_tool_schemas
from .utils import estimate_tokens

# --- Global Definitions ---
session = PromptSession()
ADAPTER_MAP = {
    "openai_chat": OpenAIChatAdapter(),
    "legacy_completion": LegacyCompletionAdapter(),
}


# --- Data Models ---
# All data classes for state, history, and requests have been moved to `pai/models.py`
# to improve modularity and break circular dependencies.


# --- [REFACTORED] StreamingDisplay for Stable UI ---
class StreamingDisplay:
    """Manages all console output, ensuring prompt-toolkit UI is not corrupted."""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self._printer = print  # Default to standard print
        self._is_interactive = False
        self.output_buffer: Buffer | None = None
        self.actor_name = "🤖 Assistant"

        # State for UI
        self.status = "Idle"
        # Holds the stats for the *in-progress* request.
        self.current_request_stats: RequestStats | None = None
        self._last_token_time: float | None = None

        # Response tracking
        self.line_count: int = 0
        self.chunk_count: int = 0
        self.current_response: str = ""
        self.first_token_received = False

    def set_printer(self, printer: callable, is_interactive: bool):
        """Sets the function used for printing to the console."""
        self._printer = printer
        self._is_interactive = is_interactive

    def start_response(self, tokens_sent: int = 0, actor_name: str | None = None):
        """Prepares for a new response stream."""
        self.current_response = ""
        if self.output_buffer:
            self.output_buffer.reset()

        self.actor_name = actor_name or "🤖 Assistant"
        self.current_request_stats = RequestStats(tokens_sent=tokens_sent)
        self._last_token_time = None
        self.line_count = 0
        self.chunk_count = 0
        self.first_token_received = False
        self.status = "Waiting..."

    @property
    def time_since_last_token(self) -> float:
        """Returns seconds since the last token was received, for spinner UI."""
        if self.status == "Streaming" and self._last_token_time:
            return time.time() - self._last_token_time
        return 0.0

    def _print(self, *args, **kwargs):
        """Internal print-router."""
        # In interactive mode, self._printer is prompt_toolkit's thread-safe
        # print_formatted_text function, which handles redrawing the prompt.
        self._printer(*args, **kwargs)

    def show_raw_line(self, line: str):
        if self.debug_mode:
            if not self.first_token_received:
                self._print("🔍 DEBUG MODE: Showing raw protocol traffic\n" + "=" * 60)
                self.first_token_received = (
                    True  # Prevent header from printing multiple times
                )
            self.line_count += 1
            duration = (
                self.current_request_stats.current_duration
                if self.current_request_stats
                else 0
            )
            prefix = f"⚪ [{duration:6.2f}s] L{self.line_count:03d}: "
            if line.startswith("data: "):
                prefix = f"🔵 [{duration:6.2f}s] L{self.line_count:03d}: "
            self._print(f"{prefix}{repr(line)}")

    def show_parsed_chunk(self, chunk_data: dict, chunk_text: str):
        """Handles printing a parsed chunk of text from the stream."""
        # Don't do anything for empty chunks, which some providers send.
        if not chunk_text:
            return

        self._last_token_time = time.time()

        # Update state first
        if not self.first_token_received:
            self.status = "Streaming"
            if self.current_request_stats:
                self.current_request_stats.record_first_token()
            self.first_token_received = True

        self.current_response += chunk_text
        self.chunk_count += 1
        if self.current_request_stats:
            # To ensure the live count is consistent with the final count,
            # we recalculate from the full response string on every chunk.
            self.current_request_stats.tokens_received = estimate_tokens(
                self.current_response
            )

        # Handle rendering
        if self._is_interactive and self.output_buffer:
            # For interactive mode with a buffer, update the buffer's content.
            # This will be displayed live in the UI's live output window.
            self.output_buffer.text = f"{self.actor_name}: {self.current_response}"
            # Move the cursor to the end of the buffer to ensure scrolling.
            self.output_buffer.cursor_position = len(self.output_buffer.text)
        elif not self._is_interactive:
            # For non-interactive, print header once, then stream chunks.
            if self.chunk_count == 1:
                self._print(f"\n{self.actor_name}: ", end="")
            self._print(chunk_text, end="", flush=True)

        if self.debug_mode:
            duration = (
                self.current_request_stats.current_duration
                if self.current_request_stats
                else 0
            )
            self._print(
                f"🟢 [{duration:6.2f}s] C{self.chunk_count:03d} TEXT: {repr(chunk_text)}"
            )

    def finish_response(self, success: bool = True) -> RequestStats | None:
        """Finalizes the response, prints stats, and resets the display state."""
        self.status = "Done"
        if self.current_request_stats:
            self.current_request_stats.finish(success=success)
            # The token count is now calculated live. This final assignment
            # ensures it's perfectly accurate based on the complete final string,
            # but the live value should already match this.
            self.current_request_stats.tokens_received = estimate_tokens(
                self.current_response
            )

        # In interactive mode, if we have a response, print it to the scrollback
        # history. This "finalizes" it, moving it from the temporary live
        # buffer to the main conversation transcript. This happens regardless
        # of success to ensure partial/cancelled outputs are preserved.
        if self._is_interactive and self.current_response:
            self._print(
                HTML(f"{escape(self.actor_name)}: {escape(self.current_response)}")
            )

        # On success, print final stats.
        if success and self.current_request_stats:
            stats = self.current_request_stats
            if self.debug_mode:
                self._print(
                    "=" * 60
                    + f"\n🔍 DEBUG SUMMARY: {self.line_count} lines, {self.chunk_count} chunks, {stats.response_time:.2f}s\n"
                    + "=" * 60
                )
            elif not self._is_interactive and self.first_token_received:
                # For non-interactive mode, print the final stats line.
                tok_per_sec = stats.final_tok_per_sec
                ttft_str = (
                    f" | TTFT: {stats.ttft:.2f}s" if stats.ttft is not None else ""
                )
                self._print(
                    f"\n\n📊 Response in {stats.response_time:.2f}s ({stats.tokens_received} tokens, {tok_per_sec:.1f} tok/s{ttft_str})"
                )

        # The buffer is reset by start_response(), so we just reset status here.
        # Keeping the buffer allows the final response to remain visible.
        self.status = "Idle"
        return self.current_request_stats


class APIError(Exception):
    pass


class PolyglotClient:
    def __init__(
        self,
        runtime_config: RuntimeConfig,
        toml_config: PolyglotConfig,
        http_session: httpx.AsyncClient,
    ):
        self.toml_config = toml_config
        self.config = EndpointConfig()
        self.stats = TestSession()
        self.display = StreamingDisplay(runtime_config.debug)
        self.http_session = http_session
        self.tools_enabled = runtime_config.tools
        self.switch_endpoint(runtime_config.endpoint)
        if self.config and runtime_config.model:
            self.config.model_name = runtime_config.model
        if self.config and runtime_config.timeout:
            self.config.timeout = runtime_config.timeout

    def switch_endpoint(self, name: str):
        endpoint_data = next(
            (e for e in self.toml_config.endpoints if e.name.lower() == name.lower()),
            None,
        )
        if not endpoint_data:
            self.display._print(
                f"❌ Error: Endpoint '{name}' not found in configuration file."
            )
            return
        api_key = os.getenv(endpoint_data.api_key_env)
        if not api_key:
            self.display._print(
                f"❌ Error: API key for '{name}' not found. Set {endpoint_data.api_key_env}."
            )
            return
        self.config.name = endpoint_data.name
        self.config.base_url = endpoint_data.base_url
        self.config.api_key = api_key
        self.config.chat_adapter = ADAPTER_MAP.get(endpoint_data.chat_adapter)
        self.config.completion_adapter = ADAPTER_MAP.get(
            endpoint_data.completion_adapter
        )
        # Configure the httpx client for the selected endpoint
        self.http_session.base_url = self.config.base_url
        self.http_session.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "PolyglotAI/0.1.0",
        }
        self.display._print(f"✅ Switched to endpoint: {self.config.name}")

    async def generate(
        self,
        request: CompletionRequest | ChatRequest,
        verbose: bool,
        actor_name: str | None = None,
    ) -> dict[str, Any]:
        is_chat = isinstance(request, ChatRequest)
        adapter = (
            self.config.chat_adapter if is_chat else self.config.completion_adapter
        )
        if not adapter:
            raise APIError(
                f"Endpoint '{self.config.name}' does not support {'chat' if is_chat else 'completion'} mode."
            )
        if verbose:
            self.display._print(
                f"\n🚀 Sending request via endpoint '{self.config.name}' using adapter '{type(adapter).__name__}'"
            )
            self.display._print(f"📝 Model: {self.config.model_name}")
            self.display._print(
                f"🎛️ Parameters: temp={request.temperature}, max_tokens={request.max_tokens}, stream={request.stream}"
            )

        context = ProtocolContext(
            http_session=self.http_session,
            display=self.display,
            stats=self.stats,
            config=self.config,
            tools_enabled=self.tools_enabled,
        )
        return await adapter.generate(context, request, verbose, actor_name=actor_name)


def print_banner():
    print("🪶 Polyglot AI: A Universal CLI for the OpenAI API Format 🪶")


# Logging and statistics functions have been moved to `pai/log_utils.py`.


class InteractiveUI:
    """Encapsulates all logic for the text user interface."""

    def __init__(self, client: "PolyglotClient", runtime_config: RuntimeConfig):
        self.client = client
        self.runtime_config = runtime_config
        self.is_chat_mode = runtime_config.chat
        self.conversation = Conversation()
        if self.is_chat_mode and self.runtime_config.system:
            self.conversation.set_system_prompt(self.runtime_config.system)

        # Setup directories
        self.session_dir = pathlib.Path("sessions") / datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S-interactive"
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        pai_user_dir = pathlib.Path.home() / ".pai"
        pai_user_dir.mkdir(exist_ok=True)
        self.prompts_dir = pathlib.Path("prompts")
        self.prompts_dir.mkdir(exist_ok=True)

        # Setup UI components
        self.pt_printer = print_formatted_text
        self.client.display.set_printer(self.pt_printer, is_interactive=True)
        self.history = FileHistory(str(pai_user_dir / "history.txt"))
        self.streaming_output_buffer = Buffer()
        self.client.display.output_buffer = self.streaming_output_buffer

        self.input_buffer = Buffer(
            name="input_buffer",
            multiline=False,
            history=self.history,
            enable_history_search=True,
            accept_handler=self._on_buffer_accepted,
        )

        # State management
        self.native_agent_mode = False
        self.legacy_agent_mode = False
        # Arena state
        self.arena_state: ArenaState | None = None
        self.arena_paused_event = asyncio.Event()

        self.generation_in_progress = asyncio.Event()
        self.generation_task: asyncio.Task | None = None
        self.spinner_chars = ["|", "/", "-", "\\"]
        self.spinner_idx = 0

        # Build the application
        self.app = self._create_application()

        # Command handler
        self.command_handler = CommandHandler(self)

    def _get_mode_display_name(self) -> str:
        """Returns the string name of the current interaction mode."""
        if self.arena_state:
            status = (
                "Paused"
                if not self.arena_paused_event.is_set()
                and self.generation_in_progress.is_set()
                else "Running"
            )
            judge_str = " w/ Judge" if self.arena_state.arena_config.judge else ""
            return f"Arena: {self.arena_state.arena_config.name}{judge_str} ({status})"
        if self.native_agent_mode:
            return "Agent"
        if self.legacy_agent_mode:
            return "Legacy Agent"
        if self.is_chat_mode:
            return "Chat"
        return "Completion"

    def _create_application(self) -> Application:
        """Constructs the prompt_toolkit Application object."""

        # This is the main input bar at the bottom of the screen.
        def get_prompt_text() -> HTML:
            # If we are in arena mode, but there is no active generation task,
            # it means we are waiting for the very first prompt from the user.
            if self.arena_state and not self.generation_in_progress.is_set():
                initiator = self.arena_state.arena_config.get_initiator()
                # Use a specific prompt for the arena's first turn
                return HTML(
                    f"<style fg='ansigreen'>⚔️  Prompt for {initiator.name}:</style> "
                )
            return HTML(
                f"<style fg='ansigreen'>👤 ({self._get_mode_display_name()}) User:</style> "
            )

        prompt_ui = VSplit(
            [
                Window(
                    FormattedTextControl(get_prompt_text),
                    # The width must calculate the length of the *unformatted* string.
                    width=lambda: len(re.sub("<[^<]+?>", "", get_prompt_text().value))
                    + 1,
                ),
                Window(BufferControl(buffer=self.input_buffer)),
            ]
        )

        def get_status_text():
            if self.arena_state and not self.arena_paused_event.is_set():
                return HTML(
                    "<style fg='ansiyellow'>[--- Arena Paused --- Use /resume or /say &lt;...&gt; ---]</style>"
                )

            display = self.client.display
            status = display.status
            live_stats = display.current_request_stats

            if status == "Streaming" and live_stats:
                spinner = self.spinner_chars[self.spinner_idx]
                self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_chars)
                time_since_last = display.time_since_last_token
                color = "ansigreen"
                if time_since_last > 2.0:
                    color = "ansiyellow"
                if time_since_last > 5.0:
                    color = "ansired"
                return HTML(
                    f"<style fg='{color}'>[{spinner}] Streaming... ({time_since_last:.1f}s since last token)</style>"
                )
            elif status == "Waiting..." and live_stats:
                duration = live_stats.current_duration
                if duration > 5.0:
                    # After 5 seconds, show a timer to give feedback on long waits.
                    return HTML(
                        f"<style fg='ansiyellow'>[Sent. Waiting for response... ({duration:.0f}s)]</style>"
                    )
                return HTML("<style fg='ansiyellow'>[Waiting for response...]</style>")
            # Fallback for idle or other states where this component might be briefly visible.
            return HTML("")

        waiting_ui = Window(FormattedTextControl(get_status_text), height=1)

        live_output_window = ConditionalContainer(
            Window(
                content=BufferControl(buffer=self.streaming_output_buffer),
                wrap_lines=True,
            ),
            filter=Condition(
                lambda: self.generation_in_progress.is_set()
                and self.streaming_output_buffer.text
            ),
        )

        search_toolbar = SearchToolbar()
        toolbar_window = Window(
            content=FormattedTextControl(self._get_toolbar_text),
            height=3,
            style="reverse",
        )

        layout = Layout(
            HSplit(
                [
                    live_output_window,
                    ConditionalContainer(
                        prompt_ui,
                        filter=Condition(
                            lambda: not self.generation_in_progress.is_set()
                            or (
                                self.arena_state
                                and not self.arena_paused_event.is_set()
                            )
                        ),
                    ),
                    ConditionalContainer(
                        waiting_ui,
                        filter=Condition(lambda: self.generation_in_progress.is_set()),
                    ),
                    search_toolbar,
                    toolbar_window,
                ]
            ),
            focused_element=self.input_buffer,
        )

        return Application(
            layout=layout,
            key_bindings=self._create_key_bindings(),
            refresh_interval=0.2,
            full_screen=False,
        )

    def _create_key_bindings(self) -> KeyBindings:
        """Creates key bindings, including custom Ctrl+C and Ctrl+D handlers."""
        kb = KeyBindings()

        @kb.add("c-c", eager=True)
        def _(event):
            """
            Custom Ctrl+C handler.
            - If a generation is in progress, cancel it.
            - If the input buffer has text, clear it.
            - If the input buffer is empty, print a fake prompt to emulate a new line.
            """
            if self.generation_in_progress.is_set() and self.generation_task:
                self.generation_task.cancel()
            else:
                if event.app.current_buffer.text:
                    event.app.current_buffer.reset()
                else:
                    # To create a "new line" effect, we print a line that looks
                    # like our prompt to the scrollback buffer. The application
                    # will then redraw the actual interactive prompt below it.
                    self.pt_printer(
                        HTML(
                            f"<style fg='ansigreen'>👤 ({self._get_mode_display_name()}) User:</style> "
                        )
                    )

        @kb.add("c-d", filter=Condition(lambda: not self.input_buffer.text), eager=True)
        def _(event):
            """Handle Ctrl+D on an empty buffer to exit cleanly."""
            event.app.exit(result=EOFError())

        # Merge our bindings with the defaults. Ours take precedence.
        return merge_key_bindings([kb, load_key_bindings()])

    def _on_buffer_accepted(self, buffer: Buffer):
        """Callback for when the user presses Enter on the input buffer."""
        # An active task can only be interrupted if it's a paused arena.
        is_paused_arena = self.arena_state and not self.arena_paused_event.is_set()
        if self.generation_in_progress.is_set() and not is_paused_arena:
            return

        user_input = buffer.text
        stripped_input = user_input.strip()

        # Always add non-empty input to history manually for consistent behavior.
        if stripped_input:
            buffer.reset(append_to_history=True)
        else:
            buffer.reset()
            self.pt_printer(
                HTML(
                    f"<style fg='ansigreen'>👤 ({self._get_mode_display_name()}) User:</style> "
                )
            )
            return

        # Print the user's input to the log area.
        self.pt_printer(
            HTML(
                f"\n<style fg='ansigreen'>👤 ({self._get_mode_display_name()}) User:</style> {escape(user_input)}"
            )
        )

        if stripped_input.startswith("/"):
            self.command_handler.handle(stripped_input, self.app)
        else:
            # Handle plain text input.
            if self.arena_state:
                if not self.generation_task:
                    # This is the initial prompt that kicks off the arena.
                    self.arena_state.last_message = stripped_input
                    self.generation_task = asyncio.create_task(
                        self._run_arena_orchestrator()
                    )
                    self.arena_paused_event.set()  # Start the loop running.
                else:
                    # The user has typed plain text into a paused arena. This is
                    # an invalid action, so we guide them.
                    self.pt_printer(
                        HTML(
                            "<style fg='ansiyellow'>ℹ️ Arena is paused. Use /say &lt;message&gt; to interject, or /resume to continue.</style>"
                        )
                    )
            else:
                # Standard generation outside of arena mode.
                self.generation_task = asyncio.create_task(
                    self._process_and_generate(stripped_input)
                )

    # Command handling logic has been extracted to `pai/commands.py`.

    async def _process_and_generate(self, user_input_str: str):
        self.generation_in_progress.set()
        request: ChatRequest | CompletionRequest | None = None
        try:
            if self.is_chat_mode and self.legacy_agent_mode:
                await self._run_legacy_agent_loop(user_input_str)
            else:
                if self.is_chat_mode:
                    messages = self.conversation.get_messages_for_next_turn(
                        user_input_str
                    )
                    request = ChatRequest(
                        messages=messages,
                        model=self.client.config.model_name,
                        max_tokens=self.runtime_config.max_tokens,
                        temperature=self.runtime_config.temperature,
                        stream=self.runtime_config.stream,
                        tools=get_tool_schemas() if self.client.tools_enabled else [],
                    )
                else:
                    request = CompletionRequest(
                        prompt=user_input_str,
                        model=self.client.config.model_name,
                        max_tokens=self.runtime_config.max_tokens,
                        temperature=self.runtime_config.temperature,
                        stream=self.runtime_config.stream,
                    )

                result = await self.client.generate(
                    request, self.runtime_config.verbose
                )

                if self.is_chat_mode and result:
                    turn = Turn(
                        request_data=result.get("request", {}),
                        response_data=result.get("response", {}),
                        assistant_message=result.get("text", ""),
                    )
                    request_stats = self.client.stats.last_request_stats
                    self.conversation.add_turn(turn, request_stats)
                    try:
                        turn_file = self.session_dir / f"{turn.turn_id}-turn.json"
                        turn_file.write_text(
                            json.dumps(turn.to_dict(), indent=2), encoding="utf-8"
                        )
                        save_conversation_formats(
                            self.conversation,
                            self.session_dir,
                            printer=self.pt_printer,
                        )
                    except Exception as e:
                        self.pt_printer(
                            f"\n⚠️  Warning: Could not save session turn: {e}"
                        )
        except asyncio.CancelledError:
            # When cancelled, we still want to save the partial response.
            # finish_response() sets success=False and returns the incomplete stats.
            request_stats = self.client.display.finish_response(success=False)
            partial_text = self.client.display.current_response

            # Add to session stats.
            if request_stats:
                request_stats.finish_reason = "cancelled"
                # This is an approximation of tokens_sent as the final request
                # payload is constructed inside the adapter.
                if request:  # We can only save if we have the request object
                    if isinstance(request, ChatRequest):
                        request_stats.tokens_sent = sum(
                            estimate_tokens(m.get("content", ""))
                            for m in request.messages
                        )
                    elif isinstance(request, CompletionRequest):
                        request_stats.tokens_sent = estimate_tokens(request.prompt)
                    self.client.stats.add_completed_request(request_stats)

                    if self.is_chat_mode and partial_text:
                        # Create a turn with the partial data for logging.
                        request_data = request.to_dict(self.client.config.model_name)
                        # Fabricate a partial response object for logging
                        response_data = {
                            "pai_note": "This response was cancelled by the user.",
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": partial_text,
                                    }
                                }
                            ],
                        }
                        turn = Turn(
                            request_data=request_data,
                            response_data=response_data,
                            assistant_message=partial_text,
                        )
                        self.conversation.add_turn(turn)
                        try:
                            turn_file = self.session_dir / f"{turn.turn_id}-turn.json"
                            turn_file.write_text(
                                json.dumps(turn.to_dict(), indent=2), encoding="utf-8"
                            )
                            save_conversation_formats(
                                self.conversation,
                                self.session_dir,
                                printer=self.pt_printer,
                            )
                        except Exception as e:
                            self.pt_printer(
                                f"\n⚠️  Warning: Could not save cancelled session turn: {e}"
                            )

            self.pt_printer(
                HTML("\n<style fg='ansiyellow'>🚫 Generation cancelled.</style>")
            )
        except Exception as e:
            self.client.display.finish_response(success=False)
            self.pt_printer(HTML(f"<style fg='ansired'>❌ ERROR: {e}</style>"))
        finally:
            self.generation_in_progress.clear()
            self.generation_task = None

    async def _run_legacy_agent_loop(self, user_input_str: str):
        from .tools import execute_tool

        messages = self.conversation.get_messages_for_next_turn(user_input_str)
        max_loops = 5
        used_tools_in_loop = False

        for i in range(max_loops):
            request = ChatRequest(
                messages=messages,
                model=self.client.config.model_name,
                max_tokens=self.runtime_config.max_tokens,
                temperature=self.runtime_config.temperature,
                stream=self.runtime_config.stream,
                tools=[],  # Legacy mode does not use native tools
            )

            # In legacy agent mode, we stream the response to the user for a better
            # experience. The full response text is still returned from client.generate()
            # which we can then parse for tool calls after the stream is complete.
            result = await self.client.generate(request, self.runtime_config.verbose)
            assistant_response = result.get("text", "")

            # Add the assistant's raw response to the message history
            messages.append({"role": "assistant", "content": assistant_response})

            tool_match = re.search(
                r"<tool_call>(.*?)</tool_call>", assistant_response, re.DOTALL
            )

            if tool_match:
                used_tools_in_loop = True
                self.pt_printer(
                    HTML(
                        "\n<style fg='ansimagenta'>🔧 Agent wants to use a tool...</style>"
                    )
                )
                # The full tool_call text is printed by the generate() function's
                # display handler. We can't easily suppress it, but we can clear
                # it from the display object's memory so it isn't logged as a "final"
                # assistant message if the session is cancelled here.
                self.client.display.current_response = ""
                tool_xml = tool_match.group(1)
                name_match = re.search(r"<name>(.*?)</name>", tool_xml)
                args_match = re.search(r"<args>(.*?)</args>", tool_xml, re.DOTALL)

                if not name_match or not args_match:
                    tool_result = (
                        "Error: Invalid tool_call format. Missing <name> or <args> tag."
                    )
                else:
                    tool_name = name_match.group(1).strip()
                    tool_args_str = args_match.group(1).strip()
                    try:
                        tool_args = json.loads(tool_args_str)
                        self.pt_printer(
                            f"  - Executing: {tool_name}({json.dumps(tool_args)})"
                        )
                        tool_result = execute_tool(tool_name, tool_args)
                    except json.JSONDecodeError:
                        tool_result = (
                            f"Error: Invalid JSON in <args> for tool {tool_name}."
                        )
                    except Exception as e:
                        tool_result = f"Error executing tool {tool_name}: {e}"

                # Append the tool result to the conversation for the next loop
                # We use the 'user' role to make it clear this is external input.
                tool_result_message = f"TOOL_RESULT:\n```\n{tool_result}\n```"
                messages.append({"role": "user", "content": tool_result_message})
                self.pt_printer(
                    HTML(
                        f"<style fg='ansimagenta'>  - Result: {escape(str(tool_result))[:300]}...</style>"
                    )
                )
                # Continue the loop
            else:
                # No tool call found, this is the final answer. `generate()` has already
                # called finish_response(), which handles printing. We just log the turn.
                if used_tools_in_loop:
                    self.pt_printer(
                        HTML(
                            "\n<style fg='ansigreen'>✅ Agent formulated a response using tool results.</style>"
                        )
                    )
                else:
                    self.pt_printer(
                        HTML(
                            "\n<style fg='ansigreen'>✅ Agent decided to respond directly.</style>"
                        )
                    )

                turn = Turn(
                    request_data=result.get("request", {}),
                    response_data=result.get("response", {}),
                    assistant_message=assistant_response,
                )
                request_stats = self.client.stats.last_request_stats
                self.conversation.add_turn(turn, request_stats)
                save_conversation_formats(
                    self.conversation, self.session_dir, printer=self.pt_printer
                )
                break
        else:
            self.pt_printer("⚠️ Agent reached maximum loops.")

    async def _run_arena_orchestrator(self):
        self.generation_in_progress.set()
        original_endpoint_name = self.client.config.name  # Save original state
        try:
            state = self.arena_state
            while state.current_speech < state.max_speeches:
                # This is the core of the pause/resume mechanic.
                # The loop will not proceed until the event is set.
                await self.arena_paused_event.wait()

                participant_id = state.turn_order_ids[0]
                participant = state.arena_config.participants[participant_id]
                turn_num = (state.current_speech // len(state.turn_order_ids)) + 1

                # Switch client endpoint AND model for this participant
                if self.client.config.name.lower() != participant.endpoint.lower():
                    self.client.switch_endpoint(participant.endpoint)
                self.client.config.model_name = participant.model

                # Print turn header
                self.pt_printer(
                    HTML(
                        f"\n<style bg='ansiblue' fg='white'> 🥊 TURN {turn_num} | Participant: {participant.name} ({participant.endpoint}/{participant.model}) </style>"
                    )
                )

                messages = participant.conversation.get_messages_for_next_turn(
                    state.last_message
                )
                request = ChatRequest(
                    messages=messages,
                    model=participant.model,
                    max_tokens=self.runtime_config.max_tokens,
                    temperature=self.runtime_config.temperature,
                    stream=self.runtime_config.stream,
                    tools=get_tool_schemas() if self.client.tools_enabled else [],
                )

                actor_name = f"🤖 {participant.name}"
                result = await self.client.generate(
                    request, self.runtime_config.verbose, actor_name=actor_name
                )
                assistant_message = result.get("text", "")

                # Create a Turn object with all metadata
                turn = Turn(
                    request_data=result.get("request", {}),
                    response_data=result.get("response", {}),
                    assistant_message=assistant_message,
                    participant_name=participant.name,
                    model_name=participant.model,
                )
                request_stats = self.client.stats.last_request_stats

                # Add to the main unified conversation for logging
                self.conversation.add_turn(turn, request_stats)
                # Add to the participant's individual conversation history
                participant.conversation.add_turn(turn, request_stats)
                save_conversation_formats(
                    self.conversation, self.session_dir, printer=self.pt_printer
                )

                # Update state for the next iteration
                state.last_message = assistant_message
                state.current_speech += 1
                state.turn_order_ids = (
                    state.turn_order_ids[1:] + state.turn_order_ids[:1]
                )

            self.pt_printer("\n🏁 Arena finished: Maximum turns reached.")
            await self._run_arena_judge()

        except asyncio.CancelledError:
            self.pt_printer(
                HTML("\n<style fg='ansiyellow'>🚫 Arena cancelled by user.</style>")
            )
        except Exception as e:
            if self.client.display.current_request_stats:
                self.client.display.finish_response(success=False)
            self.pt_printer(HTML(f"<style fg='ansired'>❌ ARENA ERROR: {e}</style>"))
        finally:
            if self.arena_state:
                # If cancelled, still try to run the judge for a summary
                await self._run_arena_judge()
                self.pt_printer(
                    f"\n🏁 Arena '{self.arena_state.arena_config.name}' session ended."
                )

            # Restore client to the original endpoint state
            if self.client.config.name.lower() != original_endpoint_name.lower():
                self.pt_printer(
                    f"✅ Restoring client to original endpoint: {original_endpoint_name}"
                )
                self.client.switch_endpoint(original_endpoint_name)

            # Reset all arena-related state
            self.arena_state = None
            self.generation_in_progress.clear()
            self.generation_task = None

    async def _run_arena_judge(self):
        """If a judge is configured, run it to get a final verdict."""
        if not self.arena_state or not self.arena_state.arena_config.judge:
            return

        judge = self.arena_state.arena_config.judge
        self.pt_printer(
            HTML(
                f"\n<style bg='ansiyellow' fg='black'> 🧑‍⚖️ The Judge is now deliberating... </style>"
            )
        )

        # The judge sees the entire conversation, so we create a new history
        # that includes the judge's own system prompt.
        messages = judge.conversation.get_history()
        # It's a list comprehension, but it gets the `content` of the `user` message
        # that started it all, then the `assistant` message of every turn.
        full_dialogue = "\n\n".join(
            [
                f"{t.participant_name or 'user'}: {t.assistant_message or t.request_data.get('messages', [{}])[-1].get('content', '')}"
                for t in self.conversation.turns
            ]
        )
        messages.append(
            {
                "role": "user",
                "content": f"Here is the full conversation transcript:\n\n---\n{full_dialogue}\n---\n\nPlease provide your summary and verdict.",
            }
        )

        # Switch to the judge's endpoint and model
        if self.client.config.name.lower() != judge.endpoint.lower():
            self.client.switch_endpoint(judge.endpoint)
        self.client.config.model_name = judge.model

        request = ChatRequest(
            messages=messages,
            model=judge.model,
            max_tokens=self.runtime_config.max_tokens,
            temperature=self.runtime_config.temperature,
            stream=self.runtime_config.stream,
        )

        actor_name = f"🧑‍⚖️ {judge.name}"
        result = await self.client.generate(
            request, self.runtime_config.verbose, actor_name=actor_name
        )
        assistant_message = result.get("text", "")

        # Log the judge's turn
        turn = Turn(
            request_data=result.get("request", {}),
            response_data=result.get("response", {}),
            assistant_message=assistant_message,
            participant_name=judge.name,
            model_name=judge.model,
        )
        request_stats = self.client.stats.last_request_stats
        self.conversation.add_turn(turn, request_stats)
        save_conversation_formats(
            self.conversation, self.session_dir, printer=self.pt_printer
        )

    def _get_toolbar_text(self) -> HTML:
        """Generates the HTML for the multi-line bottom toolbar."""
        try:
            client, session_dir = self.client, self.session_dir
            endpoint, model = client.config.name, client.config.model_name
            session_stats, display = client.stats, client.display
            live_stats = display.current_request_stats

            # Escape any potentially problematic strings before embedding in HTML
            endpoint_esc = escape(endpoint)
            model_esc = escape(model)
            status_esc = escape(display.status)
            session_dir_esc = escape(str(session_dir))

            # --- Line 1: Overall Session Stats ---
            total_tokens = (
                session_stats.total_tokens_sent + session_stats.total_tokens_received
            )
            total_time = session_stats.total_response_time
            total_received = session_stats.total_tokens_received

            # Add live data during streaming for a real-time view
            if live_stats and display.status == "Streaming":
                total_tokens += live_stats.tokens_sent + live_stats.tokens_received
                total_time += live_stats.current_duration
                total_received += live_stats.tokens_received

            avg_tok_per_sec = total_received / max(total_time, 1)

            session_tokens = self.conversation.session_token_count
            if live_stats and display.status in ["Waiting...", "Streaming"]:
                session_tokens += live_stats.tokens_sent + live_stats.tokens_received

            if self.arena_state:
                p_configs = self.arena_state.arena_config.participants
                p_details = " vs ".join(
                    [
                        f"{p.name} ({p.model})"
                        for p_id, p in p_configs.items()
                        if p_id != "judge"
                    ]
                )
                judge_str = " w/ Judge" if self.arena_state.arena_config.judge else ""
                arena_name_esc = escape(self.arena_state.arena_config.name)
                p_details_esc = escape(p_details)
                line1 = f"<b><style bg='ansiblue' fg='white'> ⚔️ ARENA: {arena_name_esc}{judge_str} </style></b> | {p_details_esc}"
            else:
                line1 = f"<b><style bg='ansiblack' fg='white'> {endpoint_esc.upper()}:{model_esc} </style></b> | <b>Total:</b> {total_tokens} | <b>Session:</b> {session_tokens} | <b>Avg Tok/s:</b> {avg_tok_per_sec:.1f}"

            # --- Line 2: Live and Last Request Stats ---
            last_req = session_stats.last_request_stats

            if live_stats and display.status in ["Waiting...", "Streaming"]:
                # Display live stats for the in-progress request
                live_tps = live_stats.live_tok_per_sec
                live_tokens = live_stats.tokens_received
                status_color = (
                    "ansigreen" if display.status == "Streaming" else "ansiyellow"
                )
                live_tps_str = f"<b><style fg='{status_color}'>Live Tokens: {live_tokens}, Live Tok/s: {live_tps:.1f}</style></b>"
                line2_parts = [
                    f"<style fg='ansimagenta'><b>Status: {status_esc}</b></style>",
                    live_tps_str,
                ]
            else:
                # Display stats for the last completed request
                last_tokens_str = f"{last_req.tokens_received}" if last_req else "N/A"
                last_ttft_str = (
                    f"{last_req.ttft:.2f}s" if last_req and last_req.ttft else "N/A"
                )
                last_tps_str = (
                    f"{last_req.final_tok_per_sec:.1f}" if last_req else "N/A"
                )
                last_finish_reason_str = (
                    escape(last_req.finish_reason or "N/A") if last_req else "N/A"
                )
                line2_parts = [
                    f"<style fg='ansimagenta'><b>Status: {status_esc}</b></style>",
                    f"<b>Prev Tokens:</b> {last_tokens_str}",
                    f"<b>Prev TTFT:</b> {last_ttft_str}",
                    f"<b>Prev Tok/s:</b> {last_tps_str}",
                    f"<b>Finish:</b> {last_finish_reason_str}",
                ]

            tools_status = (
                "<style fg='ansigreen'>ON</style>"
                if client.tools_enabled
                else "<style fg='ansired'>OFF</style>"
            )
            debug_status = (
                "<style fg='ansiyellow'>ON</style>" if display.debug_mode else "OFF"
            )
            mode_str = escape(self._get_mode_display_name())

            line3_parts = [
                f"<b>Tools:</b> {tools_status}",
                f"<b>Debug:</b> {debug_status}",
                f"<b>Mode:</b> {mode_str}",
                f"<style fg='grey'>Log: {session_dir_esc}</style>",
            ]

            return HTML(
                f"{line1}\n{' | '.join(p for p in line2_parts if p)}\n{' | '.join(line3_parts)}"
            )
        except Exception as e:
            # If any rendering fails, return a safe, minimal toolbar to prevent crashing.
            return HTML(
                f"<style bg='ansired' fg='white'>[Toolbar Error: {escape(str(e))}]</style>"
            )

    async def run(self):
        """Starts the interactive UI."""
        self.pt_printer(
            f"🎯 {'Chat' if self.is_chat_mode else 'Completion'} Mode | Endpoint: {self.client.config.name} | Model: {self.client.config.model_name}"
        )
        self.pt_printer(f"💾 Session logs will be saved to: {self.session_dir}")
        self.pt_printer("Type '/help' for commands, '/quit' to exit.")
        self.pt_printer("-" * 60)

        # This try/finally ensures that closing stats are printed even if the app
        # exits with an exception (e.g., Ctrl+C/Ctrl+D). The exceptions are
        # allowed to propagate up to main() where they are handled for a clean exit.
        try:
            # run_async() returns the value from app.exit(). The default Ctrl+D
            # handler passes an EOFError. We re-raise it to be caught in main().
            result = await self.app.run_async()
            if isinstance(result, EOFError):
                raise result
        finally:
            closing(self.client.stats, printer=self.pt_printer)


# NEW: typer application replaces main(), async_main(), and argparse
app = typer.Typer(
    name="pai",
    help="🪶 Polyglot AI: A Universal CLI for the OpenAI API Format 🪶",
    add_completion=False,
    no_args_is_help=True,
)


def load_toml_config(path: str) -> PolyglotConfig:
    """Loads and validates the TOML configuration file."""
    try:
        with open(path, encoding="utf-8") as f:
            data = toml.load(f)
            return PolyglotConfig.model_validate(data)
    except FileNotFoundError:
        sys.exit(f"❌ FATAL: Config file not found at '{path}'")
    except Exception as e:
        sys.exit(f"❌ FATAL: Could not parse '{path}': {e}")


async def _run(runtime_config: RuntimeConfig):
    """The core async logic of the application."""
    toml_config = load_toml_config(runtime_config.config)

    # Conditionally load tools only if the --tools flag is active.
    if runtime_config.tools:
        print("🛠️  --tools flag detected. Loading tools...")
        if toml_config.tool_config:
            if tool_dirs := toml_config.tool_config.directories:
                from .tools import load_tools_from_directory

                for tool_dir in tool_dirs:
                    load_tools_from_directory(tool_dir, printer=print)
        else:
            print("  (No 'tool_config' section in polyglot.toml)")

    # Use a single httpx client session for the application's lifecycle
    transport = httpx.AsyncHTTPTransport(retries=3)
    async with httpx.AsyncClient(transport=transport, timeout=30.0) as http_session:
        try:
            client = PolyglotClient(runtime_config, toml_config, http_session)
            if runtime_config.prompt:
                # Non-interactive mode
                if runtime_config.chat:
                    messages = (
                        [{"role": "system", "content": runtime_config.system}]
                        if runtime_config.system
                        else []
                    )
                    messages.append({"role": "user", "content": runtime_config.prompt})
                    request = ChatRequest(
                        messages=messages,
                        model=client.config.model_name,
                        max_tokens=runtime_config.max_tokens,
                        temperature=runtime_config.temperature,
                        stream=runtime_config.stream,
                        tools=get_tool_schemas() if client.tools_enabled else [],
                    )
                else:
                    request = CompletionRequest(
                        prompt=runtime_config.prompt,
                        model=client.config.model_name,
                        max_tokens=runtime_config.max_tokens,
                        temperature=runtime_config.temperature,
                        stream=runtime_config.stream,
                    )
                await client.generate(request, runtime_config.verbose)
                print_stats(client.stats)
            else:
                # Interactive mode
                ui = InteractiveUI(client, runtime_config)
                await ui.run()
        except (APIError, ValueError, httpx.RequestError, ConnectionError) as e:
            # Typer/rich will print a nice error message, no need to print it ourselves.
            raise typer.Exit(code=1) from e


@app.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=True,
)
def run(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Option(
        None, "-p", "--prompt", help="Send a single prompt and exit."
    ),
    chat: bool = typer.Option(False, help="Enable chat mode. Required for tool use."),
    system: Optional[str] = typer.Option(
        None, help="Set a system prompt for chat mode."
    ),
    model: Optional[str] = typer.Option(
        None, help="Override the default model for the session."
    ),
    endpoint: str = typer.Option(
        "openai", help="The name of the endpoint from the config file to use."
    ),
    max_tokens: int = typer.Option(2000, help="Set the max tokens for the response."),
    temperature: float = typer.Option(
        0.7, help="Set the temperature for the response."
    ),
    timeout: Optional[int] = typer.Option(
        None, help="Set the request timeout in seconds for the session."
    ),
    stream: bool = typer.Option(
        True, "--stream/--no-stream", help="Enable/disable streaming."
    ),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable verbose logging of request parameters."
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable raw protocol debug mode for streaming."
    ),
    tools: bool = typer.Option(
        False,
        "--tools",
        help="Enable tool-use capabilities by loading tools from directories in config.",
    ),
    config: str = typer.Option(
        "polyglot.toml", help="Path to the TOML configuration file."
    ),
):
    """Main application entrypoint."""
    print("🪶 Polyglot AI: A Universal CLI for the OpenAI API Format 🪶")
    runtime_config = RuntimeConfig(
        prompt=prompt,
        chat=chat,
        system=system,
        model=model,
        endpoint=endpoint,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        stream=stream,
        verbose=verbose,
        debug=debug,
        tools=tools,
        config=config,
    )
    asyncio.run(_run(runtime_config))


def main():
    try:
        app()
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
